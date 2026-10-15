param(
    [ValidatePattern('^[A-Za-z0-9_.-]+$')][string]$Distro = 'Ubuntu-24.04',
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/[A-Za-z0-9_./-]+$')][string]$ProjectRoot,
    [switch]$SmokeTest,
    [switch]$SmokeDiagnostic,
    [switch]$SmokeCurrentSubjects,
    [switch]$SmokeTrackedPeople,
    [ValidateSet('status','start','stop')][string]$SmokeAction = 'status',
    [string]$ScreenshotPath = ''
)

if (($SmokeCurrentSubjects -or $SmokeTrackedPeople) -and (-not $SmokeTest -or $SmokeDiagnostic -or $SmokeAction -ne 'status' -or ($SmokeCurrentSubjects -and $SmokeTrackedPeople))) {
    throw 'current subject fixture is restricted to status smoke testing'
}

# Windows-only UI adapter. No server, config/credential reads, or model calls.
# Read this UTF-8 source with an explicit encoding under Windows PowerShell 5.1.
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
$script:request = $null
$script:lastReply = $null
$script:lastReplyReceivedTicks = 0
$script:lastReplyTransitSeconds = 0
$script:nextRefresh = [DateTime]::UtcNow
$script:completedActions = 0
$script:lastOk = $false
$wsl = Join-Path $env:WINDIR 'System32\wsl.exe'

$form = New-Object System.Windows.Forms.Form
$form.Text = '宝宝监控 · 本机控制台'
$form.ClientSize = New-Object System.Drawing.Size(940, 740)
$form.MinimumSize = New-Object System.Drawing.Size(920, 770)
$form.StartPosition = 'CenterScreen'
$form.BackColor = [System.Drawing.Color]::FromArgb(245, 247, 251)
$form.Font = New-Object System.Drawing.Font('Microsoft YaHei UI', [single]10, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Point)

function Add-Label([string]$text, [int]$x, [int]$y, [int]$width, [int]$height, [float]$size = 10) {
    $label = New-Object System.Windows.Forms.Label
    $label.Text = $text
    $label.Location = New-Object System.Drawing.Point($x, $y)
    $label.Size = New-Object System.Drawing.Size($width, $height)
    $label.Font = New-Object System.Drawing.Font('Microsoft YaHei UI', [single]$size, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Point)
    $form.Controls.Add($label)
    return $label
}

$title = Add-Label '宝宝监控' 28 22 600 40 22
$subtitle = Add-Label '四路 YOLO 初检 + Bonsai 本地复核   |   不上传家庭画面' 30 69 880 28
$statusLabel = Add-Label '正在读取本机状态…' 30 115 870 40 17
$statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(64, 79, 100)
$detailLabel = Add-Label '仅控制当前本地联调服务，不修改开机启动或生产配置。' 30 158 875 28

function Add-Button([string]$text, [int]$x, [int]$width) {
    $button = New-Object System.Windows.Forms.Button
    $button.Text = $text
    $button.Location = New-Object System.Drawing.Point($x, 200)
    $button.Size = New-Object System.Drawing.Size($width, 42)
    $button.FlatStyle = 'Flat'
    $button.BackColor = [System.Drawing.Color]::White
    $form.Controls.Add($button)
    return $button
}
$startButton = Add-Button '开始运行' 30 150
$startButton.BackColor = [System.Drawing.Color]::FromArgb(39, 94, 215)
$startButton.ForeColor = [System.Drawing.Color]::White
$stopButton = Add-Button '停止运行' 195 150
$refreshButton = Add-Button '刷新状态' 360 135
$diagnosticButton = Add-Button '历史识别诊断' 510 180
$diagnosticButton.Enabled = $false
$busyLabel = Add-Label '' 710 208 200 30

$metricsLabel = Add-Label 'YOLO 检测 —     Bonsai 完成 —     检测 / 复核错误 —' 30 265 880 28
$grid = New-Object System.Windows.Forms.ListView
$grid.Location = New-Object System.Drawing.Point(30, 306)
$grid.Size = New-Object System.Drawing.Size(880, 153)
$grid.View = 'Details'
$grid.FullRowSelect = $true
$grid.GridLines = $false
$grid.HeaderStyle = 'Nonclickable'
$grid.Anchor = 'Top, Left, Right'
foreach ($column in @(@('视频路',90), @('接流',110), @('YOLO采样FPS',130), @('候选框（非人数）',150), @('Bonsai复核',210), @('证据图龄',155))) {
    [void]$grid.Columns.Add([string]$column[0], [int]$column[1])
}
$form.Controls.Add($grid)
$subjectsLabel = Add-Label '逐人诊断（模型推测，未验真）：等待状态' 30 478 880 130
$storageLabel = Add-Label '本地证据存储 —' 30 620 880 26
$warning = Add-Label '识别能力仍在调试：尚不能保证宝宝在哪个房间、是否睡着或安全。' 30 656 880 28
$warning.ForeColor = [System.Drawing.Color]::FromArgb(163, 91, 16)
$footer = Add-Label '每 5 秒刷新  ·  关闭窗口不停止服务；暂停请点「停止运行」。' 30 702 880 24 9

function Render-CurrentSubjects {
    $reply = $script:lastReply
    if (-not $script:lastOk -or $null -eq $reply) {
        $subjectsLabel.Text = '逐人诊断：没有已确认的新鲜状态，旧声称已隐藏。'
        return
    }
    # WSL and Windows wall clocks may differ; age the received snapshot locally.
    $elapsed = ([System.Diagnostics.Stopwatch]::GetTimestamp() - $script:lastReplyReceivedTicks) / [double][System.Diagnostics.Stopwatch]::Frequency + $script:lastReplyTransitSeconds
    if ($elapsed -lt 0 -or $elapsed -gt 10) {
        $subjectsLabel.Text = '逐人诊断：状态刷新已过期，旧声称已隐藏。'
        return
    }
    if ($reply.review_task -notin @('subjects','hybrid')) {
        $subjectsLabel.Text = '逐人诊断：当前未启用此模式。历史诊断不代表当前宝宝位置。'
        return
    }
    if ($reply.status_reason -eq 'local_review_suspended') {
        $subjectsLabel.Text = '逐人诊断：二层正在调试暂停，旧声称已隐藏；YOLO 初筛继续，不提供宝宝状态确认。'
        return
    }
    if ($null -ne $reply.baby_supervision -and $reply.baby_supervision.enabled -eq $true) {
        Render-TrackedPeople $reply.baby_supervision $elapsed
        return
    }
    $lines = @('逐人诊断（模型推测，未验真；不是宝宝状态确认）')
    $ages = @{baby='疑似宝宝'; adult='疑似成人'; unknown='年龄不明'}
    $hintCameras = @()
    $freshCount = 0
    $multipleBaby = $false
    foreach ($camera in $reply.cameras) {
        $probe = $camera.current_subjects
        if ($null -eq $probe -or $probe.semantics -ne 'unverified_model_claims' -or ($probe.evidence_age_seconds + $elapsed) -gt 30) {
            $lines += ([string]$camera.camera_id + '：暂无新鲜逐人结果（可能正在分析、过期或断流）')
            continue
        }
        $text = if (@($probe.age_claims).Count -eq 0) {'模型未报告人物，不代表确定无人'} else { (@($probe.age_claims | ForEach-Object { $ages[[string]$_] }) -join '、') }
        $freshCount++
        $babyCount = @($probe.age_claims | Where-Object { $_ -eq 'baby' }).Count
        if ($babyCount -gt 0) { $hintCameras += [string]$camera.camera_id }
        if ($babyCount -gt 1) { $multipleBaby = $true }
        $stamp = [DateTimeOffset]::Parse([string]$probe.frame_received_at).ToLocalTime().ToString('HH:mm:ss')
        $age = [Math]::Round($probe.evidence_age_seconds + $elapsed, 1)
        if ($probe.update_in_progress -eq $true) { $text += '（上一帧，更新中）' }
        $lines += ([string]$camera.camera_id + "：$text  |  收帧 $stamp，图龄 $age 秒")
    }
    $hintText = if ($hintCameras.Count -eq 0) {'暂无新鲜宝宝线索，不代表宝宝不在画面中'} elseif ($hintCameras.Count -gt 1 -or $multipleBaby) {'宝宝声称存在歧义：' + ($hintCameras -join '、') + '，不能确定位置'} else {'疑似宝宝线索：' + $hintCameras[0] + '（未确认身份或房间）'}
    $lines[0] = $hintText + "；新鲜复核 $freshCount/4 路"
    $subjectsLabel.Text = ($lines -join "`r`n")
}

function Render-TrackedPeople($view, [double]$elapsed) {
    $postures = @{lying='躺卧'; sitting='坐着'; standing='站立'; held='被抱着'; unknown='姿态看不清'}
    $ages = @{baby='疑似宝宝'; adult='成人候选'; unknown='角色待复核'}
    $current = $view.current
    $lines = @()
    if ($null -ne $current -and ($current.detection_age_seconds + $elapsed) -le 2 -and ($current.evidence_age_seconds + $elapsed) -le 30) {
        $where = if ($current.room_name) {$current.room_name} else {$current.camera_id}
        $lines += "宝宝位置（模型估计）：$where；$($postures[[string]$current.visible_posture])，证据 $([Math]::Round($current.evidence_age_seconds + $elapsed,1)) 秒前"
    } elseif ($view.status -eq 'ambiguous') {
        $lines += '宝宝位置：多个候选存在歧义，等待复核'
    } elseif ($view.status -eq 'candidate') {
        $places = @($view.candidates | ForEach-Object {if ($_.room_name) {$_.room_name} else {$_.camera_id}} | Select-Object -Unique)
        $lines += ('宝宝位置候选：' + ($places -join '、') + '；逐人对应待确认')
    } elseif ($null -ne $view.last_seen) {
        $where = if ($view.last_seen.room_name) {$view.last_seen.room_name} else {$view.last_seen.camera_id}
        $stamp = [DateTimeOffset]::Parse([string]$view.last_seen.last_detected_at).ToLocalTime().ToString('HH:mm:ss')
        $lines += "最后看见的宝宝候选：$where（$stamp）；当前尚未确认"
    } else {
        $lines += '宝宝位置：尚未形成有效估计'
    }
    foreach ($camera in $view.cameras) {
        $where = if ($camera.room_name) {$camera.room_name} else {$camera.camera_id}
        $parts = @()
        foreach ($person in $camera.people) {
            $short = ([string]$person.track_id)
            $short = $short.Substring([Math]::Max(0,$short.Length - 6))
            $fresh = $null -ne $person.evidence_age_seconds -and ($person.evidence_age_seconds + $elapsed) -le 30
            $role = if ($fresh) {$ages[[string]$person.age_claim]} else {'角色待复核'}
            $parts += "#$short $role"
        }
        $text = if (-not $camera.available -or ($camera.detection_age_seconds + $elapsed) -gt 2) {'轨迹等待刷新'} elseif ($parts.Count -eq 0) {'暂无可追踪人物；整帧复核继续'} else {($parts -join '、')}
        $lines += "$where：$text"
    }
    $subjectsLabel.Text = ($lines -join "`r`n")
}

function Format-Number($value, [string]$suffix = '') {
    if ($null -eq $value) { return '—' }
    return ([string]$value + $suffix)
}

function Apply-Reply($reply) {
    $script:lastReply = $reply
    $script:lastReplyReceivedTicks = [System.Diagnostics.Stopwatch]::GetTimestamp()
    $diagnosticButton.Enabled = ($null -ne $reply.subject_diagnostic)
    $names = @{running='运行中'; stopped='已停止'; starting='服务已启动，等待实时状态'; partial='部分服务运行'; degraded='运行异常 / 部分能力不可用'}
    $statusLabel.Text = $names[[string]$reply.state]
    if ([string]::IsNullOrEmpty($statusLabel.Text)) { $statusLabel.Text = '状态未知' }
    $statusLabel.ForeColor = if ($reply.state -eq 'running') { [System.Drawing.Color]::FromArgb(25, 120, 77) } elseif ($reply.state -in @('partial','degraded')) { [System.Drawing.Color]::FromArgb(169, 91, 15) } else { [System.Drawing.Color]::FromArgb(64, 79, 100) }
    $gateway = [string]$reply.services.'xiaomi-stream-gateway.service'.state
    $worker = [string]$reply.services.'baby-monitor-yolo-bonsai-live.service'.state
    $phaseNames = @{analyzing='正在复核'; waiting='等待候选'; suspending='等待本次复核结束并卸载'; suspended='调试暂停，模型已卸载'; storage_deferred='存储到限，复核暂停'; memory_deferred='显存不足，复核暂停'; worker_failed='复核线程失败'; unknown='未知'}
    $phase = $phaseNames[[string]$reply.review_phase]
    if ($reply.status_reason -eq 'local_review_suspended' -and @($reply.cameras | Where-Object { $_.receiving }).Count -eq 4) {
        $statusLabel.Text = 'YOLO 检测中；Bonsai 调试暂停'
    }
    if ($reply.review_phase -eq 'storage_deferred') {
        $storageReasons = @{capacity_limit='留证预算到限'; entry_limit='留证条目到限'; disk_space_low='磁盘余量不足'; unsafe_symlink='存储路径安全检查失败'; io_error='存储检查失败'; unknown='存储原因待核对'}
        $storageReason = $storageReasons[[string]$reply.storage_reason]
        if ([string]::IsNullOrEmpty($storageReason)) { $storageReason = '存储原因待核对' }
        $phase = "$storageReason，复核暂停"
        if ($reply.status_reason -eq 'local_review_unavailable' -and @($reply.cameras | Where-Object { $_.receiving }).Count -eq 4) {
            $statusLabel.Text = 'YOLO 检测中；Bonsai 复核暂停'
        }
    }
    if ([string]::IsNullOrEmpty($phase)) { $phase = '—' }
    $detailLabel.Text = "采集网关：$gateway     检测服务：$worker     Bonsai：$phase     心跳距今：$(Format-Number $reply.heartbeat_age_seconds '秒')"
    $metricsLabel.Text = "YOLO $(Format-Number $reply.detected_frames) 张   事件 $(Format-Number $reply.review_successes) 次   逐人 $(Format-Number $reply.subject_probe_successes) 次   错误（检测/事件/逐人）$(Format-Number $reply.detection_failures)/$(Format-Number $reply.review_failures)/$(Format-Number $reply.subject_probe_failures)"
    $reviewNames = @{not_reviewed_this_session='本轮尚未复核'; analyzing='正在复核'; latest_analysis_failed='最新复核失败'; recent_completed_review='近期完成（未验真）'; stale_completed_review='结果已过期'; previous_source_generation='重连前旧结果'; source_generation_unavailable='来源代次未知'; receipt_clock_unavailable='时钟不可用'; source_not_receiving='未接流'; monitor_not_detecting='检测已停止'; local_review_suspended='二层调试暂停，旧声称隐藏'; attempt_unavailable='尝试无效'; unknown='无可用结果'}
    $grid.BeginUpdate()
    $grid.Items.Clear()
    foreach ($camera in $reply.cameras) {
        $item = New-Object System.Windows.Forms.ListViewItem([string]$camera.camera_id)
        [void]$item.SubItems.Add($(if ($camera.receiving) {'正在接流'} else {'未接流'}))
        [void]$item.SubItems.Add((Format-Number $camera.sample_fps))
        [void]$item.SubItems.Add((Format-Number $camera.candidate_boxes))
        [void]$item.SubItems.Add([string]$reviewNames[[string]$camera.review_status])
        [void]$item.SubItems.Add((Format-Number $camera.evidence_age_seconds '秒'))
        [void]$grid.Items.Add($item)
    }
    if ($grid.Items.Count -eq 0) {
        foreach ($number in 1..4) {
            $item = New-Object System.Windows.Forms.ListViewItem(('camera_{0:00}' -f $number))
            foreach ($text in @('未确认','—','—','无本轮实时结果','—')) { [void]$item.SubItems.Add($text) }
            [void]$grid.Items.Add($item)
        }
    }
    $grid.EndUpdate()
    if ($null -eq $reply.storage_used_bytes -or $null -eq $reply.storage_limit_bytes) { $storageLabel.Text = '本地证据存储：尚无本轮状态' }
    else {
        $used = [Math]::Round($reply.storage_used_bytes / 1GB, 2)
        $limit = [Math]::Round($reply.storage_limit_bytes / 1GB, 2)
        $storageLabel.Text = "本地证据存储：$used / $limit GiB    不自动删除或扩容"
        if ($reply.storage_scan_complete -eq $false) { $storageLabel.Text = "本地证据存储：至少 $used GiB（扫描未完成）    不自动删除或扩容" }
        elseif ($null -ne $reply.storage_writable_budget_bytes) {
            $remaining = [Math]::Round($reply.storage_writable_budget_bytes / 1MB, 1)
            $storageLabel.Text += "；扣除预留后剩余 $remaining MiB"
        }
    }
}

function Show-SubjectDiagnostic {
    $diagnostic = $script:lastReply.subject_diagnostic
    if ($null -eq $diagnostic -or $diagnostic.scope -ne 'historical_model_claims') { return }
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('历史调试结果 — 不是当前宝宝位置或状态')
    $lines.Add('以下均为模型推测，未经身份验真；空间对应不证明识别正确。')
    $lines.Add(('视频路：' + [string]$diagnostic.camera_id))
    $lines.Add('')
    $ages = @{baby='疑似宝宝'; adult='疑似成人'; unknown='年龄不明'}
    $matches = @{unique_spatial_match='头点与头框唯一对应'; no_head_match='第一层无对应头框'; head_not_located='未定位头部'; ambiguous_head_match='头框对应有歧义'}
    foreach ($row in $diagnostic.rows) {
        $stamp = [DateTimeOffset]::Parse([string]$row.received_at).ToLocalTime().ToString('yyyy-MM-dd HH:mm:ss zzz')
        $lines.Add("收帧时间：$stamp（本机收帧，非曝光时间）")
        $lines.Add(('证据距上次刷新约 ' + [Math]::Round($row.age_seconds / 60, 1) + ' 分钟'))
        if (@($row.subjects).Count -eq 0) { $lines.Add('  模型没有报告人物，不代表已确认无人。') }
        foreach ($subject in $row.subjects) {
            $lines.Add(('  模型推测：' + $ages[[string]$subject.age_claim] + '；' + $matches[[string]$subject.spatial_status]))
        }
        $lines.Add('')
    }
    $dialog = New-Object System.Windows.Forms.Form
    $dialog.Text = '历史识别诊断 · 未验真'
    $dialog.ClientSize = New-Object System.Drawing.Size(790, 540)
    $dialog.StartPosition = 'CenterParent'
    $box = New-Object System.Windows.Forms.TextBox
    $box.Multiline = $true
    $box.ReadOnly = $true
    $box.ScrollBars = 'Vertical'
    $box.Dock = 'Fill'
    $box.Font = $form.Font
    $box.Text = ($lines -join "`r`n")
    $dialog.Controls.Add($box)
    $dialog.add_Shown({ $box.Select(0, 0) })
    try {
        if ($SmokeTest -and $SmokeDiagnostic) {
            $dialog.ShowInTaskbar = $false
            $dialog.Opacity = 0
            $dialog.Show()
            [System.Windows.Forms.Application]::DoEvents()
            if ($ScreenshotPath) {
                if (Test-Path -LiteralPath $ScreenshotPath) { throw 'existing screenshot preserved' }
                $bitmap = New-Object System.Drawing.Bitmap($dialog.Width, $dialog.Height)
                try {
                    $dialog.DrawToBitmap($bitmap, (New-Object System.Drawing.Rectangle(0, 0, $dialog.Width, $dialog.Height)))
                    $bitmap.Save($ScreenshotPath, [System.Drawing.Imaging.ImageFormat]::Png)
                } finally { $bitmap.Dispose() }
            }
        } else { [void]$dialog.ShowDialog($form) }
    } finally { $dialog.Dispose() }
}

function Start-Keeper {
    Start-Process -FilePath $wsl -WindowStyle Hidden -ArgumentList @('-d',$Distro,'--cd',$ProjectRoot,'--','.venv/bin/python','scripts/hold_wsl_live_session.py') -ErrorAction Stop | Out-Null
}

function Begin-Action([string]$action) {
    if ($null -ne $script:request) { return }
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $wsl
    $info.Arguments = "-d $Distro --cd $ProjectRoot -- .venv/bin/python scripts/live_desktop_backend.py --action $action"
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $info.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    try {
        if ($action -eq 'start') { Start-Keeper }
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $info
        [void]$process.Start()
        $script:request = @{process=$process; output=$process.StandardOutput.ReadToEndAsync(); error=$process.StandardError.ReadToEndAsync(); action=$action; began=[DateTime]::UtcNow; beganTicks=[System.Diagnostics.Stopwatch]::GetTimestamp()}
        $startButton.Enabled = $false
        $stopButton.Enabled = $false
        $refreshButton.Enabled = $false
        $busyLabel.Text = @{status='读取状态…';start='正在启动（等待模型加载）…';stop='正在停止（等待模型退出）…'}[$action]
    } catch {
        $statusLabel.Text = '无法连接本机 WSL'
        $detailLabel.Text = '请确认 Ubuntu-24.04 和项目路径可用；未回显底层错误。'
        $script:lastOk = $false
        $script:completedActions++
        $interval = if ($null -ne $script:lastReply.baby_supervision) {1} else {5}
        $script:nextRefresh = [DateTime]::UtcNow.AddSeconds($interval)
    }
}

function Tick-Panel {
    Render-CurrentSubjects
    if ($null -eq $script:request) {
        if (-not $SmokeTest -and [DateTime]::UtcNow -ge $script:nextRefresh) { Begin-Action 'status' }
        return
    }
    $request = $script:request
    $limit = if ($request.action -eq 'status') { 40 } else { 300 }
    if (([DateTime]::UtcNow - $request.began).TotalSeconds -gt $limit) {
        if (-not $request.process.HasExited) { $request.process.Kill() }
        $statusLabel.Text = '操作超时，服务状态待确认'
        $detailLabel.Text = '没有确认开始或停止成功；请刷新状态。'
        $script:lastOk = $false
    } elseif (-not $request.process.HasExited -or -not $request.output.IsCompleted -or -not $request.error.IsCompleted) { return }
    else {
        try {
            $text = $request.output.Result
            if ($text.Length -gt 524288) { throw 'bounded output required' }
            $reply = $text | ConvertFrom-Json
            if ($request.process.ExitCode -ne 0 -or $reply.ok -ne $true -or $reply.version -ne 1) { throw 'unconfirmed action' }
            $script:lastReplyTransitSeconds = ([System.Diagnostics.Stopwatch]::GetTimestamp() - $request.beganTicks) / [double][System.Diagnostics.Stopwatch]::Frequency
            Apply-Reply $reply
            $script:lastOk = $true
            Render-CurrentSubjects
        } catch {
            $statusLabel.Text = '操作未确认 / 无法读取状态'
            $detailLabel.Text = '不把失败当作运行或停止成功；未显示底层输出。'
            $script:lastOk = $false
        }
    }
    $request.process.Dispose()
    $script:request = $null
    $script:completedActions++
    $interval = if ($null -ne $script:lastReply.baby_supervision) {1} else {5}
    $script:nextRefresh = [DateTime]::UtcNow.AddSeconds($interval)
    $busyLabel.Text = ''
    $startButton.Enabled = $true
    $stopButton.Enabled = $true
    $refreshButton.Enabled = $true
}

$startButton.add_Click({ Begin-Action 'start' })
$stopButton.add_Click({ Begin-Action 'stop' })
$refreshButton.add_Click({ Begin-Action 'status' })
$diagnosticButton.add_Click({ Show-SubjectDiagnostic })
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 200
$timer.add_Tick({ Tick-Panel })
$form.add_FormClosing({
    param($sender, $eventArgs)
    if ($null -ne $script:request -and $script:request.action -ne 'status') {
        $eventArgs.Cancel = $true
        $busyLabel.Text = '请等待当前控制操作结束再关闭窗口。'
    }
})

try {
    if ($SmokeTest) {
        $form.ShowInTaskbar = $false
        $form.Opacity = 0
        $form.Show()
        switch ($SmokeAction) { start {$startButton.PerformClick()} stop {$stopButton.PerformClick()} default {$refreshButton.PerformClick()} }
        $deadline = [DateTime]::UtcNow.AddSeconds(310)
        while ($script:completedActions -eq 0 -and [DateTime]::UtcNow -lt $deadline) {
            Tick-Panel
            [System.Windows.Forms.Application]::DoEvents()
            Start-Sleep -Milliseconds 100
        }
        if (-not $script:lastOk) { throw 'desktop smoke action not confirmed' }
        if ($SmokeCurrentSubjects) {
            # Synthetic UI-only fixture: never saved to backend or used as a live claim.
            $form.Text = '合成 UI 测试 · 不是家庭识别结果'
            $warning.Text = '合成 UI 测试数据：窗口中的人物类别不是本次家庭画面的识别结果。'
            $script:lastReply | Add-Member -NotePropertyName ui_fixture -NotePropertyValue $true
            $script:lastReply | Add-Member -NotePropertyName baby_supervision -NotePropertyValue $null -Force
            $script:lastReply.review_task = 'subjects'
            $script:lastReply.checked_at = [DateTimeOffset]::UtcNow.ToString('o')
            $script:lastReplyTransitSeconds = 0
            $script:lastReplyReceivedTicks = [System.Diagnostics.Stopwatch]::GetTimestamp()
            $index = 0
            foreach ($camera in $script:lastReply.cameras) {
                $camera.current_subjects = @{semantics='unverified_model_claims';frame_received_at=[DateTimeOffset]::UtcNow.AddSeconds(-8).ToString('o');evidence_age_seconds=8;age_claims=@('baby','adult','unknown')}
                if ($index -eq 1) { $camera.current_subjects.age_claims = @() }
                if ($index -eq 2) { $camera.current_subjects = $null }
                if ($index -eq 3) { $camera.current_subjects.evidence_age_seconds = 31 }
                $index++
            }
            Render-CurrentSubjects
            if (-not $subjectsLabel.Text.Contains('疑似宝宝') -or -not $subjectsLabel.Text.Contains('不代表确定无人') -or -not $subjectsLabel.Text.Contains('暂无新鲜')) { throw 'subject fixture render failed' }
            if (-not $subjectsLabel.Text.Contains('新鲜复核 2/4 路')) { throw 'fresh coverage summary failed' }
            $script:lastReply.cameras[0].current_subjects.update_in_progress = $true
            Render-CurrentSubjects
            if (-not $subjectsLabel.Text.Contains('上一帧，更新中')) { throw 'refreshing evidence label missing' }
            $script:lastReply.cameras[0].current_subjects.update_in_progress = $false
            $script:lastReply.cameras[0].current_subjects.age_claims = @('baby','baby')
            Render-CurrentSubjects
            if (-not $subjectsLabel.Text.Contains('存在歧义')) { throw 'same-camera ambiguity summary failed' }
            $script:lastReply.cameras[0].current_subjects.age_claims = @('baby','adult','unknown')
            $script:lastReply.cameras[1].current_subjects.age_claims = @('baby')
            Render-CurrentSubjects
            if (-not $subjectsLabel.Text.Contains('存在歧义')) { throw 'multi-camera ambiguity summary failed' }
            $script:lastReply.cameras[1].current_subjects.age_claims = @()
            $script:lastReply.checked_at = '2000-01-01T00:00:00Z'
            Render-CurrentSubjects
            if (-not $subjectsLabel.Text.Contains('疑似宝宝')) { throw 'remote wall clock incorrectly aged local snapshot' }
            $script:lastReplyReceivedTicks = [System.Diagnostics.Stopwatch]::GetTimestamp() - 11 * [System.Diagnostics.Stopwatch]::Frequency
            Render-CurrentSubjects
            if ($subjectsLabel.Text.Contains('疑似宝宝') -or -not $subjectsLabel.Text.Contains('过期')) { throw 'stale subject fixture was retained' }
            $script:lastReply.checked_at = [DateTimeOffset]::UtcNow.ToString('o')
            $script:lastReplyReceivedTicks = [System.Diagnostics.Stopwatch]::GetTimestamp()
            $script:lastOk = $false
            Render-CurrentSubjects
            if ($subjectsLabel.Text.Contains('疑似宝宝')) { throw 'failed refresh retained subject fixture' }
            $script:lastOk = $true
            Render-CurrentSubjects
        }
        if ($SmokeTrackedPeople) {
            $form.Text = '合成 UI 测试 · 不是家庭识别结果'
            $warning.Text = '合成 UI 测试数据：人物轨迹、房间和姿态均为测试样例。'
            $script:lastReply.review_task = 'subjects'
            $script:lastReplyTransitSeconds = 0
            $script:lastReplyReceivedTicks = [System.Diagnostics.Stopwatch]::GetTimestamp()
            $person = @{track_id='track_0123456789'; age_claim='baby'; role_status='supported'; visible_posture='lying'; evidence_age_seconds=8}
            $current = @{camera_id='camera_04'; room_name='测试婴儿床区域'; visible_posture='lying'; evidence_age_seconds=8; detection_age_seconds=0.1}
            $view = @{enabled=$true;status='current_model_estimate';current=$current;last_seen=@{room_name='测试婴儿床区域';last_detected_at=[DateTimeOffset]::UtcNow.AddSeconds(-8).ToString('o')};cameras=@()}
            foreach ($id in @('camera_01','camera_02','camera_03','camera_04')) {
                $view.cameras += @{camera_id=$id;room_name=$id;available=$true;detection_age_seconds=0.1;people=$(if ($id -eq 'camera_04') {@($person)} else {@()})}
            }
            $script:lastReply | Add-Member -NotePropertyName baby_supervision -NotePropertyValue $view -Force
            Render-CurrentSubjects
            if (-not $subjectsLabel.Text.Contains('躺卧') -or -not $subjectsLabel.Text.Contains('#456789')) {throw 'tracked people render failed'}
            $view.current = $null
            $view.status = 'ambiguous'
            Render-CurrentSubjects
            if (-not $subjectsLabel.Text.Contains('歧义')) {throw 'tracked ambiguity render failed'}
            $view.status = 'last_seen'
            Render-CurrentSubjects
            if (-not $subjectsLabel.Text.Contains('最后看见')) {throw 'last seen render failed'}
            $view.current = $current
            $view.status = 'current_model_estimate'
            Render-CurrentSubjects
        }
        if ($SmokeDiagnostic) {
            if (-not $diagnosticButton.Enabled) { throw 'no historical diagnostic to display' }
            $diagnosticButton.PerformClick()
        }
        if ($ScreenshotPath -and -not $SmokeDiagnostic) {
            if (Test-Path -LiteralPath $ScreenshotPath) { throw 'existing screenshot preserved' }
            $bitmap = New-Object System.Drawing.Bitmap($form.Width, $form.Height)
            try {
                $rectangle = New-Object System.Drawing.Rectangle(0, 0, $form.Width, $form.Height)
                $form.DrawToBitmap($bitmap, $rectangle)
                $bitmap.Save($ScreenshotPath, [System.Drawing.Imaging.ImageFormat]::Png)
            } finally { $bitmap.Dispose() }
        }
        [Console]::WriteLine(($script:lastReply | ConvertTo-Json -Depth 8 -Compress))
    } else {
        $timer.Start()
        [void]$form.ShowDialog()
    }
} finally {
    $timer.Stop()
    $timer.Dispose()
    if ($null -ne $script:request -and $script:request.action -eq 'status' -and -not $script:request.process.HasExited) { $script:request.process.Kill() }
    $form.Dispose()
}
