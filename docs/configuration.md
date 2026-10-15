# 配置参考

本页由 `scripts/generate_config_reference.py` 从 `baby_monitor.config` 的实现自动生成，请勿手工编辑。

## 配置加载顺序

1. 代码内置默认值（本页表格中的默认列）；
2. TOML 配置文件：CLI `--config` 指定，缺省 `config/default.toml`；
3. 环境变量覆盖：仅表格中列出环境变量列的设置支持，前缀固定为 `BABY_MONITOR_`。

层级只增不改：环境变量只覆盖同名 TOML 键，TOML 只覆盖内置默认值。

## 隐私约定

- `config/*.example.*` 是可公开的示例；真实部署使用 Git 忽略的`*.local.*` 文件或私有路径。
- 密码、token、API key、RTSP 凭据只允许存在于环境变量或仓库外 `0600` 私有文件，禁止写入任何被跟踪文件。
- `release-audit` 会在发布前扫描凭据、私有媒体扩展名和机器专属 home 路径。

## 设置项

### `[general]`

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `timezone` | str | `'Asia/Shanghai'` | — |

### `[storage]`

本地/NAS 存储与保留期；删除类开关默认必须保持 false。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `local_root` | Path | `data` | `BABY_MONITOR_LOCAL_ROOT` |
| `nas_root` | Path \| None | `None` | `BABY_MONITOR_NAS_ROOT` |
| `prefer_nas` | bool | `False` | `BABY_MONITOR_PREFER_NAS` |
| `database` | Path | `data/events.sqlite3` | `BABY_MONITOR_DATABASE` |
| `retention_days` | int | `14` | `BABY_MONITOR_RETENTION_DAYS` |
| `retention_scan_enabled` | bool | `False` | `BABY_MONITOR_RETENTION_SCAN_ENABLED` |
| `retention_delete_enabled` | bool | `False` | `BABY_MONITOR_RETENTION_DELETE_ENABLED` |

### `[camera]`

默认摄像头适配器；实时流由部署 profile 显式提供。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `camera_id` | str | `'nursery_cam_01'` | `BABY_MONITOR_CAMERA_ID` |
| `kind` | str | `'file'` | `BABY_MONITOR_CAMERA_KIND` |
| `source` | str | `''` | `BABY_MONITOR_CAMERA_SOURCE` |

### `[detector]`

低成本检测器；YOLO 需要本地权重与已复核 manifest，禁止自动下载。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `backend` | str | `'mock'` | `BABY_MONITOR_DETECTOR_BACKEND` |
| `yolo_model_path` | Path \| None | `None` | `BABY_MONITOR_YOLO_MODEL_PATH` |
| `yolo_manifest_path` | Path \| None | `None` | `BABY_MONITOR_YOLO_MANIFEST_PATH` |
| `yolo_runner_command` | tuple[str, ...] | `()` | — |
| `yolo_runner_timeout_seconds` | float | `600.0` | `BABY_MONITOR_YOLO_RUNNER_TIMEOUT_SECONDS` |
| `yolo_execution_lock_path` | Path \| None | `None` | `BABY_MONITOR_YOLO_EXECUTION_LOCK_PATH` |
| `yolo_execution_lock_timeout_seconds` | float | `600.0` | `BABY_MONITOR_YOLO_EXECUTION_LOCK_TIMEOUT_SECONDS` |
| `device` | str | `'0'` | `BABY_MONITOR_YOLO_DEVICE` |
| `confidence` | float | `0.35` | `BABY_MONITOR_YOLO_CONFIDENCE` |
| `sample_fps` | float | `1.0` | `BABY_MONITOR_SAMPLE_FPS` |
| `image_size` | int | `640` | `BABY_MONITOR_YOLO_IMAGE_SIZE` |
| `rotation_degrees` | int | `0` | `BABY_MONITOR_ROTATION_DEGREES` |
| `candidate_gap_seconds` | float | `3.0` | — |
| `scene_threshold` | float | `0.15` | `BABY_MONITOR_SCENE_THRESHOLD` |
| `scene_max_detections` | int | `200` | `BABY_MONITOR_SCENE_MAX_DETECTIONS` |
| `labels` | tuple[str, ...] | `('person', 'cat', 'dog')` | — |

### `[media]`

事件窗口、关键帧数量与 FFmpeg 可执行文件。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `pre_event_seconds` | float | `5.0` | `BABY_MONITOR_PRE_EVENT_SECONDS` |
| `post_event_seconds` | float | `7.0` | `BABY_MONITOR_POST_EVENT_SECONDS` |
| `keyframe_count` | int | `5` | `BABY_MONITOR_KEYFRAME_COUNT` |
| `ffmpeg` | str | `'ffmpeg'` | — |
| `ffprobe` | str | `'ffprobe'` | — |

### `[buffer]`

循环缓冲目录、分段与重连参数；淘汰必须双重确认。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `root` | Path | `data/buffer` | `BABY_MONITOR_BUFFER_ROOT` |
| `segment_seconds` | float | `2.0` | `BABY_MONITOR_BUFFER_SEGMENT_SECONDS` |
| `max_segments` | int | `30` | `BABY_MONITOR_BUFFER_MAX_SEGMENTS` |
| `reconnect_initial_seconds` | float | `0.5` | — |
| `reconnect_max_seconds` | float | `8.0` | — |
| `poll_interval_seconds` | float | `0.25` | — |
| `eviction_enabled` | bool | `False` | `BABY_MONITOR_BUFFER_EVICTION_ENABLED` |

### `[media_health]`

媒体健康快照的新鲜度判定。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `stale_after_seconds` | float | `10.0` | `BABY_MONITOR_MEDIA_HEALTH_STALE_SECONDS` |
| `future_tolerance_seconds` | float | `5.0` | `BABY_MONITOR_MEDIA_HEALTH_FUTURE_TOLERANCE_SECONDS` |

### `[candidate_worker]`

有界候选扫描 worker 的运行限制与候选启动门限。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `max_segments_per_run` | int | `32` | `BABY_MONITOR_CANDIDATE_MAX_SEGMENTS_PER_RUN` |
| `idle_exit_seconds` | float | `30.0` | `BABY_MONITOR_CANDIDATE_IDLE_EXIT_SECONDS` |
| `max_runtime_seconds` | float | `3600.0` | `BABY_MONITOR_CANDIDATE_MAX_RUNTIME_SECONDS` |
| `poll_seconds` | float | `1.0` | `BABY_MONITOR_CANDIDATE_POLL_SECONDS` |
| `max_event_span_seconds` | float | `30.0` | `BABY_MONITOR_CANDIDATE_MAX_EVENT_SPAN_SECONDS` |
| `candidate_start_confidence` | float | `0.35` | `BABY_MONITOR_CANDIDATE_START_CONFIDENCE` |
| `max_pending_detections` | int | `1000` | `BABY_MONITOR_CANDIDATE_MAX_PENDING_DETECTIONS` |
| `detector_timeout_seconds` | float | `120.0` | `BABY_MONITOR_CANDIDATE_DETECTOR_TIMEOUT_SECONDS` |

### `[analysis_queue]`

持久分析队列容量、租约与重试。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `max_pending` | int | `32` | `BABY_MONITOR_ANALYSIS_QUEUE_MAX_PENDING` |
| `lease_seconds` | int | `300` | `BABY_MONITOR_ANALYSIS_QUEUE_LEASE_SECONDS` |
| `retry_delay_seconds` | int | `30` | `BABY_MONITOR_ANALYSIS_QUEUE_RETRY_DELAY_SECONDS` |
| `max_attempts` | int | `3` | `BABY_MONITOR_ANALYSIS_QUEUE_MAX_ATTEMPTS` |

### `[analysis_worker]`

顺序分析 worker 生命周期与可选 GPU 门槛。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `max_jobs_per_run` | int | `32` | `BABY_MONITOR_ANALYSIS_MAX_JOBS_PER_RUN` |
| `idle_exit_seconds` | float | `30.0` | `BABY_MONITOR_ANALYSIS_IDLE_EXIT_SECONDS` |
| `max_runtime_seconds` | float | `3600.0` | `BABY_MONITOR_ANALYSIS_MAX_RUNTIME_SECONDS` |
| `poll_seconds` | float | `2.0` | `BABY_MONITOR_ANALYSIS_POLL_SECONDS` |
| `gpu_guard_enabled` | bool | `False` | `BABY_MONITOR_ANALYSIS_GPU_GUARD_ENABLED` |
| `gpu_device` | str | `'0'` | `BABY_MONITOR_ANALYSIS_GPU_DEVICE` |
| `min_free_vram_mib` | int | `6000` | `BABY_MONITOR_ANALYSIS_MIN_FREE_VRAM_MIB` |
| `max_gpu_utilization_percent` | int | `20` | `BABY_MONITOR_ANALYSIS_MAX_GPU_UTILIZATION_PERCENT` |
| `gpu_probe_timeout_seconds` | int | `5` | — |

### `[archive_queue]`

归档队列容量、租约与重试。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `max_pending` | int | `64` | `BABY_MONITOR_ARCHIVE_QUEUE_MAX_PENDING` |
| `lease_seconds` | int | `3600` | `BABY_MONITOR_ARCHIVE_QUEUE_LEASE_SECONDS` |
| `retry_delay_seconds` | int | `300` | `BABY_MONITOR_ARCHIVE_QUEUE_RETRY_DELAY_SECONDS` |
| `max_attempts` | int | `20` | `BABY_MONITOR_ARCHIVE_QUEUE_MAX_ATTEMPTS` |

### `[archive_worker]`

顺序归档 worker 生命周期。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `max_jobs_per_run` | int | `32` | `BABY_MONITOR_ARCHIVE_MAX_JOBS_PER_RUN` |
| `idle_exit_seconds` | float | `30.0` | `BABY_MONITOR_ARCHIVE_IDLE_EXIT_SECONDS` |
| `max_runtime_seconds` | float | `3600.0` | `BABY_MONITOR_ARCHIVE_MAX_RUNTIME_SECONDS` |
| `poll_seconds` | float | `5.0` | `BABY_MONITOR_ARCHIVE_POLL_SECONDS` |

### `[local_vlm]`

主本地 VLM 后端；同一时间只允许一个 VLM 驻留。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `backend` | str | `'mock'` | `BABY_MONITOR_LOCAL_VLM_BACKEND` |
| `name` | str | `'mock'` | `BABY_MONITOR_LOCAL_VLM_NAME` |
| `revision` | str | `''` | `BABY_MONITOR_LOCAL_VLM_REVISION` |
| `command` | tuple[str, ...] | `()` | — |
| `timeout_seconds` | int | `180` | `BABY_MONITOR_LOCAL_VLM_TIMEOUT_SECONDS` |
| `confidence_threshold` | float | `0.65` | `BABY_MONITOR_LOCAL_CONFIDENCE_THRESHOLD` |
| `gpu_device` | str | `'0'` | `BABY_MONITOR_LOCAL_VLM_GPU_DEVICE` |

### `[review_policy]`

复核升级策略；频率只提升复核层级。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `signal_frequency_window_seconds` | int | `600` | `BABY_MONITOR_SIGNAL_FREQUENCY_WINDOW_SECONDS` |
| `signal_frequency_secondary_threshold` | int | `3` | `BABY_MONITOR_SIGNAL_FREQUENCY_SECONDARY_THRESHOLD` |
| `priority_signal_types` | tuple[str, ...] | `('fall_hint')` | — |

### `[secondary_vlm]`

可选第二本地复核器；必须在主模型之后顺序加载。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `enabled` | bool | `False` | `BABY_MONITOR_SECONDARY_VLM_ENABLED` |
| `backend` | str | `'mock'` | `BABY_MONITOR_SECONDARY_VLM_BACKEND` |
| `name` | str | `'secondary_mock'` | `BABY_MONITOR_SECONDARY_VLM_NAME` |
| `revision` | str | `''` | `BABY_MONITOR_SECONDARY_VLM_REVISION` |
| `command` | tuple[str, ...] | `()` | — |
| `timeout_seconds` | int | `180` | `BABY_MONITOR_SECONDARY_VLM_TIMEOUT_SECONDS` |
| `gpu_device` | str | `'0'` | `BABY_MONITOR_SECONDARY_VLM_GPU_DEVICE` |

### `[remote_vlm]`

远端复核；功能开关与上传同意缺一不可，默认全关。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `enabled` | bool | `False` | `BABY_MONITOR_REMOTE_ENABLED` |
| `upload_consent` | bool | `False` | `BABY_MONITOR_REMOTE_UPLOAD_CONSENT` |
| `base_url` | str | `''` | `BABY_MONITOR_REMOTE_BASE_URL` |
| `model` | str | `''` | `BABY_MONITOR_REMOTE_MODEL` |
| `api_key_env` | str | `'BABY_MONITOR_REMOTE_API_KEY'` | `BABY_MONITOR_REMOTE_API_KEY_ENV` |
| `max_frames` | int | `3` | `BABY_MONITOR_REMOTE_MAX_FRAMES` |
| `timeout_seconds` | int | `60` | `BABY_MONITOR_REMOTE_TIMEOUT_SECONDS` |
| `max_upload_bytes` | int | `5242880` | `BABY_MONITOR_REMOTE_MAX_UPLOAD_BYTES` |
| `max_response_bytes` | int | `262144` | `BABY_MONITOR_REMOTE_MAX_RESPONSE_BYTES` |
| `max_requests_per_day` | int | `20` | `BABY_MONITOR_REMOTE_MAX_REQUESTS_PER_DAY` |
| `min_interval_seconds` | int | `60` | `BABY_MONITOR_REMOTE_MIN_INTERVAL_SECONDS` |

### `[identity]`

身份能力总开关；默认全部关闭。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `enabled` | bool | `False` | `BABY_MONITOR_IDENTITY_ENABLED` |
| `biometric_enrollment_consent` | bool | `False` | `BABY_MONITOR_BIOMETRIC_ENROLLMENT_CONSENT` |
| `event_matching_enabled` | bool | `False` | `BABY_MONITOR_IDENTITY_EVENT_MATCHING_ENABLED` |
| `external_identity_disclosure_enabled` | bool | `False` | `BABY_MONITOR_IDENTITY_EXTERNAL_DISCLOSURE_ENABLED` |

### `[notification]`

通知 outbox 与外部投递；外部投递独立开关且默认关闭。

| TOML 键 | 类型 | 默认值 | 环境变量 |
|---|---|---|---|
| `enabled` | bool | `False` | `BABY_MONITOR_NOTIFICATION_ENABLED` |
| `external_delivery_enabled` | bool | `False` | `BABY_MONITOR_NOTIFICATION_EXTERNAL_DELIVERY_ENABLED` |
| `auto_file_enabled` | bool | `False` | `BABY_MONITOR_NOTIFICATION_AUTO_FILE_ENABLED` |
| `auto_file_sink_id` | str | `'shadow_local'` | `BABY_MONITOR_NOTIFICATION_AUTO_FILE_SINK_ID` |
| `event_policy_path` | Path \| None | `None` | `BABY_MONITOR_NOTIFICATION_EVENT_POLICY_PATH` |
| `max_pending` | int | `64` | `BABY_MONITOR_NOTIFICATION_MAX_PENDING` |
| `max_pending_age_seconds` | int | `3600` | `BABY_MONITOR_NOTIFICATION_MAX_PENDING_AGE_SECONDS` |
| `lease_seconds` | int | `300` | `BABY_MONITOR_NOTIFICATION_LEASE_SECONDS` |
| `retry_delay_seconds` | int | `30` | `BABY_MONITOR_NOTIFICATION_RETRY_DELAY_SECONDS` |
| `max_attempts` | int | `5` | `BABY_MONITOR_NOTIFICATION_MAX_ATTEMPTS` |

## `config/` 示例文件

| 文件 | 用途 |
|---|---|
| `default.toml` | 核心 TOML 默认值（本参考的数据来源之一） |
| `ha-signal-routes.example.json` | HA 信号到本地媒体路由的私有绑定示例 |
| `home-assistant-signals.example.json` | HA entity 到 SignalEvent 的显式映射示例 |
| `household-event-policy.example.json` | 家庭事件包路由策略示例（规则引擎输入） |
| `iou-tracker.example.json` | IoU 跟踪器配置示例 |
| `mediamtx.loopback.yml` | 回环媒体路由器配置 |
| `motion-two-stage-tracker.example.json` | motion-two-stage 跟踪器配置示例 |
| `notification-smtp.example.json` | SMTP 外部投递私有 profile 示例 |
| `notification-webhook.example.json` | HTTPS webhook 私有 profile 示例 |
| `opencv-sface-2021dec.reviewed.example.json` | SFace 人脸模型 manifest 示例 |
| `opencv-yunet-2023mar.reviewed.example.json` | YuNet 人脸检测模型 manifest 示例 |
| `room-registry.example.json` | 房间/物理摄像头/逻辑流注册表示例 |
| `temporal-zones.example.json` | 时序区域配置示例 |
| `temporal-zones.person-public-smoke.example.json` | 公开 smoke 用的区域配置 |
| `temporal-zones.scene-smoke.example.json` | 场景检测 smoke 用的区域配置 |
| `vlm-ab.example.toml` | VLM A/B 评测配置示例 |
| `yolo-model.example.json` | YOLO 权重 manifest 示例 |
| `yolo11n-coco.reviewed.example.json` | 已复核 YOLO11n 权重 manifest |
| `yolo11n-pose-coco.reviewed.example.json` | 已复核 YOLO11n-Pose 权重 manifest |
| `yolo11n-subprocess.example.toml` | YOLO 子进程运行配置示例 |
| `yolo26m-coco.reviewed.example.json` | 已复核 YOLO26m 权重 manifest |
| `yolo26m-pose-coco.reviewed.example.json` | 已复核 YOLO26m-Pose 权重 manifest |

## 私有运行器环境变量

模型运行器还有少量独立便利变量（如 `BABY_MONITOR_MAGE_MODEL_PATH`、`BABY_MONITOR_MINICPM_MODEL_PATH`、`BABY_MONITOR_SMOLVLM2_MODEL_PATH`、`BABY_MONITOR_MAGE_CODE_MANIFEST`），只被对应 runner 读取，不参与核心 Settings；路径属于私有部署信息，不应提交。
