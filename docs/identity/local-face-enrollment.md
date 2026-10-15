# 本地人脸录入与身份推断设计

状态：2026-08-11，原子录入、只读列举和精确删除 CLI 已实现并完成本地安全回归；尚未实际录入任何家庭成员。

## 当前实现边界

- 身份能力默认关闭。除 consent/enrollment/deletion preflight 外，产品 CLI 已提供 `identity-enroll`、
  `identity-profiles` 与 `identity-delete-profile`。实际操作仍要求持久开关和独立的一次性确认；未授权时在
  模型、照片、manifest 和模板库 I/O 前失败。
- 内部已具备严格 FaceEmbeddingBatch / IdentityAssessment 契约、保守 open-set matcher、
  AES-256-GCM SQLite 模板库原语，以及固定 YuNet 2023mar + SFace 2021dec 的本地 runner。
  录入以 3～12 张单脸样本一次事务写入；列举不加载 embedding/密钥、不创建或迁移库；删除使模板快照
  失效，并区分“事务已提交”和“WAL 清理待重试”。
- 三份经准入的 NASA 成人公开图片已在 Git 外完成质量门 smoke：3 cases、检测 8 张脸、4 张通过。
  单人图各 1/1，多人图 6 张脸中 2 张通过；其余事实原因是 excessive_yaw=2、eyes_too_close=1、
  invalid_five_point_geometry=2。素材仅为 Tier B 本地工程验证，不进入 Git、CI、训练或再分发；发布方
  描述不是人工身份真值，结果也不是准确率。
- 尚未读取家庭录入照片，未做家庭宝宝准确率、独立 holdout 或 identity shadow。现有 24 小时零外发
  shadow 明确关闭身份能力，不能与未来身份 shadow 合并。完成家庭真实录入、人工遮挡/同主体复核、家庭独立
  holdout 和至少一周身份 shadow 前，不能称系统已经识别宝宝。
- 当前单 profile 策略固定返回 unknown(reason=single_profile_uncalibrated)，且身份不沿 tracker、事件或
  摄像头传播，不改变风险、通知、RecognitionAttempt 或 VLM provenance。

## 目标与非目标

目标是在完全本地、显式选择、可列举和可删除的前提下，为每个已检测的人脸给出保守的 open-set 身份推断。它是宝宝
看护与家庭安防的可选上下文，不是安全结论本身。

非目标：

- 不从 RTSP、事件片段、NAS 或 HA 历史中自动挖掘家庭成员；
- 不做跨摄像头长期人员追踪；
- 不做情绪、种族、性别、年龄或亲属关系推断；
- 不把 `unknown` 写成“陌生人”或“入侵者”；
- 不让身份识别替代跌落、危险区域、长时间静止等可见风险规则；
- 不调用远程 API，不上传原图、人脸 crop、embedding 或模板；
- 不使用 LLM/VLM 决定身份。

## 数据对象

| 对象 | 内容 | 存储规则 |
|---|---|---|
| `subject` | 随机 `subject_id`、本地显示别名、状态、创建/更新时间 | 文件名和事件 ID 只用 opaque ID；显示别名不得进入媒体文件名 |
| `enrollment_example` | 随机 example ID、来源类型、质量指标、模型 provenance | 原图/crop 默认仅在事务期间存在；保留必须另行 opt-in |
| `face_template` | 归一化 embedding、subject/example ID、模型 ID/revision/SHA-256 | 权限受限的本地模板库；不得写日志或通知 |
| `identity_inference` | face instance、结果枚举、top-1 score、margin、原因码 | 进入 `inferences`；不把 score 当概率，不沿 tracker 传播 |
| `identity_audit` | 操作、UTC 时间、opaque subject/example ID、数量、结果 | 不含原图、crop、embedding、显示别名或家庭路径 |

模板必须绑定生成它的 detector/embedder 名称、revision、SHA-256、预处理版本和 embedding 维度。模型或预处理变化时，旧模板
保持可读但不得被新模型静默复用；必须显式重建并重新评测。

## 多人画面的含义

一帧可以有 0、1、2 或更多人脸。系统按 face instance，而不是按整帧，处理身份：

```text
frame
├── face_instance A → matched(profile_x)
└── face_instance B → unknown
```

同帧两个框永远是两个独立候选。每帧的每张脸都独立评估；首版不会沿人脸 tracker、人体 tracker、遮挡后重现或跨摄像头
传播身份。未来若产生事件级 consensus，也必须由多张独立证据帧在同一 profile 上一致支持；任何绑定或一致性不确定都回退为
`unknown`，不能为了保持名字而强行续接。

## 录入流程

1. **显式启用**：身份能力默认关闭；配置须同时启用本地模型、模板库和 identity shadow，且网络策略仍为本地-only。
2. **确认授权**：成年人由本人同意；宝宝等未成年人由监护人确认。记录的是授权状态和时间，不记录法律文书或敏感备注。
3. **创建主体**：生成 opaque `subject_id`，显示别名只存在模板库元数据。禁止把姓名写进文件名、事件 ID 或模型目录。
4. **选择来源与框**：只接受用户明确选择的本地照片。一次请求必须先声明要创建/更新的 opaque `subject_id`、
   `declared_kind` 和精确 face box，并再次确认“这一个框属于这个主体”。当前产品仅做 preflight，不读取图片或
   写入模板。实际接线前若没有能展示全部检测框的本地选择器，只允许恰好一张通过门的人脸，并把唯一检测框作为
   待确认框；0 张或 2+ 张全部拒绝。未来多人照片只有在用户看见所有框并明确点选其中一个后才可录入，不能用
   “画面里大概是同一个人”代替主体与框的绑定。
5. **质量门**：检查最小脸尺寸、检测分数、清晰度、曝光、五点 landmarks、yaw/pitch、遮挡和边界截断。失败时返回原因，
   不把低质量 embedding 塞进模板库。
6. **样本多样性**：每个主体使用多张、相互独立的清晰样本，覆盖自然角度和光照；宝宝另需高机位与夜视条件。重复帧不能
   伪装成多个独立样本。
7. **事务提交**：先计算并校验全部 embedding，再一次性写入 subject/examples/templates；中途失败全部回滚，不留下半个主体。
8. **原图处理**：事务成功后默认清除录入工作区的临时 crop。若用户选择保留用于复核，单独记录保留范围和期限。
9. **独立复核**：用没有参与录入的样本确认该主体能匹配，并用其他家庭成员确认不会串人。录入成功不等于可以进入通知。

宝宝模板不得通过连续视频自动刷新。外观变化导致 unknown 增多时，系统只能提示“建议人工复核模板”，由用户选择新样本并
重新走上述流程。

## 推断策略

每个有效 face instance 依次执行检测质量门、五点对齐、embedding、模板检索和 open-set policy。建议的状态机为：

```text
poor face ───────────────────────────────────→ quality_rejected
valid face + no templates ──────────────────→ unknown(reason=no_templates)
valid face + no candidate above threshold ──→ unknown
valid face + close/conflicting candidates ──→ ambiguous
valid face + score + distinct-profile margin → matched
```

没有检测到脸时不产生 identity inference，也不因而推断宝宝或某个人不在画面。`matched` 必须同时满足：

- top-1 score 高于按家庭 holdout 校准的阈值；
- 已录入 profile 不少于两个时，top-1 与不同 profile 的 runner-up 差值高于 margin 门；当前只有一个
  profile 时固定返回 `unknown(reason=single_profile_uncalibrated)`，不得仅因最近模板分数高就输出名字，
  也不得伪造 runner-up。未来只有在独立家庭 unknown 负例、不同日期/光照 holdout 和版本化绝对拒绝阈值
  通过后，才可另提 policy 变更；
- 没有模型 provenance、维度、模板库版本或质量错误。

任一门不满足就降级为 `ambiguous`、`unknown` 或 `quality_rejected`，不得挑一个最接近的名字。模板库为空时对每张
合格脸返回 `unknown(reason=no_templates)`；这仍不等于“陌生人”。所有身份结果都不沿 tracker 传播，也不跨长时间或跨
摄像头继承。

## 事实、推断与告警

`observations` 只允许承载可见或测量事实，例如 face box、detector confidence、脸尺寸、清晰度、曝光、landmarks 和质量拒绝原因。
`inferences` 承载 profile candidate、`match_score`、runner-up margin、`matched/unknown/ambiguous/quality_rejected`、模型 provenance 和原因码。

身份状态不直接提高或降低安全风险：

- `matched(declared_kind=baby)` 只表示可能匹配用户声明为宝宝的 profile，不证明宝宝安全；仍由姿态、区域、持续时间和证据完整度判断风险；
- `matched(declared_kind=adult)` 不证明该行为被授权；
- `unknown` 不等于陌生人，也不自动产生安防告警；
- `ambiguous` 和 `quality_rejected` 不得被文案润色成确定身份。

家庭 shadow gate 通过前，对外通知只写“检测到 1/2 个可见人员”等非身份事实。通过后也应允许用户分别控制“本地 UI 展示
身份”和“通知携带身份”，默认通知仍不附图、不附 embedding。

## 删除与导出

当前产品只实现 `identity-deletion-preflight`：它验证持续同意和一次确认，但不打开模板库、不删除记录。
以下 list/delete/export 是接通实际原子 enrollment 前的产品准入要求，不应写成已经可用。

模板库必须提供：列出所有主体、列出某主体全部 examples、删除单个 example、删除整个 subject、清空模板库和导出非媒体
审计摘要。删除整个 subject 前，CLI/UI 显示精确 opaque ID、别名和待删数量，并要求一次确认。

删除成功的原子条件：

1. 删除该 example/subject 的 embeddings、派生 centroid/index entry 和保留的 enrollment crop；
2. 使内存索引失效并从磁盘重新加载，确认无法再命中；
3. 删除未提交事务的 staging/cache；
4. 写入只含 opaque ID、数量和 UTC 时间的删除审计；
5. 返回逐类删除数量和仍存在的引用数，引用数非零则操作失败并停止使用该模板库。

普通事件片段和历史通知不因模板删除被改写；它们本来就不应保存 embedding。若未来引入模板库备份，备份的范围、加密、
保留与彻底删除必须另设持久 opt-in 和恢复/清除说明，不能假装删除在线库等于删除所有备份。

## 本地安全边界

- 当前内部模板库原语使用每库随机 32-byte key 与 AES-256-GCM；每条 embedding 使用随机 12-byte nonce，
  AAD 绑定 declared kind、维度、模型 fingerprint、profile/template/schema。embedding 加密，必要索引元数据
  不加密，且原语不保存原始录入图或 crop。因此取得主机用户权限、读取元数据、内存或明文运行时仍是剩余风险，
  不能把 at-rest encryption 宣传为主机攻陷防护。
- 私有根目录要求位于 WSL 绝对路径并拒绝 `/mnt/c`；目录为 `0700`，key、SQLite、WAL/SHM 为 `0600`，
  同时检查 owner、符号链接和硬链接。实际产品 enrollment/list/delete 尚未接到该原语，权限目标通过不等于
  产品流程已经可用。
- 模型权重与模板库分目录；模型可校验和，模板属于家庭私有数据，二者都不提交 Git。
- 不记录人脸原图、crop、embedding 数值、相似度向量、家庭姓名、私有媒体路径或摄像头 URI。
- 适配器默认无远程网络权限；模型下载是单独、显式、可审计的准入动作，不在运行时自动发生。
- 数据库损坏、模型缺失、checksum 不符、维度不符或锁竞争时 fail closed，不产生 `matched`。
- anti-spoofing 若未来加入，只能作为独立 veto 信号；夜视/俯拍下的 liveness 失败不得据此推断成另一个人或入侵者。

## 家庭 shadow 评测门

既有 24 小时零外发 shadow 是 detector/tracker/事件/本地通知链路验收，运行时身份明确关闭。身份必须在
原子录入和家庭 holdout 通过后另跑至少一周 identity-only shadow；期间 identity 仍不进入通知、风险或
自动化。两个观察窗的分母、结论和验收记录不得合并。

录入和上线使用两套分离样本。holdout 至少按 `profile × camera × day/night × pose × occlusion` 记录覆盖情况，并包含单人、
多人、交叉和再出现。公开阈值只用于 smoke，不能作为家庭阈值。

报告至少包含：

- 人脸检测覆盖率与 `quality_rejected`/无脸原因分布；
- 每个 profile、每个摄像头和每种条件的 matched/unknown/ambiguous/quality_rejected；
- genuine/impostor score 与 top-1/top-2 margin 分布；
- false matched、跨 profile 错认、false unknown、ambiguous 数量和实际分母；
- 多帧 consensus 前后变化；
- 单帧和每个 track 的 p50/p95 延迟、CPU、RSS；
- 模型、预处理、模板库、数据 manifest 和配置 checksum。

进入通知/自动化的最低门：家庭 holdout 中观察到的跨 profile `matched` 错认为 0；低质量、夜视失败和遮挡失败都保持 fail closed；
多人画面没有合并或 identity bleed；删除后旧主体无法命中。报告必须同时写明样本量，不能把“本次观察为 0”宣传成真实世界
误认率为 0。只要模型、阈值、预处理、摄像头位置或模板库发生实质变化，就重新跑 gate。

## 开源参考与儿童数据结论

- [Frigate](https://github.com/blakeblackshear/frigate) 0.16+ / 当前
  [face recognition 文档](https://docs.frigate.video/configuration/face_recognition/)提供了最接近的
  本地产品参考：能力默认关闭，先检测 person 再检测 face；small FaceNet 走 CPU、large ArcFace 走
  GPU；人脸库通过 Add Face 和 Recent/Train 人工归类；unknown_score 与 recognition_threshold 分开；
  min_faces 要求多次一致后才向 person 加 sub-label；另有 blur filter。其建议从 1–5 张高质量正脸起步，
  逐步加入不同角度/光照，避免批量相似帧或过早混入 IR。这些 UX/策略值得借鉴。
- 我们与 Frigate 的安全差异是：不把 identity 沿 person/tracker 传播，不自动保存或训练最近尝试，单
  profile 固定 unknown，结果不进入风险/通知。Frigate 使用的 FaceNet/ArcFace 权重和训练数据尚未完成
  本项目独立许可审计，所以只借策略，不借代码路径或权重。
- [Double Take](https://github.com/jakowenko/double-take) 的 MIT 项目可参考 detector/NVR adapter、
  train/untrain、API/MQTT 边界；但其保存 match/unknown 图片、对外服务和外部识别后端不符合本项目
  默认隐私边界。[Viseron dlib](https://viseron.netlify.app/components-explorer/components/dlib)可作为
  传统本地路线参考，但不是当前 YuNet + SFace 首选。

婴幼儿可合法直接用于“同一宝宝 open-set 身份验证”的公开 benchmark 很稀缺：

- CLF 覆盖约 2–18 岁，论文或申请访问不等于允许再分发；其研究还明确说明 ITWCC 不在 public domain。
  参见 [CLF 论文](https://arxiv.org/abs/1711.03990)和
  [ICB 2018 longitudinal study](https://biometrics.cse.msu.edu/Publications/Face/Debetal_LongitudinalStudyOfChildFaceRecognition_ICB2018.pdf)。
- YFA 属于受控研究库；FG-NET 虽有少量从 0 岁开始的照片，但规模、许可和身份分组不足以成为本项目
  准入 benchmark；City Infant Faces / E-TIF 主要是表情刺激库，通常需要申请，也不适合回答同一身份
  open-set 问题。背景可参考
  [儿童人脸识别综述](https://www.mdpi.com/1424-8220/22/15/5842)。

因此项目不会自动下载或借用儿童图片。公开成人 smoke 只验证检测、对齐、embedding 与失败协议；只有
用户明确提供并授权、Git 外保存的家庭 Tier C 独立 holdout，才能回答当前宝宝、当前摄像头和当前条件下
的身份表现，且必须同时报告样本量和 unknown/误识分母。

## 分阶段实施

1. 已完成严格契约、保守 matcher、AES-GCM 模板库、原子 enrollment、只读 list、精确 delete 与产品 CLI 接线。
2. 已固定并本机 smoke YuNet 2023mar + SFace 2021dec；runner 通过 systemd 临时单元执行，强制
   `IPAddressDeny=any`，模型不接家庭媒体且运行时不自动下载。
3. 已用三份 Tier B NASA 成人公开图片完成 3 cases / 8 detected faces / 4 quality accepted 的协议 smoke；
   blur、exposure、roll/yaw、最小脸/眼距和五点几何门已固定进入 pipeline fingerprint。自动遮挡检测、
   anti-spoofing 和人工身份 ground truth 仍未实现，不能从该结果计算身份准确率。
4. 未开始：用户明确选择家庭主体、精确 face box 和多张独立录入样本，随后建立不参与录入的家庭 Tier C holdout。
5. 未开始：家庭 holdout 通过后另跑至少一周 identity-only shadow，再决定是否只向本地 UI 开放身份上下文；
   通知携带身份必须再单独授权。

来源与许可边界见[第三方许可证](../../THIRD_PARTY_LICENSES.md)。
