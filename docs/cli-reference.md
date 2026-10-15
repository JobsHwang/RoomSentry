# CLI 参考

本页由 `scripts/generate_cli_reference.py` 从当前 CLI 定义自动生成，请勿手工编辑。
命令按字典序排列；隐私、凭据与授权边界以 [SPEC.md](../SPEC.md) 为准。

| 命令 | 说明 |
|---|---|

| `ab-run` | run sequential VLM A/B evaluation |
| `analysis-gpu-status` | read NVIDIA capacity and show whether configured analysis thresholds allow a job |
| `analysis-queue-status` | show bounded analysis queue counts without exposing event content |
| `analysis-retry` | explicitly reset one failed analysis job for another bounded attempt cycle |
| `analysis-run-once` | claim and process at most one persisted analysis job |
| `analysis-worker` | consume persisted analysis jobs sequentially within explicit stopping bounds |
| `archive-copy` | atomically copy one strict manifest without overwriting unknown files |
| `archive-enqueue` | validate and persist one archive manifest without copying it |
| `archive-enqueue-event` | bind one stored event's clip and evidence frames to an archive job |
| `archive-enqueue-report` | bind one generated Markdown/HTML daily report to an archive job |
| `archive-plan` | hash a bounded set of local spool files into a strict private manifest |
| `archive-queue-status` | show archive queue aggregates without exposing paths or manifests |
| `archive-retry` | explicitly reset one failed archive job for another bounded attempt cycle |
| `archive-run-once` | claim and process at most one persisted archive job |
| `archive-worker` | process archive jobs sequentially within explicit stopping bounds |
| `attest-household-pose-fusion-temporal-holdout` | bind a frozen-profile temporal holdout to its disjoint profile-selection batch without granting accuracy or deployment authority |
| `authorize-ha` | authorize this LAN-only client in a browser and store refreshable HA credentials in a new 0600 file without disclosing tokens |
| `buffer-extract` | assemble an event clip and keyframes from a segment manifest |
| `candidate-health-status` | read path-free freshness and execution metrics for candidate workers |
| `candidate-presence-health-init` | create or validate one stopped lifetime health baseline for a presence-only worker |
| `candidate-soak-finish` | finish the active private YOLO soak after all fixed workers stop |
| `candidate-soak-start` | capture a private v2 baseline before the fixed manual YOLO soak |
| `candidate-soak-status` | read the path-free latest manual YOLO soak report |
| `candidate-worker` | scan new buffer segments and enqueue mature low-cost candidate events |
| `capture-ha-token` | receive one HA access value through a loopback-only browser form and store it in a new 0600 file without disclosing it |
| `combine-household-model-temporal-diagnostic-batches` | combine two or more disjoint same-model temporal diagnostic reports into one camera-stratified diagnostic-only aggregate |
| `combine-household-pose-fusion-diagnostic-batches` | combine disjoint same-profile detector/pose fusion diagnostics |
| `compare-household-model-diagnostic-batches` | compare one detector across two disjoint model-reviewed household batches under explicit diagnostic-only drift tolerances |
| `compare-household-model-presence-diagnostics` | compare two or more diagnostic reports that share one model-generated review, source set, bin grid, and threshold grid without selecting a winner |
| `compare-household-trigger-coverage` | compare two or more household trigger coverage reports that share the same annotation, media, sample grid, and thresholds |
| `compare-tracker-predictions` | compare anonymous association links from two predictions on the same batch without claiming accuracy |
| `daily-digest-export` | decode one private facts-only plan into a bounded local MP4 |
| `daily-digest-export-status` | inspect private digest exports without reading video or creating files |
| `daily-digest-plan` | hash admitted local clips and persist one facts-only daily highlight plan |
| `daily-digest-plan-status` | inspect private daily digest plans without reading media or creating files |
| `discover-ha-signals` | read a bounded HA entity inventory over a LAN-only WebSocket and write private event/binary-sensor candidates |
| `enqueue-buffer-event` | extract buffered evidence and persist a bounded analysis job without loading VLM |
| `enqueue-signal-event` | bind one strict hardware/HA signal to buffered evidence and the analysis queue |
| `evaluate-candidate-start-gate-sweep` | measure candidate trigger coverage across start-confidence gates from one admitted low-threshold person scan |
| `evaluate-detector-box-quality` | evaluate one admitted person detector scan against aligned publisher-human boxes, including empty sampled frames |
| `evaluate-detector-threshold-sweep` | derive a strict confidence-threshold quality curve from one admitted low-threshold person scan without rerunning the model |
| `evaluate-event-policy` | replay completed events through a household event policy and report route/suppress decisions without delivering anything |
| `evaluate-event-policy-cascade` | replay DecisionRecord-bound completed events through a household event policy without reading media or delivering anything |
| `evaluate-household-human-pose-fusion-audit` | replay a frozen temporal-holdout detector/Pose profile against a bound single-human blind frame reference |
| `evaluate-household-model-presence-diagnostic` | diagnose person-detector confidence thresholds against a bound model-generated household presence review without rerunning a model |
| `evaluate-household-model-temporal-diagnostic` | diagnose causal detector holds by anonymous camera and clear/partial visibility using only a saved model review and detector scans |
| `evaluate-household-pose-fusion-diagnostic` | replay one explicit per-camera detector/pose OR profile over admitted saved scans without reading media |
| `evaluate-household-sampled-coverage` | measure actual-PTS person detection coverage using approved household annotation v1 or elapsed-source-PTS v2 and an explicit hashed timeline binding; no candidate grouping or danger recognition |
| `evaluate-household-trigger-coverage` | measure nominal-grid person-trigger coverage from approved household annotation v1; elapsed-source-PTS annotation v2 requires evaluate-household-sampled-coverage |
| `evaluate-pose-quality` | measure keypoint coverage and adjacent-frame coordinate stability without identity or posture inference |
| `evaluate-tracker-association` | score tracker association and fragmentation only on approved human-labelled boxes |
| `evaluate-tracker-disagreement` | compare relative errors only on human-reviewed exclusive links; do not claim overall accuracy |
| `ha-signal-dispatch-once` | lease at most one HA signal and enqueue bounded local buffer evidence without loading a model |
| `ha-signal-dispatch-worker` | consume the local HA signal outbox sequentially and enqueue bounded buffer evidence without loading a model |
| `ha-signal-outbox-ack` | acknowledge one leased HA signal and scrub its persisted payload |
| `ha-signal-outbox-claim` | lease at most one persisted normalized HA signal |
| `ha-signal-outbox-fail` | record a bounded downstream failure for one leased HA signal |
| `ha-signal-outbox-retry` | explicitly return one failed HA signal to the bounded pending queue |
| `ha-signal-outbox-status` | read HA signal outbox health without disclosing payloads or paths |
| `home-control` | control the fixed local camera, HA, and bounded shadow units as one stack |
| `home-daily-report` | generate a path-free aggregate local daily report without reading media |
| `home-dashboard` | serve the fixed manual home controller on IPv4 loopback only |
| `identity-consent-status` | show path-free identity feature and consent gates without reading media |
| `identity-delete-profile` | delete exactly one opaque local identity profile and its templates |
| `identity-deletion-preflight` | authorize deletion of one opaque identity profile without opening or mutating identity storage |
| `identity-enroll` | enroll one local identity profile from a private 3-12 image manifest; household execution requires systemd IPAddressDeny=any |
| `identity-enrollment-preflight` | authorize one enrollment attempt without opening media, models, event history, or identity storage |
| `identity-profiles` | list opaque local identity profiles without loading or exporting embeddings |
| `import-caviar-tracking-ground-truth` | verify and convert admitted CAVIAR CVML person boxes into a path-free association batch and publisher-human ground truth |
| `import-household-human-presence-audit` | strictly bind a completed local blind human frame review to its uniform selection manifest |
| `init-db` | initialize the SQLite event database |
| `list-events` | list recent SQLite events |
| `listen-ha-signals` | subscribe to bounded LAN-only HA state_changed events and emit only explicitly mapped vendor-neutral signals |
| `local-review-retry` | retry one historically bound event using only the configured local secondary model |
| `local-review-status` | read path-free local review retry attempts without running models |
| `media-health-status` | assess one MediaSource health snapshot without restarting any service |
| `normalize-ha-signals` | normalize an offline HA WebSocket event capture without connecting to HA |
| `notification-enqueue-event` | enqueue one stored final or pending-review event for a local sink |
| `notification-local-test` | write one explicit test-only notification through the local file outbox |
| `notification-queue-status` | show the path-free local notification outbox without delivering |
| `notification-retry` | explicitly reset one failed local notification job |
| `notification-run-once` | deliver at most one queued notification to a configured sink |
| `notification-transport-preview` | inspect a private SMTP/webhook profile without using the network or exposing secrets |
| `notification-worker` | run a bounded notification delivery loop for one configured sink |
| `pose-scan-admitted-file` | scan one admitted bounded local video with a reviewed offline YOLO pose model |
| `pose-scan-file` | scan one bounded local video with a reviewed offline YOLO pose model |
| `prepare-detector-tracking-batch` | convert an admitted detector scan into nominal-grid or actual container-PTS tracker frames, including empty samples, without claiming ground truth |
| `prepare-detector-tracking-evaluation` | convert an exact detector quality run into sampled detector frames with publisher-subject matches and explicit false positives |
| `prepare-tracker-disagreement-review` | prepare one pending human same/different question for every exclusive tracker link |
| `prepare-tracker-review` | create a path-free pending human association review for every detection in one fixed batch |
| `process-buffer-event` | persist one buffered RTSP event through VLM, SQLite, and daily reports |
| `process-file` | process one bounded local video |
| `recent-observations` | read local model descriptions per mapped camera; not live baby location or state |
| `recognition-audit-status` | show path-free local, secondary, and remote recognition attempts |
| `release-audit` | scan the Git index for private or heavyweight release artifacts |
| `release-bundle` | atomically create a deterministic audited source tar.gz with SBOM |
| `release-sbom` | generate a deterministic CycloneDX source SBOM from the safe Git index |
| `remote-audit-status` | show media-free remote review attempts from SQLite |
| `remote-mock-server` | serve a bounded OpenAI-compatible fixture on an IP loopback address only |
| `remote-preview` | show the path-free remote target and selected frame budget without uploading |
| `report` | generate Markdown and HTML daily reports |
| `report-refresh-request` | durably request a coalesced local daily report without reading media |
| `report-refresh-retry` | explicitly retry one failed daily refresh without repeating recognition |
| `report-refresh-run-once` | render at most one queued daily report without models or notifications |
| `report-refresh-status` | read report refresh counts without creating files or disclosing private paths |
| `report-refresh-worker` | refresh daily reports independently within explicit stopping bounds |
| `retention-scan` | list expired media; never delete |
| `room-presence-evaluate` | evaluate conservative identity-free room presence from bounded scan facts |
| `room-registry-status` | read the private physical-camera, logical-stream, and room topology |
| `rtsp-buffer` | record a bounded RTSP segment buffer and write a redacted manifest |
| `rtsp-soak` | continuously decode a loopback RTSP stream and report health metrics |
| `runtime-status` | aggregate read-only media, queue, storage, GPU, and safety status |
| `runtime-trial-finish` | settle one bounded home-runtime trial after its workers stop |
| `runtime-trial-start` | capture a private baseline for one bounded home-runtime trial |
| `runtime-trial-status` | read the latest path-free bounded home-runtime trial result |
| `schema` | print the strict VLM JSON schema |
| `select-candidate-start-gate-operating-point` | select one dataset-specific candidate start threshold under explicit event-coverage and negative-activation constraints |
| `select-detector-operating-point` | select a dataset-specific point from a strict threshold sweep under explicit precision, recall, and longest-miss constraints |
| `select-household-model-presence-working-point` | select one diagnostic-only next shadow working point under explicit model-generated presence constraints |
| `select-household-model-temporal-batch-working-point` | select one diagnostic-only cross-batch temporal shadow point under aggregate, visibility, per-camera, and hold constraints |
| `select-household-model-temporal-working-point` | select one diagnostic-only next shadow temporal point under aggregate, visibility, per-negative-camera, and hold constraints |
| `select-household-trigger-operating-point` | select one dataset-specific household trigger threshold under mandatory danger-episode recall and explicit coverage constraints |
| `smoke-demo` | run a no-video/no-AI plumbing demo |
| `store-ha-token` | store one HA access value from a hidden prompt in a new 0600 file without command-line or stdout disclosure |
| `systemd-render` | render hardened user units without installing, enabling, or starting them |
| `temporal-zone-ack` | acknowledge temporal signals only after downstream persistence succeeds |
| `temporal-zone-pending` | replay unacknowledged temporal zone signals from the local outbox |
| `temporal-zone-run` | convert tracked observations into persistent zone signals without a model |
| `track-iou-ack` | acknowledge tracked observations after downstream persistence succeeds |
| `track-iou-pending` | replay unacknowledged tracked observations from the local outbox |
| `track-iou-predict` | export deterministic IoU tracking assignments for one fixed DetectionFrameBatch |
| `track-iou-run` | associate normalized detections into anonymous short-lived tracks |
| `track-iou-zone-run` | persist pending tracked observations into temporal zones before acknowledging them |
| `track-motion-ack` | acknowledge motion-tracker observations after downstream persistence succeeds |
| `track-motion-pending` | replay unacknowledged motion-tracker observations from the local outbox |
| `track-motion-predict` | export deterministic constant-velocity, two-stage global association assignments for one fixed DetectionFrameBatch |
| `track-motion-run` | associate normalized detections with the persistent evaluated two-stage motion tracker |
| `track-motion-zone-run` | persist pending motion-tracker observations into temporal zones before acknowledging them |
| `validate-ab-dataset` | validate a strict local VLM evaluation JSONL dataset and frame hashes |
| `validate-admitted-pose-scan` | validate a strict path-free AdmittedPoseScan v1 JSON file |
| `validate-candidate-start-gate-sweep` | validate a strict path-free CandidateStartGateSweepReport v1 JSON file |
| `validate-decision-json` | validate one strict DecisionRecord v1 JSON object |
| `validate-detection-frames` | validate a strict path-free DetectionFrameBatch v1 JSON file |
| `validate-detector-box-quality` | validate a strict path-free DetectorBoxQualityReport v1 JSON file |
| `validate-detector-scan` | validate a strict path-free AdmittedDetectorScan v1 JSON file |
| `validate-detector-threshold-sweep` | validate a strict path-free DetectorThresholdSweepReport v1 JSON file |
| `validate-event-policy` | validate a strict path-free HouseholdEventPolicy v1 household event-package routing policy |
| `validate-event-policy-cascade` | validate a strict path-free EventPolicyCascadeReport v1 JSON file |
| `validate-household-episode-annotation` | validate path-free household annotation v1 nominal-grid or v2 elapsed-source-PTS JSON without upgrading pending labels to human-verified ground truth |
| `validate-household-human-pose-fusion-audit` | validate one conservative single-human frozen-fusion audit report |
| `validate-household-human-presence-audit` | validate one strict path-free single-human blind frame audit |
| `validate-household-model-diagnostic-drift` | validate a strict path-free HouseholdModelDiagnosticDriftReport v1 JSON file |
| `validate-household-model-presence-diagnostic` | validate a strict path-free HouseholdModelDiagnosticReport v1 JSON file |
| `validate-household-model-presence-review` | validate a strict path-free model-generated household visible-person presence review |
| `validate-household-model-temporal-batch-diagnostic` | validate a strict path-free HouseholdModelTemporalBatchDiagnosticReport v1 JSON file |
| `validate-household-model-temporal-diagnostic` | validate a strict path-free HouseholdModelTemporalDiagnosticReport v1 JSON file |
| `validate-household-pose-fusion-batch-diagnostic` | validate one strict cross-batch detector/pose fusion report |
| `validate-household-pose-fusion-diagnostic` | validate one strict evaluation-only detector/pose fusion report |
| `validate-household-pose-fusion-profile` | validate one strict evaluation-only per-camera detector/pose profile |
| `validate-household-pose-fusion-temporal-holdout` | validate one strict frozen-profile temporal holdout attestation |
| `validate-household-trigger-coverage` | validate a strict path-free HouseholdTriggerCoverageReport v1 JSON file |
| `validate-json` | validate one strict VLM JSON file |
| `validate-notification-json` | validate one strict path-free NotificationMessage v1 JSON file |
| `validate-pose-frames` | validate a strict path-free PoseFrameBatch v1 JSON file |
| `validate-pose-quality` | validate a strict path-free PoseQualityReport v1 JSON file |
| `validate-signal-json` | validate one strict vendor-neutral SignalEvent v1 JSON object |
| `validate-source-admission` | validate a strict external model/data source admission manifest without reading the referenced asset |
| `validate-tracked-observations` | validate strict path-free TrackedObjectObservation v1 JSON |
| `validate-tracker-association` | validate a path-free tracker association report |
| `validate-tracker-comparison` | validate a path-free, no-ground-truth tracker disagreement report |
| `validate-tracker-disagreement` | validate a path-free human-reviewed relative disagreement report |
| `validate-tracker-disagreement-review` | validate human same/different labels against the exact two prediction link sets |
| `validate-tracker-predictions` | validate tracker assignments against the exact source detection batch |
| `validate-tracker-review` | validate pending or approved anonymous labels against one detection batch |
| `verify-source-asset` | verify one bounded local external asset against an approved admission manifest without network access |
| `verify-yolo-model` | verify a reviewed local YOLO weight without importing or running the model |
| `verify-yolo-pose-model` | verify a reviewed local YOLO pose weight without importing or running the model |
| `visual-predicate-health-status` | assess one private resident visual-predicate health snapshot read-only |
| `visual-predicate-human-calibration-report` | compare latest local human visible-fact labels with bound shadow answers without reading media or changing decisions |
| `visual-predicate-review-report` | apply the deterministic shadow-only manual-review policy to recent digest-bound atomic predicate results without reading media |
| `visual-predicate-shadow-queue-status` | show the path-free shadow attempt ledger without reading media |
| `visual-predicate-shadow-report` | show recent path-free shadow states and atomic answers without reading media |
| `visual-predicate-shadow-retry` | explicitly reset one failed shadow attempt without changing main analysis |
| `visual-predicate-shadow-run` | run the default-off local atomic-predicate shadow path from explicitly admitted candidate metadata and private frames |
| `visual-predicate-shadow-run-next` | sync pending hardware evidence and completed detector events, then process at most one isolated shadow attempt |
| `visual-predicate-shadow-worker` | reuse one local adapter while consuming isolated shadow attempts |
| `watch-ha-signals` | run a bounded LAN-only HA state_changed watcher with sanitized health counters and exponential reconnect |
| `yolo-scan-file` | scan one admitted bounded local video with a reviewed offline YOLO detector |
