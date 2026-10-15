"""Domain contracts for the baby-monitor core.

Historically a single ``domain.py``; now a package with one module per
aggregate. The public surface is re-exported here so
``from baby_monitor.domain import X`` keeps working unchanged.
"""

from __future__ import annotations

from baby_monitor.domain._shared import (  # noqa: F401
    SchemaError,
)
from baby_monitor.domain.candidate import (  # noqa: F401
    CandidateEvent,
    Detection,
    EventMedia,
    EvidenceGap,
)
from baby_monitor.domain.decision import (  # noqa: F401
    DecisionRecord,
)
from baby_monitor.domain.enums import (  # noqa: F401
    EventType,
    FinalSource,
    ReviewStatus,
    ReviewTier,
    RiskLevel,
    SignalType,
    SubjectKind,
)
from baby_monitor.domain.recognition import (  # noqa: F401
    VLM_EVENT_KEYS,
    RecognitionProvenance,
    RecognitionRunMetrics,
    Subject,
    VLMEvent,
)
from baby_monitor.domain.signal import (  # noqa: F401
    SignalEvent,
)

__all__ = [
    "CandidateEvent",
    "DecisionRecord",
    "Detection",
    "EventMedia",
    "EventType",
    "EvidenceGap",
    "FinalSource",
    "RecognitionProvenance",
    "RecognitionRunMetrics",
    "ReviewStatus",
    "ReviewTier",
    "RiskLevel",
    "SchemaError",
    "SignalEvent",
    "SignalType",
    "Subject",
    "SubjectKind",
    "VLMEvent",
    "VLM_EVENT_KEYS",
]
