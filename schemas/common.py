from enum import Enum


class SportType(str, Enum):
    FOOTBALL = "football"
    CRICKET = "cricket"
    F1 = "f1"


class JobStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    RANKED = "RANKED"
    TOPIC_CHOSEN = "TOPIC_CHOSEN"
    FACTS_VERIFIED = "FACTS_VERIFIED"
    SCRIPT_DRAFTED = "SCRIPT_DRAFTED"
    SCRIPT_APPROVED = "SCRIPT_APPROVED"
    MEDIA_READY = "MEDIA_READY"
    RENDERED = "RENDERED"
    VIDEO_APPROVED = "VIDEO_APPROVED"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"


class StageName(str, Enum):
    DISCOVERY = "discovery"
    RANKING = "ranking"
    SOURCES = "sources"
    FACTS = "facts"
    SCRIPT = "script"
    VERIFY = "verify"
    MEDIA = "media"
    VOICE = "voice"
    CAPTIONS = "captions"
    RENDER = "render"
    PUBLISH = "publish"


class SourceTier(str, Enum):
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    UNKNOWN = "unknown"


class FactType(str, Enum):
    SCORE = "score"
    DATE = "date"
    NAME = "name"
    NUMBER = "number"
    QUOTE = "quote"
    TRANSFER = "transfer"
    INJURY = "injury"
    RECORD = "record"
    RANKING = "ranking"
    OTHER = "other"


class CorroborationStatus(str, Enum):
    VERIFIED = "verified"
    REPORTED = "reported"
    CONFLICTED = "conflicted"
    UNVERIFIED = "unverified"


class VerificationVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
