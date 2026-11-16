from datetime import datetime, timezone
import json
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from schemas.common import JobStatus, SportType, StageName


class JobRecord(BaseModel):
    job_id: str
    sport: SportType
    status: JobStatus = JobStatus.DISCOVERED
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error_message: Optional[str] = None


class StageRunRecord(BaseModel):
    id: Optional[int] = None
    job_id: str
    stage_name: StageName
    input_json: str = "{}"
    output_json: str = "{}"
    status: str = "COMPLETED"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def get_input(self) -> Dict[str, Any]:
        try:
            return json.loads(self.input_json)
        except Exception:
            return {}

    def get_output(self) -> Dict[str, Any]:
        try:
            return json.loads(self.output_json)
        except Exception:
            return {}
