from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
import uuid

from app.logging_config import get_logger
from pipeline.db import DatabaseManager
from schemas.common import JobStatus, SportType, StageName
from schemas.job import JobRecord, StageRunRecord

logger = get_logger(__name__)


# Valid forward progression transitions
FORWARD_TRANSITIONS: Dict[JobStatus, Set[JobStatus]] = {
    JobStatus.DISCOVERED: {JobStatus.RANKED, JobStatus.FAILED},
    JobStatus.RANKED: {JobStatus.TOPIC_CHOSEN, JobStatus.FAILED},
    JobStatus.TOPIC_CHOSEN: {JobStatus.FACTS_VERIFIED, JobStatus.FAILED},
    JobStatus.FACTS_VERIFIED: {JobStatus.SCRIPT_DRAFTED, JobStatus.FAILED},
    JobStatus.SCRIPT_DRAFTED: {JobStatus.SCRIPT_APPROVED, JobStatus.FAILED},
    JobStatus.SCRIPT_APPROVED: {JobStatus.MEDIA_READY, JobStatus.FAILED},
    JobStatus.MEDIA_READY: {JobStatus.RENDERED, JobStatus.FAILED},
    JobStatus.RENDERED: {JobStatus.VIDEO_APPROVED, JobStatus.FAILED},
    JobStatus.VIDEO_APPROVED: {JobStatus.PUBLISHED, JobStatus.FAILED},
    JobStatus.PUBLISHED: set(),
    JobStatus.FAILED: set(),
}

# Stages mapped to resulting successful job status
STAGE_RESULT_STATUS: Dict[StageName, JobStatus] = {
    StageName.DISCOVERY: JobStatus.DISCOVERED,
    StageName.RANKING: JobStatus.RANKED,
    StageName.FACTS: JobStatus.FACTS_VERIFIED,
    StageName.SCRIPT: JobStatus.SCRIPT_DRAFTED,
    StageName.MEDIA: JobStatus.MEDIA_READY,
    StageName.RENDER: JobStatus.RENDERED,
    StageName.PUBLISH: JobStatus.PUBLISHED,
}

# Allowed rollback/re-run transitions
# Allows returning to earlier states or recovering from FAILED
ALLOWED_RERUN_STATUSES: Set[JobStatus] = {
    JobStatus.DISCOVERED,
    JobStatus.RANKED,
    JobStatus.TOPIC_CHOSEN,
    JobStatus.FACTS_VERIFIED,
    JobStatus.SCRIPT_DRAFTED,
    JobStatus.SCRIPT_APPROVED,
    JobStatus.MEDIA_READY,
    JobStatus.RENDERED,
    JobStatus.VIDEO_APPROVED,
    JobStatus.FAILED,
}


class InvalidStateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class PipelineStateMachine:
    """Manages the job lifecycle and enforces state transitions."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def create_new_job(self, sport: SportType, job_id: Optional[str] = None) -> JobRecord:
        """Create a new job in DISCOVERED status."""
        jid = job_id or str(uuid.uuid4())
        job = self.db.create_job(job_id=jid, sport=sport, status=JobStatus.DISCOVERED)
        logger.info("Created new job %s for sport %s", jid, sport.value)
        return job

    def can_transition(self, current_status: JobStatus, target_status: JobStatus, allow_rerun: bool = False) -> bool:
        """Check if transition from current_status to target_status is valid."""
        if target_status == JobStatus.FAILED:
            return True

        # Check standard forward transitions
        valid_next = FORWARD_TRANSITIONS.get(current_status, set())
        if target_status in valid_next:
            return True

        # If re-running or recovering from failure, allow transition to valid re-run target
        if allow_rerun and target_status in ALLOWED_RERUN_STATUSES:
            return True

        return False

    def transition_job(
        self,
        job_id: str,
        target_status: JobStatus,
        error_message: Optional[str] = None,
        allow_rerun: bool = False,
    ) -> JobRecord:
        """Execute a state transition for the given job."""
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if not self.can_transition(job.status, target_status, allow_rerun=allow_rerun):
            raise InvalidStateTransitionError(
                f"Cannot transition job {job_id} from {job.status.value} to {target_status.value} "
                f"(allow_rerun={allow_rerun})"
            )

        updated = self.db.update_job_status(job_id, target_status, error_message=error_message)
        if not updated:
            raise RuntimeError(f"Failed to update job {job_id} status in database")

        logger.info(
            "Job %s transitioned from %s to %s",
            job_id,
            job.status.value,
            target_status.value,
        )

        refreshed = self.db.get_job(job_id)
        assert refreshed is not None
        return refreshed

    def record_stage_execution(
        self,
        job_id: str,
        stage_name: StageName,
        input_data: Any,
        output_data: Any,
        success: bool = True,
        error_message: Optional[str] = None,
        allow_rerun: bool = False,
    ) -> StageRunRecord:
        """Record stage execution inputs/outputs and optionally advance job status."""
        status_str = "COMPLETED" if success else "FAILED"
        run_record = self.db.record_stage_run(
            job_id=job_id,
            stage_name=stage_name,
            input_data=input_data,
            output_data=output_data,
            status=status_str,
        )

        if not success:
            self.transition_job(
                job_id=job_id,
                target_status=JobStatus.FAILED,
                error_message=error_message or f"Stage {stage_name.value} failed",
                allow_rerun=True,
            )
        elif stage_name in STAGE_RESULT_STATUS:
            target = STAGE_RESULT_STATUS[stage_name]
            self.transition_job(job_id=job_id, target_status=target, allow_rerun=allow_rerun)

        return run_record

    def get_resumable_jobs(self) -> List[JobRecord]:
        """Find jobs that can be resumed after an interruption or crash."""
        all_jobs = self.db.list_jobs(limit=100)
        # Exclude completed, published, or aborted jobs
        return [j for j in all_jobs if j.status not in (JobStatus.PUBLISHED, JobStatus.FAILED)]
