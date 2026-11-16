import pytest
from pipeline.state_machine import InvalidStateTransitionError, PipelineStateMachine
from schemas.common import JobStatus, SportType, StageName


def test_valid_forward_transitions(state_machine: PipelineStateMachine):
    """Verify that a job can progress forward through valid lifecycle states."""
    job = state_machine.create_new_job(sport=SportType.CRICKET, job_id="lifecycle-test")
    assert job.status == JobStatus.DISCOVERED

    # Advance stage by stage
    progression = [
        JobStatus.RANKED,
        JobStatus.TOPIC_CHOSEN,
        JobStatus.FACTS_VERIFIED,
        JobStatus.SCRIPT_DRAFTED,
        JobStatus.SCRIPT_APPROVED,
        JobStatus.MEDIA_READY,
        JobStatus.RENDERED,
        JobStatus.VIDEO_APPROVED,
        JobStatus.PUBLISHED,
    ]

    current_status = job.status
    for next_status in progression:
        assert state_machine.can_transition(current_status, next_status) is True
        job = state_machine.transition_job(job.job_id, next_status)
        assert job.status == next_status
        current_status = next_status


def test_invalid_transition_raises_error(state_machine: PipelineStateMachine):
    """Verify illegal transitions (e.g. DISCOVERED -> PUBLISHED directly) are blocked."""
    job = state_machine.create_new_job(sport=SportType.FOOTBALL, job_id="illegal-test")
    assert job.status == JobStatus.DISCOVERED

    assert state_machine.can_transition(JobStatus.DISCOVERED, JobStatus.PUBLISHED) is False
    with pytest.raises(InvalidStateTransitionError):
        state_machine.transition_job(job.job_id, JobStatus.PUBLISHED)


def test_transition_to_failed_from_any_state(state_machine: PipelineStateMachine):
    """Verify any state can transition to FAILED on error."""
    job = state_machine.create_new_job(sport=SportType.F1, job_id="failure-test")
    job = state_machine.transition_job(job.job_id, JobStatus.RANKED)

    failed_job = state_machine.transition_job(
        job.job_id,
        JobStatus.FAILED,
        error_message="Simulated error in ranking",
    )
    assert failed_job.status == JobStatus.FAILED
    assert failed_job.error_message == "Simulated error in ranking"


def test_rerun_and_resume_transitions(state_machine: PipelineStateMachine):
    """Verify jobs can be re-run or resumed from previous states when allow_rerun=True."""
    job = state_machine.create_new_job(sport=SportType.FOOTBALL, job_id="rerun-test")
    state_machine.transition_job(job.job_id, JobStatus.RANKED)
    state_machine.transition_job(job.job_id, JobStatus.TOPIC_CHOSEN)
    state_machine.transition_job(job.job_id, JobStatus.FAILED, error_message="Network glitch")

    # Re-running from failure back to TOPIC_CHOSEN
    resumed = state_machine.transition_job(job.job_id, JobStatus.TOPIC_CHOSEN, allow_rerun=True)
    assert resumed.status == JobStatus.TOPIC_CHOSEN


def test_record_stage_execution_advances_status(state_machine: PipelineStateMachine):
    """Verify record_stage_execution stores stage runs and automatically advances status."""
    job = state_machine.create_new_job(sport=SportType.CRICKET, job_id="auto-advance-test")
    assert job.status == JobStatus.DISCOVERED

    # Discovery stage completes -> updates to RANKED
    state_run = state_machine.record_stage_execution(
        job_id=job.job_id,
        stage_name=StageName.RANKING,
        input_data={"stories": 10},
        output_data={"ranked": [1, 2, 3]},
        success=True,
    )

    refreshed = state_machine.db.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.RANKED
    assert state_run.status == "COMPLETED"
