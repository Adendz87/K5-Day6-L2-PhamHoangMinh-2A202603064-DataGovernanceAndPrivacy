"""Unit tests for JobLifecycleManager."""
from pathlib import Path
from service.core.database import JobDatabase
from service.core.job_lifecycle import JobLifecycleManager, JobState


def test_legal_and_illegal_job_transitions(tmp_path: Path):
    db = JobDatabase(tmp_path / "lifecycle.sqlite")
    mgr = JobLifecycleManager(db)
    job_id = "job_life_1"
    run_id = "run_01"

    db.create_job(job_id, run_id, initial_status=JobState.CREATED.value)

    # CREATED -> RECEIVING (legal)
    assert mgr.transition_job(job_id, run_id, JobState.RECEIVING) is True
    assert mgr.get_job_state(job_id, run_id) == JobState.RECEIVING

    # RECEIVING -> VALIDATING (legal)
    assert mgr.transition_job(job_id, run_id, JobState.VALIDATING) is True

    # VALIDATING -> INPUT_COMPLETE (legal)
    assert mgr.transition_job(job_id, run_id, JobState.INPUT_COMPLETE) is True

    # INPUT_COMPLETE -> PROCESSING (legal)
    assert mgr.transition_job(job_id, run_id, JobState.PROCESSING) is True

    # PROCESSING -> COMPLETED (legal via direct completion)
    assert mgr.transition_job(job_id, run_id, JobState.COMPLETED) is True

    # COMPLETED -> PROCESSING (illegal - terminal state)
    assert mgr.transition_job(job_id, run_id, JobState.PROCESSING) is False
    assert mgr.get_job_state(job_id, run_id) == JobState.COMPLETED
