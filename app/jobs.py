import uuid
from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"  # finished, but citations failed the grounding check


class Job(BaseModel):
    id: str
    status: JobStatus = JobStatus.PENDING
    result: str | None = None
    error: str | None = None
    trace: list[dict] = Field(default_factory=list)  # filled from trace_events on read
    grounding: dict | None = None  # {"ok", "cited", "invented", "external", "judge"?}
    migration: dict | None = None  # proposed fix; only offered when verified in the sandbox


def create_job() -> Job:
    return Job(id=str(uuid.uuid4()))
