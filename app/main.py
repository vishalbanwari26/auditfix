import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import store
from app.jobs import Job, create_job
from app.runner import run_job

_tasks: set[asyncio.Task] = set()  # strong refs so tasks aren't garbage collected


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(store.init_db)
    await asyncio.to_thread(store.fail_interrupted_jobs)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/jobs", status_code=202)
async def submit_job() -> dict:
    job = create_job()
    await asyncio.to_thread(store.save_job, job)  # saved BEFORE the task starts
    task = asyncio.create_task(run_job(job))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"job_id": job.id, "status_url": f"/jobs/{job.id}"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> Job:
    job = await asyncio.to_thread(store.load_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    job.trace = await asyncio.to_thread(store.load_trace, job_id)
    return job
