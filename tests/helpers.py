import asyncio
import time

import httpx


async def wait_for_status(
    client: httpx.AsyncClient,
    job_id: str,
    target: str,
    timeout: float = 5.0,
    interval: float = 0.1,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = (await client.get(f"/jobs/{job_id}")).json()
        if job["status"] == target:
            return job
        await asyncio.sleep(interval)
    raise TimeoutError(f"job {job_id} never reached {target!r} within {timeout}s")
