import httpx
import pytest

from app.agent import groq_client

REAL_ASYNC_CLIENT = httpx.AsyncClient


@pytest.fixture
def http(monkeypatch):
    """Route Groq HTTP through a handler; record requests and sleeps."""
    state = {"requests": [], "sleeps": [], "responses": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        item = state["responses"].pop(0)
        if isinstance(item, Exception):  # simulate a network failure
            raise item
        return item

    def factory(**kw):
        return REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kw)

    async def fake_sleep(seconds):
        state["sleeps"].append(seconds)

    monkeypatch.setattr(groq_client.httpx, "AsyncClient", factory)
    monkeypatch.setattr(groq_client.asyncio, "sleep", fake_sleep)
    return state
