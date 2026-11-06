"""A fake OpenAI-compatible upstream for load and chaos tests: `gateway mock-upstream --latency-ms 50 --fail-rate 0.3`."""

from __future__ import annotations

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import ProviderConfig
from .providers import MockProvider, UpstreamError


def create_mock_app(latency_ms: float = 0.0, fail_rate: float = 0.0, token_delay_ms: float = 0.0,
                    seed: int | None = None) -> FastAPI:
    app = FastAPI()
    mock = MockProvider(ProviderConfig("mock", "mock", latency_ms=latency_ms, fail_rate=fail_rate,
                                       token_delay_ms=token_delay_ms), seed=seed)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        req = await request.json()
        if not req.get("stream"):
            try:
                return await mock.complete(req["model"], req)
            except UpstreamError as e:
                return JSONResponse({"error": {"message": str(e)}}, e.status or 500)
        it = mock.stream(req["model"], req).__aiter__()
        try:
            first = await it.__anext__()
        except UpstreamError as e:
            return JSONResponse({"error": {"message": str(e)}}, e.status or 500)
        include_usage = (req.get("stream_options") or {}).get("include_usage")

        async def body():
            yield f"data: {json.dumps(first)}\n\n"
            async for chunk in it:
                if chunk.get("usage") and not include_usage:
                    continue
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    @app.get("/stats")
    async def stats():
        return {"calls": mock.calls}

    return app
