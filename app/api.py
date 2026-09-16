# app/api.py
"""HTTP entrypoint for the research agent.

Run with:
    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

This is a thin wrapper: the endpoint just calls the same build_workflow()
that app/main.py (the CLI) uses, and reshapes the resulting state into JSON.
All the actual logic (planning, research, fact-checking) lives in
app/graph/workflow.py - unchanged by adding this HTTP layer on top of it.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph.workflow import build_workflow
from app.llm import LLMError

app = FastAPI(title="AI Research & Analyst Agent")


class ResearchRequest(BaseModel):
    question: str


class SourceOut(BaseModel):
    source: str
    title: str | None = None


class ClaimOut(BaseModel):
    claim: str
    verdict: str
    explanation: str


class VerificationOut(BaseModel):
    supported: int
    total: int
    revisions: int
    unresolved: list[ClaimOut]


class ResearchResponse(BaseModel):
    question: str
    report: str
    verification: VerificationOut | None
    sources: list[SourceOut]


@lru_cache(maxsize=1)
def _workflow():
    """Compile the graph once and reuse it - compiling is cheap but pointless
    to repeat on every request."""
    return build_workflow()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(payload: ResearchRequest) -> ResearchResponse:
    """Run the full research workflow for one question and return the result.

    This call is synchronous and can take 1-3 minutes: the workflow makes
    several sequential Gemini calls (planner, researcher tool loop, analyst,
    writer, fact-checker, possibly a revision), each paced by our own
    client-side rate limiting. FastAPI runs sync routes in a thread pool, so
    this does not block other requests - but the caller should expect to
    wait.
    """
    try:
        state = _workflow().invoke({"question": payload.question})
    except LLMError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    checks = state.get("fact_checks") or []
    verification = None
    if checks:
        supported = sum(1 for check in checks if check["verdict"] == "supported")
        unresolved = [
            ClaimOut(claim=c["claim"], verdict=c["verdict"], explanation=c["explanation"])
            for c in checks
            if c["verdict"] != "supported"
        ]
        verification = VerificationOut(
            supported=supported,
            total=len(checks),
            revisions=state.get("revisions", 0),
            unresolved=unresolved,
        )

    seen: dict[str, str | None] = {}
    for item in state.get("evidence") or []:
        source = item.get("source")
        if source and source not in seen:
            seen[source] = item.get("title")

    return ResearchResponse(
        question=payload.question,
        report=state.get("final_report", ""),
        verification=verification,
        sources=[SourceOut(source=s, title=t) for s, t in seen.items()],
    )
