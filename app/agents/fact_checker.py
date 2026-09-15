# app/agents/fact_checker.py
"""Fact-checker: verifies a written report against the raw evidence behind it.

The report writer only sees a summary of findings, so it can subtly drift
from the source (wrong number, dropped caveat, changed date). This module
re-reads the RAW evidence and calculations - not the summary - and checks
every claim against them, so drift gets caught instead of silently shipping.

The checker never performs arithmetic itself. It only ever writes a
corrected_expression; the sandboxed calculate() tool actually runs it. This
keeps every number in the system traceable to either the source documents or
a real, auditable calculation.

Run a demo:
    python -m app.agents.fact_checker
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from app.llm import ask_json
from app.tools.calculator import CalculatorError, calculate

CHECK_SYSTEM = (
    "You are a strict fact-checker. You will be given a written report, the "
    "raw evidence passages it was supposed to be based on, and the raw "
    "calculations that were actually performed. "
    "Break the report into its individual factual and numeric claims. "
    "For each claim, decide a verdict: "
    "'supported' - the evidence or calculations directly back it up; "
    "'contradicted' - the evidence or calculations show a different value "
    "or fact; "
    "'unsupported' - nothing in the evidence or calculations addresses it. "
    "Quote the exact evidence sentence in evidence_quote, or write 'none'. "
    "Never compute anything yourself. If a claim is wrong because of bad "
    "arithmetic, write the correct arithmetic expression in "
    "corrected_expression, using only plain numbers and + - * / ** % ( ) - "
    "no words, no units. Note: 1 crore = 10,000,000, so convert Indian "
    "numbering to plain numbers first, e.g. '71 crore' becomes 710000000."
)


class ClaimCheck(BaseModel):
    """The verdict on one factual or numeric claim from the report."""

    claim: str
    verdict: Literal["supported", "contradicted", "unsupported"]
    evidence_quote: str
    explanation: str
    correction: str | None = None
    corrected_expression: str | None = None


class FactCheckReport(BaseModel):
    """The full set of claim-by-claim verdicts for one report."""

    checks: list[ClaimCheck]


# Matches numbers like 84,523.81 or 1200 - used by the rounding guard below.
_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _numbers_in(text: str) -> list[float]:
    """Pull every number out of a piece of text, ignoring thousands commas."""
    numbers = []
    for match in _NUMBER_PATTERN.findall(text):
        try:
            numbers.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return numbers


def _apply_rounding_guard(check: ClaimCheck) -> None:
    """Undo a false 'contradicted' verdict caused by rounding, not error.

    The checker sometimes flags a claim like "...84,524..." as contradicted
    against a recomputed 84,523.81, purely because the report rounded. If a
    number already in the claim matches the corrected expression's result
    within 0.5% (or 1 unit for small numbers), the claim was right all
    along - fix the verdict instead of forcing a pointless revision.
    """
    if check.verdict != "contradicted" or not check.corrected_expression:
        return

    try:
        correct_value = calculate(check.corrected_expression)
    except CalculatorError:
        return

    tolerance = max(abs(correct_value) * 0.005, 1.0)
    for number in _numbers_in(check.claim):
        if abs(number - correct_value) <= tolerance:
            check.verdict = "supported"
            check.correction = None
            check.corrected_expression = None
            return


def format_evidence(evidence: list[dict]) -> str:
    """Turn raw, possibly-duplicated search matches into a numbered block."""
    seen: dict[str, str] = {}
    for item in evidence:
        text = (item.get("text") or "").strip()
        if text and text not in seen:
            seen[text] = item.get("source", "unknown")

    return "\n\n".join(
        f"[E{index}] ({source}) {text}"
        for index, (text, source) in enumerate(seen.items(), start=1)
    )


def format_problems(checks: list[ClaimCheck]) -> str:
    """Turn contradicted/unsupported verdicts into instructions for the writer."""
    problems = [check for check in checks if check.verdict != "supported"]
    if not problems:
        return ""

    lines = []
    for check in problems:
        line = f"- CLAIM: {check.claim}\n  PROBLEM ({check.verdict}): {check.explanation}"
        if check.correction:
            line += f"\n  CORRECTION: {check.correction}"
        lines.append(line)
    return "\n\n".join(lines)


def print_checks(checks: list[ClaimCheck]) -> None:
    """Pretty-print verdicts to the console."""
    for check in checks:
        print(f"[{check.verdict.upper()}] {check.claim}")
        if check.verdict != "supported":
            print(f"    reason: {check.explanation}")
            if check.correction:
                print(f"    correction: {check.correction}")


def check_report(report: str, evidence: list[dict], calculations: list[dict]) -> list[ClaimCheck]:
    """Verify every claim in `report` against the raw evidence and calculations."""
    evidence_block = format_evidence(evidence) or "(no evidence was retrieved)"
    calc_block = "\n".join(
        f"[C{i}] {c['expression']} = {c['result']}"
        for i, c in enumerate(calculations, start=1)
    ) or "(no calculations were performed)"

    prompt = (
        f"REPORT:\n{report}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"CALCULATIONS PERFORMED:\n{calc_block}"
    )

    result = ask_json(prompt, FactCheckReport, system=CHECK_SYSTEM)
    for check in result.checks:
        _apply_rounding_guard(check)
    return result.checks


def main() -> None:
    """Demo: fact-check a report containing the known Orbitra capex bug."""
    demo_evidence = [
        {
            "source": "orbitra_strategy_brief.md",
            "score": 0.796,
            "text": (
                "Capital expenditure for FY2026 is budgeted at 96 crore "
                "rupees, of which 71 crore is allocated to new scooters and "
                "14 crore to the two service centres."
            ),
        },
        {
            "source": "orbitra_strategy_brief.md",
            "score": 0.752,
            "text": "Grow the deployed fleet from 18,600 to 27,000 scooters.",
        },
    ]
    demo_calculations = [
        {"expression": "27000 - 18600", "result": 8400.0},
        {"expression": "960000000 / 8400", "result": 114285.71428571429},
    ]
    demo_report = (
        "Orbitra Mobility has budgeted approximately 114,286 rupees in "
        "capital expenditure for each new scooter added to the fleet in "
        "FY2026."
    )

    print("--- REPORT UNDER TEST ---")
    print(demo_report)
    print("\n--- CHECKING ---")

    checks = check_report(demo_report, demo_evidence, demo_calculations)
    print_checks(checks)

    problems = format_problems(checks)
    print("\n--- PROBLEMS FOR REVISION ---")
    print(problems or "All claims supported.")


if __name__ == "__main__":
    main()
