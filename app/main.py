"""Entry point for the AI Research & Analyst Agent."""

from __future__ import annotations

from app.graph.workflow import build_workflow
from app.llm import LLMError
from app.utils.config import ConfigError, get_settings

QUESTION = (
    "For Orbitra Mobility, how much FY2026 capital expenditure is budgeted "
    "per new scooter added to the fleet, and what is the biggest risk to "
    "that expansion plan?"
)


def main() -> None:
    print("AI Research Agent is starting...")

    try:
        settings = get_settings()
    except ConfigError as error:
        print(f"\nConfiguration problem:\n{error}")
        return

    print(f"Provider: {settings.llm_provider}")
    print(f"Model:    {settings.model}")

    workflow = build_workflow()

    # Paste this diagram into https://mermaid.live to see the graph.
    print("\n--- workflow ---")
    print(workflow.get_graph().draw_mermaid())

    print(f"\nQ: {QUESTION}")

    try:
        result = workflow.invoke({"question": QUESTION})
    except LLMError as error:
        print(f"\n[failed] {error}")
        return

   # app/main.py
    print("\n=== FINAL REPORT ===\n")
    print(result["final_report"])

    checks = result.get("fact_checks") or []
    if checks:
        supported = sum(1 for check in checks if check["verdict"] == "supported")
        revisions = result.get("revisions", 0)
        print(f"\n=== VERIFICATION ===\n")
        print(f"{supported} of {len(checks)} claims supported after {revisions} revision(s).")

        remaining = [check for check in checks if check["verdict"] != "supported"]
        if remaining:
            print("\nStill unresolved:")
            for check in remaining:
                print(f"  - [{check['verdict']}] {check['claim']}")
                print(f"    {check['explanation']}")


if __name__ == "__main__":
    main()
