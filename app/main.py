"""Entry point for the AI Research & Analyst Agent."""

from __future__ import annotations

from app.graph.workflow import build_workflow
from app.llm import LLMError
from app.utils.config import ConfigError, get_settings

QUESTION = (
    "A SaaS company has 12,400 customers paying $49/month with 3.2% monthly "
    "churn. What is the annual revenue impact of reducing churn to 2.1%?"
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

    print("\n=== FINAL REPORT ===\n")
    print(result["final_report"])


if __name__ == "__main__":
    main()
