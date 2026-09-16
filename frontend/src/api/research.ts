import type { ResearchResponse } from "../types";

export class ResearchError extends Error {}

export async function runResearch(question: string): Promise<ResearchResponse> {
  const response = await fetch("/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ResearchError(body?.detail ?? `Request failed (${response.status})`);
  }

  return response.json();
}
