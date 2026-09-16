import { useState } from "react";
import { QuestionForm } from "./components/QuestionForm";
import { StatusBar } from "./components/StatusBar";
import { ReportView } from "./components/ReportView";
import { VerificationPanel } from "./components/VerificationPanel";
import { SourcesList } from "./components/SourcesList";
import { Card } from "./components/Card";
import { runResearch, ResearchError } from "./api/research";
import type { ResearchResponse } from "./types";

export default function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(question: string) {
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await runResearch(question);
      setResult(data);
    } catch (err) {
      setError(
        err instanceof ResearchError
          ? err.message
          : "Something went wrong. Is the API server running on port 8000?",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 px-4 py-10 text-slate-100">
      <div className="mx-auto max-w-2xl space-y-6">
        <header>
          <h1 className="text-xl font-semibold">AI Research & Analyst Agent</h1>
          <p className="mt-1 text-sm text-slate-400">
            Plans, researches, calculates, writes, and fact-checks its own report before answering.
          </p>
        </header>

        <QuestionForm onSubmit={handleSubmit} isLoading={isLoading} />

        {isLoading && <StatusBar />}
        {error && <p className="text-sm text-red-400">{error}</p>}

        {result && (
          <div className="space-y-4">
            <Card title="Report">
              <ReportView report={result.report} />
            </Card>

            {result.verification && (
              <Card title="Verification">
                <VerificationPanel verification={result.verification} />
              </Card>
            )}

            {result.sources.length > 0 && (
              <Card title="Sources">
                <SourcesList sources={result.sources} />
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
