import { useState, type FormEvent } from "react";

interface Props {
  onSubmit: (question: string) => void;
  isLoading: boolean;
}

const EXAMPLE =
  "For Orbitra Mobility, how much FY2026 capital expenditure is budgeted per new scooter added to the fleet?";

export function QuestionForm({ onSubmit, isLoading }: Props) {
  const [question, setQuestion] = useState("");

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (trimmed) onSubmit(trimmed);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder={EXAMPLE}
        rows={4}
        disabled={isLoading}
        className="w-full resize-y rounded-lg border border-slate-700 bg-slate-900 p-3 text-sm text-slate-100 placeholder:text-slate-500 focus:border-blue-500 focus:outline-none disabled:opacity-50"
      />
      <button
        type="submit"
        disabled={isLoading || !question.trim()}
        className="rounded-lg bg-blue-600 px-5 py-2 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {isLoading ? "Researching..." : "Run research"}
      </button>
    </form>
  );
}
