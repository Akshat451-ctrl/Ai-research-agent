import type { VerificationOut } from "../types";

export function VerificationPanel({ verification }: { verification: VerificationOut }) {
  const allSupported = verification.unresolved.length === 0;

  return (
    <div>
      <span
        className={`inline-block rounded-full px-3 py-1 text-xs font-semibold ${
          allSupported ? "bg-emerald-500/15 text-emerald-400" : "bg-red-500/15 text-red-400"
        }`}
      >
        {verification.supported} of {verification.total} claims supported after{" "}
        {verification.revisions} revision(s)
      </span>

      {!allSupported && (
        <div className="mt-3 space-y-2">
          {verification.unresolved.map((claim, i) => (
            <div key={i} className="rounded-lg border border-slate-800 bg-slate-900/60 p-3 text-xs">
              <p className="font-semibold text-red-400">
                [{claim.verdict}] {claim.claim}
              </p>
              <p className="mt-1 text-slate-400">{claim.explanation}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
