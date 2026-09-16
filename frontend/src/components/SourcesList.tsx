import type { SourceOut } from "../types";

export function SourcesList({ sources }: { sources: SourceOut[] }) {
  return (
    <ul className="divide-y divide-slate-800 text-sm">
      {sources.map((s, i) => (
        <li key={i} className="py-2">
          {s.source.startsWith("http") ? (
            <a
              href={s.source}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:underline"
            >
              {s.title || s.source}
            </a>
          ) : (
            <span className="text-slate-300">{s.source}</span>
          )}
        </li>
      ))}
    </ul>
  );
}
