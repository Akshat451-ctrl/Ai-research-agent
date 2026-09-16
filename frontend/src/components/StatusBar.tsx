import { useEffect, useState } from "react";

// The workflow makes several sequential, rate-limited LLM calls, so a
// research run typically takes 1-3 minutes. An elapsed timer tells the user
// something is actually happening instead of leaving them staring at a
// static spinner wondering if the page has hung.
export function StatusBar() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const start = Date.now();
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex items-center gap-2 text-sm text-slate-400">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-slate-700 border-t-blue-500" />
      <span>Working... {elapsed}s (this can take 1-3 minutes)</span>
    </div>
  );
}
