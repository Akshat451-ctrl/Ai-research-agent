import type { ReactNode } from "react";

interface Props {
  report: string;
}

// The report is LLM-generated text with light markdown (### headings,
// * bullets, **bold**). We parse it into React elements directly rather
// than building an HTML string - React escapes all text content by
// default, so there is no dangerouslySetInnerHTML anywhere and therefore
// no way for retrieved document/web text to inject markup.
function renderInline(line: string): ReactNode {
  const parts = line.split(/(\*\*.+?\*\*)/g).filter(Boolean);
  return parts.map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i} className="font-semibold text-slate-100">
        {part.slice(2, -2)}
      </strong>
    ) : (
      <span key={i}>{part}</span>
    ),
  );
}

function parseReport(text: string): ReactNode[] {
  const blocks: ReactNode[] = [];
  let listItems: string[] = [];

  const flushList = () => {
    if (listItems.length === 0) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="ml-5 list-disc space-y-1 text-sm text-slate-300">
        {listItems.map((item, i) => (
          <li key={i}>{renderInline(item)}</li>
        ))}
      </ul>,
    );
    listItems = [];
  };

  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (line.startsWith("### ")) {
      flushList();
      blocks.push(
        <h3 key={blocks.length} className="mt-4 text-base font-semibold text-slate-100 first:mt-0">
          {renderInline(line.slice(4))}
        </h3>,
      );
    } else if (line.startsWith("* ") || line.startsWith("- ")) {
      listItems.push(line.slice(2));
    } else if (line === "") {
      flushList();
    } else {
      flushList();
      blocks.push(
        <p key={blocks.length} className="text-sm leading-relaxed text-slate-300">
          {renderInline(line)}
        </p>,
      );
    }
  }
  flushList();
  return blocks;
}

export function ReportView({ report }: Props) {
  return <div className="space-y-2">{parseReport(report)}</div>;
}
