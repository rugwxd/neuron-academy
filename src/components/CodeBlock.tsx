"use client";

import { useState } from "react";

export default function CodeBlock({
  code,
  language = "python",
  title,
}: {
  code: string;
  language?: string;
  title?: string;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group rounded-xl overflow-hidden border border-border mb-4">
      {title && (
        <div className="bg-[#1e293b] text-slate-400 text-xs px-4 py-2 border-b border-slate-600 flex justify-between items-center">
          <span>{title}</span>
          <span className="text-slate-500">{language}</span>
        </div>
      )}
      <div className="relative">
        <pre className="!rounded-none !border-0 !m-0">
          <code className={`language-${language}`}>{code}</code>
        </pre>
        <button
          onClick={handleCopy}
          className="absolute top-2 right-2 p-1.5 rounded-md bg-slate-700/50 text-slate-400 hover:text-slate-200 opacity-0 group-hover:opacity-100 transition-opacity text-xs"
          aria-label="Copy code"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
    </div>
  );
}
