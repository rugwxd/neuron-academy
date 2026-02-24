"use client";

import { useState } from "react";
import Link from "next/link";
import { curriculum, parts } from "@/lib/curriculum";

export default function MobileNav() {
  const [open, setOpen] = useState(false);

  return (
    <div className="lg:hidden">
      <button
        onClick={() => setOpen(!open)}
        className="fixed top-4 left-4 z-50 p-2 rounded-lg bg-surface border border-border shadow-lg"
        aria-label="Toggle navigation"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          {open ? (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          )}
        </svg>
      </button>
      {open && (
        <>
          <div className="fixed inset-0 bg-black/50 z-40" onClick={() => setOpen(false)} />
          <div className="fixed inset-y-0 left-0 w-72 bg-surface border-r border-border z-50 overflow-y-auto">
            <div className="p-4 border-b border-border">
              <span className="text-xl font-bold">Neuron Academy</span>
            </div>
            <nav className="p-3">
              {parts.map((part) => {
                const modules = curriculum.filter((m) => m.partNumber === part.number);
                return (
                  <div key={part.number} className="mb-3">
                    <div className="text-xs font-bold uppercase tracking-wider text-muted px-2 py-1">
                      Part {part.number}: {part.title}
                    </div>
                    {modules.map((mod) => (
                      <Link
                        key={mod.slug}
                        href={`/modules/${mod.slug}`}
                        onClick={() => setOpen(false)}
                        className="block px-2 py-1.5 text-sm hover:text-accent transition-colors"
                      >
                        <span className="text-muted text-xs font-mono mr-2">
                          {String(mod.id).padStart(2, "0")}
                        </span>
                        {mod.title}
                      </Link>
                    ))}
                  </div>
                );
              })}
            </nav>
          </div>
        </>
      )}
    </div>
  );
}
