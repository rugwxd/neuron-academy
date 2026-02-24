"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { curriculum, parts } from "@/lib/curriculum";

const partColors: Record<string, string> = {
  blue: "text-blue-500",
  green: "text-green-500",
  purple: "text-purple-500",
  orange: "text-orange-500",
  red: "text-red-500",
  teal: "text-teal-500",
  yellow: "text-yellow-500",
  pink: "text-pink-500",
};

const partBgColors: Record<string, string> = {
  blue: "bg-blue-500/10",
  green: "bg-green-500/10",
  purple: "bg-purple-500/10",
  orange: "bg-orange-500/10",
  red: "bg-red-500/10",
  teal: "bg-teal-500/10",
  yellow: "bg-yellow-500/10",
  pink: "bg-pink-500/10",
};

export default function Sidebar() {
  const pathname = usePathname();
  const [expandedModules, setExpandedModules] = useState<Set<string>>(
    new Set()
  );

  const toggleModule = (slug: string) => {
    setExpandedModules((prev) => {
      const next = new Set(prev);
      if (next.has(slug)) {
        next.delete(slug);
      } else {
        next.add(slug);
      }
      return next;
    });
  };

  return (
    <aside className="w-72 border-r border-border bg-surface overflow-y-auto h-screen sticky top-0 shrink-0 hidden lg:block">
      <div className="p-4 border-b border-border">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-2xl font-bold tracking-tight">
            NA
          </span>
          <span className="text-xs text-muted font-medium uppercase tracking-wider">
            Neuron Academy
          </span>
        </Link>
      </div>
      <nav className="p-3">
        {parts.map((part) => {
          const modules = curriculum.filter(
            (m) => m.partNumber === part.number
          );
          return (
            <div key={part.number} className="mb-4">
              <div
                className={`text-xs font-bold uppercase tracking-wider px-2 py-1.5 rounded ${partColors[part.color]} ${partBgColors[part.color]}`}
              >
                Part {part.number}: {part.title}
              </div>
              <div className="mt-1">
                {modules.map((mod) => {
                  const isExpanded = expandedModules.has(mod.slug);
                  const isActive = pathname.startsWith(
                    `/modules/${mod.slug}`
                  );
                  return (
                    <div key={mod.slug}>
                      <button
                        onClick={() => toggleModule(mod.slug)}
                        className={`w-full text-left px-2 py-1.5 rounded text-sm flex items-center justify-between hover:bg-surface-hover transition-colors ${
                          isActive ? "text-accent font-medium" : "text-foreground"
                        }`}
                      >
                        <span className="flex items-center gap-2">
                          <span className="text-muted text-xs font-mono">
                            {String(mod.id).padStart(2, "0")}
                          </span>
                          <span className="truncate">{mod.title}</span>
                        </span>
                        <svg
                          className={`w-3 h-3 text-muted transition-transform ${
                            isExpanded ? "rotate-90" : ""
                          }`}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 5l7 7-7 7"
                          />
                        </svg>
                      </button>
                      {isExpanded && (
                        <div className="ml-6 border-l border-border">
                          {mod.topics.map((topic) => {
                            const topicPath = `/modules/${mod.slug}/${topic.slug}`;
                            const isTopicActive = pathname === topicPath;
                            return (
                              <Link
                                key={topic.slug}
                                href={topicPath}
                                className={`block px-3 py-1 text-sm hover:text-accent transition-colors ${
                                  isTopicActive
                                    ? "text-accent font-medium border-l-2 border-accent -ml-px"
                                    : "text-muted"
                                }`}
                              >
                                {topic.title}
                              </Link>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
