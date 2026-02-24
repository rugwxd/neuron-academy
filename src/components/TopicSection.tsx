"use client";

import { ReactNode } from "react";

type SectionType =
  | "plain-english"
  | "math"
  | "code"
  | "see-it"
  | "in-practice"
  | "common-mistakes"
  | "interview"
  | "go-deeper";

const sectionMeta: Record<SectionType, { icon: string; title: string; color: string; bg: string }> = {
  "plain-english": { icon: "💡", title: "In Plain English", color: "text-yellow-600 dark:text-yellow-400", bg: "bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800" },
  math: { icon: "∑", title: "The Math", color: "text-blue-600 dark:text-blue-400", bg: "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800" },
  code: { icon: "⌨", title: "The Code", color: "text-green-600 dark:text-green-400", bg: "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800" },
  "see-it": { icon: "📊", title: "See It", color: "text-purple-600 dark:text-purple-400", bg: "bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800" },
  "in-practice": { icon: "🏭", title: "In Practice", color: "text-indigo-600 dark:text-indigo-400", bg: "bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800" },
  "common-mistakes": { icon: "⚠", title: "Common Mistakes", color: "text-red-600 dark:text-red-400", bg: "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800" },
  interview: { icon: "🎯", title: "Interview Question", color: "text-orange-600 dark:text-orange-400", bg: "bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800" },
  "go-deeper": { icon: "📚", title: "Go Deeper", color: "text-teal-600 dark:text-teal-400", bg: "bg-teal-50 dark:bg-teal-900/20 border-teal-200 dark:border-teal-800" },
};

export default function TopicSection({
  type,
  children,
}: {
  type: SectionType;
  children: ReactNode;
}) {
  const meta = sectionMeta[type];

  return (
    <section className={`rounded-xl border p-6 mb-6 ${meta.bg}`}>
      <h2 className={`text-lg font-bold mb-4 flex items-center gap-2 ${meta.color}`}>
        <span>{meta.icon}</span>
        {meta.title}
      </h2>
      <div className="prose max-w-none">{children}</div>
    </section>
  );
}
