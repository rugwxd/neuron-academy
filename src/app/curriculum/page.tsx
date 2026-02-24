import Link from "next/link";
import { curriculum, parts } from "@/lib/curriculum";

export default function CurriculumPage() {
  return (
    <main className="min-h-screen">
      <div className="max-w-5xl mx-auto px-6 py-16">
        <Link href="/" className="text-sm text-muted hover:text-accent transition-colors">
          &larr; Home
        </Link>
        <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight mt-6 mb-4">
          Full Curriculum
        </h1>
        <p className="text-lg text-muted mb-12">
          33 modules covering every topic from Python fundamentals to RLHF alignment.
          Each topic includes plain English explanations, formal math, working code,
          interactive visualizations, and interview prep.
        </p>
        <div className="space-y-12">
          {parts.map((part) => {
            const modules = curriculum.filter((m) => m.partNumber === part.number);
            return (
              <div key={part.number}>
                <h2 className="text-xl font-bold mb-6 pb-2 border-b border-border">
                  Part {part.number}: {part.title}
                </h2>
                <div className="space-y-6">
                  {modules.map((mod) => (
                    <div key={mod.slug}>
                      <Link
                        href={`/modules/${mod.slug}`}
                        className="text-lg font-semibold hover:text-accent transition-colors"
                      >
                        <span className="font-mono text-muted text-sm mr-2">
                          {String(mod.id).padStart(2, "0")}
                        </span>
                        {mod.title}
                      </Link>
                      <div className="ml-8 mt-2 space-y-1">
                        {mod.topics.map((topic) => (
                          <Link
                            key={topic.slug}
                            href={`/modules/${mod.slug}/${topic.slug}`}
                            className="block text-sm text-muted hover:text-accent transition-colors py-0.5"
                          >
                            {topic.title}
                            <span className="ml-2 text-xs">— {topic.description}</span>
                          </Link>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </main>
  );
}
