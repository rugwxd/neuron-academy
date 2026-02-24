import Link from "next/link";
import { curriculum, parts } from "@/lib/curriculum";

const partColorMap: Record<string, { border: string; text: string; bg: string }> = {
  blue: { border: "border-blue-500/30", text: "text-blue-500", bg: "bg-blue-500/10" },
  green: { border: "border-green-500/30", text: "text-green-500", bg: "bg-green-500/10" },
  purple: { border: "border-purple-500/30", text: "text-purple-500", bg: "bg-purple-500/10" },
  orange: { border: "border-orange-500/30", text: "text-orange-500", bg: "bg-orange-500/10" },
  red: { border: "border-red-500/30", text: "text-red-500", bg: "bg-red-500/10" },
  teal: { border: "border-teal-500/30", text: "text-teal-500", bg: "bg-teal-500/10" },
  yellow: { border: "border-yellow-500/30", text: "text-yellow-500", bg: "bg-yellow-500/10" },
  pink: { border: "border-pink-500/30", text: "text-pink-500", bg: "bg-pink-500/10" },
};

export default function Home() {
  return (
    <main className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 via-purple-500/5 to-pink-500/5" />
        <div className="relative max-w-5xl mx-auto px-6 py-20 md:py-32">
          <div className="inline-block mb-4 px-3 py-1 rounded-full bg-accent/10 text-accent text-sm font-medium">
            Free & Open Source
          </div>
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight mb-6 leading-tight">
            Neuron Academy
          </h1>
          <p className="text-xl md:text-2xl text-muted max-w-3xl mb-4 leading-relaxed">
            The most comprehensive data science, machine learning, and AI learning
            resource on the internet.
          </p>
          <p className="text-lg text-muted max-w-3xl mb-10 leading-relaxed">
            From Python basics to RLHF. Every concept explained in plain English,
            then with math, then with working code, then with interactive
            visualizations you can manipulate.
          </p>
          <div className="flex flex-wrap gap-4">
            <Link
              href="/modules/mathematics-for-ml/linear-algebra"
              className="px-6 py-3 rounded-xl bg-accent text-white font-semibold hover:bg-accent-hover transition-colors"
            >
              Start Learning
            </Link>
            <Link
              href="/curriculum"
              className="px-6 py-3 rounded-xl border border-border font-semibold hover:bg-surface-hover transition-colors"
            >
              View Curriculum
            </Link>
          </div>
        </div>
      </section>

      {/* What makes this different */}
      <section className="max-w-5xl mx-auto px-6 py-16">
        <h2 className="text-2xl font-bold mb-8">Every topic follows one structure</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { icon: "💡", title: "Plain English", desc: "Smart but never heard of this" },
            { icon: "∑", title: "The Math", desc: "Formal definitions and derivations" },
            { icon: "⌨", title: "The Code", desc: "Working Python, from scratch + library" },
            { icon: "📊", title: "See It", desc: "Interactive viz — tweak parameters" },
            { icon: "🏭", title: "In Practice", desc: "When to use, when not to" },
            { icon: "⚠", title: "Common Mistakes", desc: "What people get wrong" },
            { icon: "🎯", title: "Interview Q", desc: "Real questions with solutions" },
            { icon: "📚", title: "Go Deeper", desc: "Papers and further reading" },
          ].map((item) => (
            <div
              key={item.title}
              className="p-4 rounded-xl border border-border bg-surface hover:bg-surface-hover transition-colors"
            >
              <div className="text-2xl mb-2">{item.icon}</div>
              <div className="font-semibold text-sm mb-1">{item.title}</div>
              <div className="text-xs text-muted">{item.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Curriculum Overview */}
      <section className="max-w-5xl mx-auto px-6 py-16">
        <h2 className="text-2xl font-bold mb-2">Full Curriculum</h2>
        <p className="text-muted mb-10">
          33 modules, 200+ topics. From zero to L5+ Applied Scientist.
        </p>
        <div className="space-y-8">
          {parts.map((part) => {
            const modules = curriculum.filter(
              (m) => m.partNumber === part.number
            );
            const colors = partColorMap[part.color];
            return (
              <div key={part.number}>
                <h3
                  className={`text-sm font-bold uppercase tracking-wider mb-3 ${colors.text}`}
                >
                  Part {part.number}: {part.title}
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {modules.map((mod) => (
                    <Link
                      key={mod.slug}
                      href={`/modules/${mod.slug}`}
                      className={`group block p-4 rounded-xl border ${colors.border} bg-surface hover:bg-surface-hover transition-colors`}
                    >
                      <div className="flex items-start gap-3">
                        <span
                          className={`text-xs font-mono font-bold ${colors.text} ${colors.bg} px-2 py-0.5 rounded`}
                        >
                          {String(mod.id).padStart(2, "0")}
                        </span>
                        <div>
                          <div className="font-semibold group-hover:text-accent transition-colors">
                            {mod.title}
                          </div>
                          <div className="text-xs text-muted mt-1">
                            {mod.topics.length} topics
                          </div>
                        </div>
                      </div>
                    </Link>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8 text-center text-sm text-muted">
        <p>Neuron Academy — Free & Open Source</p>
        <p className="mt-1">
          Built for everyone who wants to go from zero to mastery.
        </p>
      </footer>
    </main>
  );
}
