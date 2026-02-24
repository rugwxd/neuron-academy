import Link from "next/link";
import { notFound } from "next/navigation";
import { curriculum, getModule, parts } from "@/lib/curriculum";
import Sidebar from "@/components/Sidebar";
import MobileNav from "@/components/MobileNav";

export function generateStaticParams() {
  return curriculum.map((mod) => ({ moduleSlug: mod.slug }));
}

export default async function ModulePage({
  params,
}: {
  params: Promise<{ moduleSlug: string }>;
}) {
  const { moduleSlug } = await params;
  const mod = getModule(moduleSlug);
  if (!mod) notFound();

  const part = parts.find((p) => p.number === mod.partNumber);

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <MobileNav />
      <main className="flex-1 min-w-0">
        <div className="max-w-4xl mx-auto px-6 py-12 lg:py-16">
          <div className="mb-8">
            <Link
              href="/"
              className="text-sm text-muted hover:text-accent transition-colors"
            >
              Home
            </Link>
            <span className="text-muted mx-2">/</span>
            <span className="text-sm text-muted">
              Part {mod.partNumber}: {part?.title}
            </span>
          </div>
          <div className="mb-2 text-sm font-mono text-accent font-bold">
            Module {String(mod.id).padStart(2, "0")}
          </div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight mb-4">
            {mod.title}
          </h1>
          <p className="text-lg text-muted mb-10">
            {mod.topics.length} topics in this module
          </p>
          <div className="space-y-3">
            {mod.topics.map((topic, i) => (
              <Link
                key={topic.slug}
                href={`/modules/${mod.slug}/${topic.slug}`}
                className="group block p-5 rounded-xl border border-border bg-surface hover:bg-surface-hover transition-colors"
              >
                <div className="flex items-start gap-4">
                  <span className="text-sm font-mono text-muted font-bold mt-0.5">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <div>
                    <div className="font-semibold text-lg group-hover:text-accent transition-colors">
                      {topic.title}
                    </div>
                    <div className="text-sm text-muted mt-1">
                      {topic.description}
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
