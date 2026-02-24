import { notFound } from "next/navigation";
import Link from "next/link";
import { curriculum, getTopic, parts } from "@/lib/curriculum";
import Sidebar from "@/components/Sidebar";
import MobileNav from "@/components/MobileNav";

// Map of available topic content components
// We dynamically import these to keep the bundle small
const topicComponents: Record<string, () => Promise<{ default: React.ComponentType }>> = {
  "mathematics-for-ml/linear-algebra": () => import("@/content/mathematics-for-ml/linear-algebra"),
  "mathematics-for-ml/calculus-and-optimization": () => import("@/content/mathematics-for-ml/calculus-and-optimization"),
  "supervised-learning/linear-regression": () => import("@/content/supervised-learning/linear-regression"),
  "supervised-learning/decision-trees": () => import("@/content/supervised-learning/decision-trees"),
  "supervised-learning/bias-variance": () => import("@/content/supervised-learning/bias-variance"),
  "transformers/self-attention": () => import("@/content/transformers/self-attention"),
  "statistical-foundations/central-limit-theorem": () => import("@/content/statistical-foundations/central-limit-theorem"),
};

export function generateStaticParams() {
  const params: { moduleSlug: string; topicSlug: string }[] = [];
  for (const mod of curriculum) {
    for (const topic of mod.topics) {
      params.push({ moduleSlug: mod.slug, topicSlug: topic.slug });
    }
  }
  return params;
}

export default async function TopicPage({
  params,
}: {
  params: Promise<{ moduleSlug: string; topicSlug: string }>;
}) {
  const { moduleSlug, topicSlug } = await params;
  const result = getTopic(moduleSlug, topicSlug);
  if (!result) notFound();

  const { module: mod, topic } = result;
  const part = parts.find((p) => p.number === mod.partNumber);

  // Find previous/next topics
  const topicIndex = mod.topics.findIndex((t) => t.slug === topicSlug);
  const prevTopic = topicIndex > 0 ? mod.topics[topicIndex - 1] : null;
  const nextTopic = topicIndex < mod.topics.length - 1 ? mod.topics[topicIndex + 1] : null;

  // Try to load the content component
  const contentKey = `${moduleSlug}/${topicSlug}`;
  const hasContent = contentKey in topicComponents;
  let ContentComponent: React.ComponentType | null = null;

  if (hasContent) {
    const mod = await topicComponents[contentKey]();
    ContentComponent = mod.default;
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <MobileNav />
      <main className="flex-1 min-w-0">
        <div className="max-w-4xl mx-auto px-6 py-12 lg:py-16">
          {/* Breadcrumbs */}
          <div className="mb-8 flex flex-wrap items-center gap-2 text-sm">
            <Link href="/" className="text-muted hover:text-accent transition-colors">
              Home
            </Link>
            <span className="text-muted">/</span>
            <span className="text-muted">Part {mod.partNumber}: {part?.title}</span>
            <span className="text-muted">/</span>
            <Link
              href={`/modules/${mod.slug}`}
              className="text-muted hover:text-accent transition-colors"
            >
              {mod.title}
            </Link>
          </div>

          {/* Header */}
          <div className="mb-10">
            <div className="text-sm font-mono text-accent font-bold mb-2">
              Module {String(mod.id).padStart(2, "0")} &middot; {topic.title}
            </div>
            <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight mb-3">
              {topic.title}
            </h1>
            <p className="text-lg text-muted">{topic.description}</p>
          </div>

          {/* Content */}
          {ContentComponent ? (
            <ContentComponent />
          ) : (
            <div className="p-8 rounded-xl border border-border bg-surface text-center">
              <div className="text-4xl mb-4">🚧</div>
              <h2 className="text-xl font-bold mb-2">Coming Soon</h2>
              <p className="text-muted">
                This topic is being written. Check back soon for the full content
                with interactive visualizations.
              </p>
            </div>
          )}

          {/* Navigation */}
          <div className="flex justify-between items-center mt-16 pt-8 border-t border-border">
            {prevTopic ? (
              <Link
                href={`/modules/${mod.slug}/${prevTopic.slug}`}
                className="group flex items-center gap-2 text-sm text-muted hover:text-accent transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                <div>
                  <div className="text-xs text-muted">Previous</div>
                  <div className="font-medium group-hover:text-accent">{prevTopic.title}</div>
                </div>
              </Link>
            ) : (
              <div />
            )}
            {nextTopic ? (
              <Link
                href={`/modules/${mod.slug}/${nextTopic.slug}`}
                className="group flex items-center gap-2 text-sm text-muted hover:text-accent transition-colors text-right"
              >
                <div>
                  <div className="text-xs text-muted">Next</div>
                  <div className="font-medium group-hover:text-accent">{nextTopic.title}</div>
                </div>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </Link>
            ) : (
              <div />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
