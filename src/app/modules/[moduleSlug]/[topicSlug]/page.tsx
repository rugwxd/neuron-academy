import { notFound } from "next/navigation";
import Link from "next/link";
import { curriculum, getTopic, parts } from "@/lib/curriculum";
import Sidebar from "@/components/Sidebar";
import MobileNav from "@/components/MobileNav";

// Map of available topic content components
// We dynamically import these to keep the bundle small
const topicComponents: Record<string, () => Promise<{ default: React.ComponentType }>> = {
  "advanced-topics/knowledge-distillation": () => import("@/content/advanced-topics/knowledge-distillation"),
  "advanced-topics/multi-modal": () => import("@/content/advanced-topics/multi-modal"),
  "advanced-topics/self-supervised": () => import("@/content/advanced-topics/self-supervised"),
  "bayesian-statistics/bayes-theorem": () => import("@/content/bayesian-statistics/bayes-theorem"),
  "bayesian-statistics/mcmc": () => import("@/content/bayesian-statistics/mcmc"),
  "cnns/architectures": () => import("@/content/cnns/architectures"),
  "cnns/convolution": () => import("@/content/cnns/convolution"),
  "data-manipulation/data-cleaning": () => import("@/content/data-manipulation/data-cleaning"),
  "data-manipulation/numpy": () => import("@/content/data-manipulation/numpy"),
  "data-manipulation/pandas": () => import("@/content/data-manipulation/pandas"),
  "distributed-computing/distributed-training": () => import("@/content/distributed-computing/distributed-training"),
  "distributed-computing/spark": () => import("@/content/distributed-computing/spark"),
  "ensemble-methods/bagging-vs-boosting": () => import("@/content/ensemble-methods/bagging-vs-boosting"),
  "ensemble-methods/xgboost-internals": () => import("@/content/ensemble-methods/xgboost-internals"),
  "evaluation-and-tuning/cross-validation": () => import("@/content/evaluation-and-tuning/cross-validation"),
  "evaluation-and-tuning/hyperparameter-tuning": () => import("@/content/evaluation-and-tuning/hyperparameter-tuning"),
  "evaluation-and-tuning/metrics": () => import("@/content/evaluation-and-tuning/metrics"),
  "experimentation/ab-testing": () => import("@/content/experimentation/ab-testing"),
  "experimentation/causal-inference": () => import("@/content/experimentation/causal-inference"),
  "feature-engineering/encoding-strategies": () => import("@/content/feature-engineering/encoding-strategies"),
  "feature-engineering/feature-stores": () => import("@/content/feature-engineering/feature-stores"),
  "generative-models/diffusion": () => import("@/content/generative-models/diffusion"),
  "generative-models/gans": () => import("@/content/generative-models/gans"),
  "generative-models/vaes": () => import("@/content/generative-models/vaes"),
  "graph-ml/gnns": () => import("@/content/graph-ml/gnns"),
  "graph-ml/graph-basics": () => import("@/content/graph-ml/graph-basics"),
  "interpretability/fairness": () => import("@/content/interpretability/fairness"),
  "interpretability/shap": () => import("@/content/interpretability/shap"),
  "interview-questions/ml-system-design-questions": () => import("@/content/interview-questions/ml-system-design-questions"),
  "interview-questions/probability-puzzles": () => import("@/content/interview-questions/probability-puzzles"),
  "interview-questions/sql-challenges": () => import("@/content/interview-questions/sql-challenges"),
  "large-language-models/fine-tuning": () => import("@/content/large-language-models/fine-tuning"),
  "large-language-models/pretraining": () => import("@/content/large-language-models/pretraining"),
  "large-language-models/prompt-engineering": () => import("@/content/large-language-models/prompt-engineering"),
  "large-language-models/rag": () => import("@/content/large-language-models/rag"),
  "large-language-models/rlhf": () => import("@/content/large-language-models/rlhf"),
  "mathematics-for-ml/calculus-and-optimization": () => import("@/content/mathematics-for-ml/calculus-and-optimization"),
  "mathematics-for-ml/information-theory": () => import("@/content/mathematics-for-ml/information-theory"),
  "mathematics-for-ml/linear-algebra": () => import("@/content/mathematics-for-ml/linear-algebra"),
  "mathematics-for-ml/optimization-theory": () => import("@/content/mathematics-for-ml/optimization-theory"),
  "mathematics-for-ml/probability": () => import("@/content/mathematics-for-ml/probability"),
  "ml-system-design/case-studies": () => import("@/content/ml-system-design/case-studies"),
  "ml-system-design/framework": () => import("@/content/ml-system-design/framework"),
  "mlops/model-serving": () => import("@/content/mlops/model-serving"),
  "mlops/monitoring": () => import("@/content/mlops/monitoring"),
  "neural-networks/activation-functions": () => import("@/content/neural-networks/activation-functions"),
  "neural-networks/backpropagation": () => import("@/content/neural-networks/backpropagation"),
  "neural-networks/optimizers": () => import("@/content/neural-networks/optimizers"),
  "neural-networks/perceptron": () => import("@/content/neural-networks/perceptron"),
  "nlp-foundations/embeddings": () => import("@/content/nlp-foundations/embeddings"),
  "nlp-foundations/tokenization": () => import("@/content/nlp-foundations/tokenization"),
  "python-for-data-science/control-flow": () => import("@/content/python-for-data-science/control-flow"),
  "python-for-data-science/functions": () => import("@/content/python-for-data-science/functions"),
  "python-for-data-science/oop": () => import("@/content/python-for-data-science/oop"),
  "python-for-data-science/variables-and-types": () => import("@/content/python-for-data-science/variables-and-types"),
  "pytorch/tensors": () => import("@/content/pytorch/tensors"),
  "pytorch/training-loop": () => import("@/content/pytorch/training-loop"),
  "recommendation-systems/collaborative-filtering": () => import("@/content/recommendation-systems/collaborative-filtering"),
  "recommendation-systems/deep-recsys": () => import("@/content/recommendation-systems/deep-recsys"),
  "reinforcement-learning/mdps": () => import("@/content/reinforcement-learning/mdps"),
  "reinforcement-learning/policy-gradients": () => import("@/content/reinforcement-learning/policy-gradients"),
  "reinforcement-learning/q-learning": () => import("@/content/reinforcement-learning/q-learning"),
  "search-and-ranking/information-retrieval": () => import("@/content/search-and-ranking/information-retrieval"),
  "search-and-ranking/learning-to-rank": () => import("@/content/search-and-ranking/learning-to-rank"),
  "sequence-models/rnns": () => import("@/content/sequence-models/rnns"),
  "sequence-models/seq2seq": () => import("@/content/sequence-models/seq2seq"),
  "sql-mastery/advanced-patterns": () => import("@/content/sql-mastery/advanced-patterns"),
  "sql-mastery/fundamentals": () => import("@/content/sql-mastery/fundamentals"),
  "sql-mastery/joins": () => import("@/content/sql-mastery/joins"),
  "sql-mastery/window-functions": () => import("@/content/sql-mastery/window-functions"),
  "statistical-foundations/central-limit-theorem": () => import("@/content/statistical-foundations/central-limit-theorem"),
  "statistical-foundations/descriptive-stats": () => import("@/content/statistical-foundations/descriptive-stats"),
  "statistical-foundations/distributions": () => import("@/content/statistical-foundations/distributions"),
  "statistical-foundations/maximum-likelihood": () => import("@/content/statistical-foundations/maximum-likelihood"),
  "statistical-inference/confidence-intervals": () => import("@/content/statistical-inference/confidence-intervals"),
  "statistical-inference/hypothesis-testing": () => import("@/content/statistical-inference/hypothesis-testing"),
  "statistical-inference/multiple-testing": () => import("@/content/statistical-inference/multiple-testing"),
  "statistical-inference/t-tests": () => import("@/content/statistical-inference/t-tests"),
  "supervised-learning/bias-variance": () => import("@/content/supervised-learning/bias-variance"),
  "supervised-learning/decision-trees": () => import("@/content/supervised-learning/decision-trees"),
  "supervised-learning/gradient-boosting": () => import("@/content/supervised-learning/gradient-boosting"),
  "supervised-learning/linear-regression": () => import("@/content/supervised-learning/linear-regression"),
  "supervised-learning/logistic-regression": () => import("@/content/supervised-learning/logistic-regression"),
  "supervised-learning/random-forests": () => import("@/content/supervised-learning/random-forests"),
  "supervised-learning/svm": () => import("@/content/supervised-learning/svm"),
  "time-series/arima": () => import("@/content/time-series/arima"),
  "time-series/components": () => import("@/content/time-series/components"),
  "transformers/full-architecture": () => import("@/content/transformers/full-architecture"),
  "transformers/positional-encoding": () => import("@/content/transformers/positional-encoding"),
  "transformers/scaling-laws": () => import("@/content/transformers/scaling-laws"),
  "transformers/self-attention": () => import("@/content/transformers/self-attention"),
  "unsupervised-learning/k-means": () => import("@/content/unsupervised-learning/k-means"),
  "unsupervised-learning/pca": () => import("@/content/unsupervised-learning/pca"),
  "unsupervised-learning/t-sne-umap": () => import("@/content/unsupervised-learning/t-sne-umap"),
  "visualization-and-eda/eda-methodology": () => import("@/content/visualization-and-eda/eda-methodology"),
  "visualization-and-eda/matplotlib": () => import("@/content/visualization-and-eda/matplotlib"),
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
