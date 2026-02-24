"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Framework() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          ML system design interviews test whether you can go from a vague problem (&quot;build a recommendation system&quot;) to a concrete, deployable ML architecture. It&apos;s not about knowing the fanciest model &mdash; it&apos;s about showing you can think through the <em>entire system</em>.
        </p>
        <p>
          The framework has seven stages, each building on the last:
        </p>
        <ol>
          <li><strong>Clarify Requirements</strong> &mdash; What exactly are we optimizing? What are the constraints (latency, scale, budget)?</li>
          <li><strong>Define the ML Problem</strong> &mdash; Is this classification, ranking, generation? What&apos;s the input/output? What&apos;s the label?</li>
          <li><strong>Design the Data Pipeline</strong> &mdash; Where does data come from? How do we label it? How do we store and process features?</li>
          <li><strong>Feature Engineering</strong> &mdash; What signals matter? Offline vs online features? How do we avoid training/serving skew?</li>
          <li><strong>Choose the Model</strong> &mdash; Start simple (logistic regression), explain when to go complex (deep learning). Justify your choice.</li>
          <li><strong>Design Serving Infrastructure</strong> &mdash; Batch vs. real-time? How do we handle latency? How do we deploy and rollback?</li>
          <li><strong>Plan Evaluation, Monitoring &amp; Iteration</strong> &mdash; Offline metrics, online A/B tests, monitoring for data drift, and retraining cadence.</li>
        </ol>
        <p>
          The key insight: <strong>most ML system failures are data problems, not model problems.</strong> Interviewers want to see that you think about data quality, feedback loops, and operational concerns &mdash; not just model architecture.
        </p>
        <p>
          Budget your 45-minute interview roughly as: requirements (5 min), ML formulation (5 min), data pipeline and features (10 min), model (5 min), serving (10 min), evaluation and monitoring (10 min). Spend the most time on what differentiates your design.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Latency-Throughput Tradeoff</h3>
        <p>For a system serving <InlineMath math="N" /> requests per second, each taking <InlineMath math="T" /> seconds:</p>
        <BlockMath math="\text{Throughput} = \frac{N_{\text{workers}}}{T_{\text{per request}}}" />
        <p>Batching increases throughput but adds latency. The optimal batch size balances these two forces.</p>

        <h3>Amdahl&apos;s Law for System Optimization</h3>
        <p>The maximum speedup from parallelizing a system with serial fraction <InlineMath math="s" />:</p>
        <BlockMath math="\text{Speedup} = \frac{1}{s + \frac{1-s}{p}}" />
        <p>where <InlineMath math="p" /> is the number of parallel processors. Even with infinite parallelism, speedup is bounded by <InlineMath math="\frac{1}{s}" />. This tells you where to focus optimization: always fix the serial bottleneck first.</p>

        <h3>Little&apos;s Law for Capacity Planning</h3>
        <BlockMath math="L = \lambda \times W" />
        <p>where <InlineMath math="L" /> is average items in the system, <InlineMath math="\lambda" /> is arrival rate, and <InlineMath math="W" /> is average processing time. Use this to estimate how many servers you need: if each request takes 100ms and you get 10K requests/second, you need at least 1,000 concurrent workers.</p>

        <h3>A/B Test Sample Size</h3>
        <p>Minimum sample size per group:</p>
        <BlockMath math="n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}" />
        <p>where <InlineMath math="\delta" /> is the minimum detectable effect and <InlineMath math="\sigma^2" /> is the variance. Detecting a 1% lift in CTR requires ~16x more samples than detecting a 4% lift.</p>

        <h3>Feature Importance via Information Gain</h3>
        <BlockMath math="IG(Y, X) = H(Y) - H(Y|X) = H(Y) - \sum_{x} P(X=x) \cdot H(Y|X=x)" />
        <p>Information gain measures how much a feature <InlineMath math="X" /> reduces uncertainty about the target <InlineMath math="Y" />. Higher IG means the feature is more predictive. Use this intuition when discussing feature selection in your design.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Recommendation System: End-to-End Design Skeleton</h3>
        <CodeBlock
          language="python"
          title="recommendation_system_design.py"
          code={`"""
Recommendation System — End-to-end ML system design skeleton.
Covers data pipeline, feature computation, model training, and serving.
"""
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict

# ─── Stage 1: Data Pipeline ───────────────────────────────────────

@dataclass
class UserEvent:
    user_id: str
    item_id: str
    event_type: str   # "view", "click", "purchase", "rating"
    timestamp: float
    metadata: dict = field(default_factory=dict)

class DataPipeline:
    """Ingests raw events and produces training-ready data."""

    def __init__(self):
        self.events: list[UserEvent] = []
        self.user_profiles: dict[str, dict] = defaultdict(dict)
        self.item_catalog: dict[str, dict] = {}

    def ingest_event(self, event: UserEvent):
        """Ingest a user interaction event."""
        self.events.append(event)
        # Update real-time user profile
        profile = self.user_profiles[event.user_id]
        profile.setdefault("event_count", 0)
        profile["event_count"] += 1
        profile["last_active"] = event.timestamp

    def generate_training_pairs(self, positive_events=("click", "purchase")):
        """Create (user, item, label) pairs for model training.

        Positive: user interacted with item.
        Negative: sample items the user did NOT interact with.
        """
        positive_pairs = []
        user_positive_items = defaultdict(set)

        for event in self.events:
            if event.event_type in positive_events:
                positive_pairs.append((event.user_id, event.item_id, 1.0))
                user_positive_items[event.user_id].add(event.item_id)

        # Negative sampling: 4 negatives per positive
        all_items = list(self.item_catalog.keys())
        negative_pairs = []
        import random
        for user_id, pos_items in user_positive_items.items():
            neg_candidates = [i for i in all_items if i not in pos_items]
            n_neg = min(len(pos_items) * 4, len(neg_candidates))
            for item_id in random.sample(neg_candidates, n_neg):
                negative_pairs.append((user_id, item_id, 0.0))

        return positive_pairs + negative_pairs


# ─── Stage 2: Feature Engineering ─────────────────────────────────

class FeatureComputer:
    """Computes features for users and items."""

    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline

    def compute_user_features(self, user_id: str) -> dict:
        """Offline + online user features."""
        profile = self.pipeline.user_profiles.get(user_id, {})
        events = [e for e in self.pipeline.events if e.user_id == user_id]

        return {
            "total_events": profile.get("event_count", 0),
            "days_since_last_active": (
                (time.time() - profile["last_active"]) / 86400
                if "last_active" in profile else -1
            ),
            "purchase_rate": (
                sum(1 for e in events if e.event_type == "purchase")
                / max(len(events), 1)
            ),
            "unique_items_viewed": len(set(
                e.item_id for e in events if e.event_type == "view"
            )),
        }

    def compute_item_features(self, item_id: str) -> dict:
        """Item-level features from catalog and engagement."""
        catalog = self.pipeline.item_catalog.get(item_id, {})
        events = [e for e in self.pipeline.events if e.item_id == item_id]

        return {
            "category": catalog.get("category", "unknown"),
            "price": catalog.get("price", 0.0),
            "total_views": sum(1 for e in events if e.event_type == "view"),
            "total_purchases": sum(
                1 for e in events if e.event_type == "purchase"
            ),
            "conversion_rate": (
                sum(1 for e in events if e.event_type == "purchase")
                / max(sum(1 for e in events if e.event_type == "view"), 1)
            ),
        }


# ─── Stage 3: Model Training ─────────────────────────────────────

class TwoStageRecommender:
    """Two-stage architecture: candidate generation + ranking."""

    def __init__(self):
        self.candidate_model = None   # retrieval (ANN, matrix factorization)
        self.ranking_model = None     # ranking (gradient-boosted trees, DNN)
        self.item_embeddings = {}

    def train_candidate_generator(self, training_pairs):
        """Train candidate generation model (simplified).
        In production: matrix factorization, two-tower DNN, etc.
        """
        # Simplified: track item popularity as a baseline
        item_scores = defaultdict(float)
        for user_id, item_id, label in training_pairs:
            item_scores[item_id] += label
        self.item_embeddings = dict(item_scores)
        print(f"Candidate model trained on {len(training_pairs)} pairs")

    def train_ranker(self, features_and_labels):
        """Train ranking model on feature vectors.
        In production: XGBoost, LambdaMART, or deep ranking network.
        """
        # Placeholder — would use XGBoost or similar
        self.ranking_model = "trained"
        print(f"Ranking model trained on {len(features_and_labels)} samples")

    def get_candidates(self, user_id: str, n: int = 100) -> list[str]:
        """Retrieve top-N candidate items (fast, approximate)."""
        sorted_items = sorted(
            self.item_embeddings.items(), key=lambda x: x[1], reverse=True
        )
        return [item_id for item_id, _ in sorted_items[:n]]

    def rank(self, user_id: str, candidate_ids: list[str]) -> list[str]:
        """Re-rank candidates using detailed features (slower, precise)."""
        # Placeholder — would compute features and run ranking model
        return candidate_ids  # passthrough for skeleton


# ─── Stage 4: Serving with FastAPI ────────────────────────────────

# In production, this would be a FastAPI app
def create_serving_app(recommender, feature_computer):
    """Create a serving endpoint for recommendations.

    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/recommend/{user_id}")
    async def recommend(user_id: str, n: int = 20):
        # Step 1: Candidate generation (~10ms)
        candidates = recommender.get_candidates(user_id, n=100)

        # Step 2: Feature computation (~5ms from feature store)
        user_features = feature_computer.compute_user_features(user_id)

        # Step 3: Ranking (~15ms)
        ranked = recommender.rank(user_id, candidates)

        # Step 4: Post-processing (diversity, business rules)
        final = apply_business_rules(ranked[:n], user_id)

        return {"user_id": user_id, "recommendations": final}
    """
    print("Serving app created — total latency budget: ~50ms")
    print("  Candidate generation: ~10ms (ANN lookup)")
    print("  Feature retrieval:    ~5ms  (feature store)")
    print("  Ranking:              ~15ms (model inference)")
    print("  Post-processing:      ~5ms  (rules, diversity)")
    print("  Network overhead:     ~15ms")


# ─── Putting It All Together ──────────────────────────────────────

pipeline = DataPipeline()

# Simulate catalog and events
for i in range(50):
    pipeline.item_catalog[f"item_{i}"] = {
        "category": ["electronics", "books", "clothing"][i % 3],
        "price": 10.0 + i * 2,
    }

import random
random.seed(42)
for _ in range(500):
    event = UserEvent(
        user_id=f"user_{random.randint(0, 19)}",
        item_id=f"item_{random.randint(0, 49)}",
        event_type=random.choice(["view", "view", "view", "click", "purchase"]),
        timestamp=time.time() - random.randint(0, 86400 * 30),
    )
    pipeline.ingest_event(event)

# Generate training data
pairs = pipeline.generate_training_pairs()
print(f"Training pairs: {len(pairs)}")

# Train model
recommender = TwoStageRecommender()
recommender.train_candidate_generator(pairs)

# Feature computation
feature_computer = FeatureComputer(pipeline)
user_feats = feature_computer.compute_user_features("user_0")
print(f"User features: {user_feats}")

# Get recommendations
candidates = recommender.get_candidates("user_0", n=10)
print(f"Top candidates for user_0: {candidates}")

# Serving
create_serving_app(recommender, feature_computer)`}
        />

        <h3>Feature Store Pattern</h3>
        <CodeBlock
          language="python"
          title="feature_store_pattern.py"
          code={`"""
Feature Store: ensures training and serving use the same feature computation.
This is a simplified pattern — production systems use Feast, Tecton, or similar.
"""
import time
from typing import Any
from collections import defaultdict

class FeatureStore:
    """Centralized feature management for ML systems."""

    def __init__(self):
        self.offline_store: dict[str, dict[str, Any]] = defaultdict(dict)
        self.online_store: dict[str, dict[str, Any]] = defaultdict(dict)
        self.feature_definitions: dict[str, dict] = {}

    def register_feature(self, name: str, dtype: str, description: str,
                         compute_fn=None):
        """Register a feature with its computation logic."""
        self.feature_definitions[name] = {
            "dtype": dtype,
            "description": description,
            "compute_fn": compute_fn,
            "registered_at": time.time(),
        }

    def write_offline(self, entity_id: str, features: dict[str, Any]):
        """Write batch-computed features (called by daily pipeline)."""
        self.offline_store[entity_id].update(features)

    def write_online(self, entity_id: str, features: dict[str, Any]):
        """Write real-time features (called by streaming pipeline)."""
        self.online_store[entity_id].update(features)
        self.online_store[entity_id]["_updated_at"] = time.time()

    def get_features(self, entity_id: str, feature_names: list[str],
                     online: bool = True) -> dict[str, Any]:
        """Retrieve features — same API for training and serving."""
        store = self.online_store if online else self.offline_store
        entity_features = store.get(entity_id, {})
        return {f: entity_features.get(f) for f in feature_names}

    def get_training_data(self, entity_ids: list[str],
                          feature_names: list[str]) -> list[dict]:
        """Get batch features for training (always use offline store)."""
        return [
            {"entity_id": eid, **self.get_features(eid, feature_names, online=False)}
            for eid in entity_ids
        ]

# Example usage
store = FeatureStore()
store.register_feature("txn_count_24h", "int", "Transaction count in last 24 hours")
store.register_feature("avg_txn_amount_30d", "float", "Avg transaction amount over 30 days")

store.write_offline("user_123", {"avg_txn_amount_30d": 45.67})
store.write_online("user_123", {"txn_count_24h": 5})

serving_features = store.get_features("user_123", ["txn_count_24h"], online=True)
print(f"Serving features: {serving_features}")`}
        />

        <h3>A/B Test Infrastructure</h3>
        <CodeBlock
          language="python"
          title="ab_test_infrastructure.py"
          code={`"""
A/B Test Infrastructure — assignment, logging, and analysis.
"""
import hashlib
import math
from collections import defaultdict

class ABTestManager:
    """Manages experiment assignment and basic significance testing."""

    def __init__(self):
        self.experiments: dict[str, dict] = {}
        self.assignments: dict[str, dict[str, str]] = defaultdict(dict)
        self.events: list[dict] = []

    def create_experiment(self, name: str, variants: list[str],
                          traffic_pct: float = 1.0):
        self.experiments[name] = {
            "variants": variants, "traffic_pct": traffic_pct
        }

    def assign_variant(self, experiment: str, user_id: str) -> str:
        """Deterministic hash-based assignment (consistent across calls)."""
        if experiment in self.assignments[user_id]:
            return self.assignments[user_id][experiment]

        hash_input = f"{experiment}:{user_id}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        exp = self.experiments[experiment]

        if (hash_val % 1000) / 1000 > exp["traffic_pct"]:
            variant = "holdout"
        else:
            variant = exp["variants"][hash_val % len(exp["variants"])]

        self.assignments[user_id][experiment] = variant
        return variant

    def log_event(self, experiment: str, user_id: str,
                  metric: str, value: float):
        self.events.append({
            "experiment": experiment, "user_id": user_id,
            "variant": self.assignments[user_id].get(experiment),
            "metric": metric, "value": value,
        })

    def analyze(self, experiment: str, metric: str) -> dict:
        """Compute per-variant means and two-sample z-test."""
        variant_values = defaultdict(list)
        for e in self.events:
            if e["experiment"] == experiment and e["metric"] == metric:
                variant_values[e["variant"]].append(e["value"])

        results = {}
        for variant, vals in variant_values.items():
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
            results[variant] = {"n": n, "mean": round(mean, 4), "var": round(var, 6)}

        keys = list(results.keys())
        if len(keys) >= 2:
            a, b = results[keys[0]], results[keys[1]]
            se = math.sqrt(a["var"] / a["n"] + b["var"] / b["n"])
            if se > 0:
                z = (a["mean"] - b["mean"]) / se
                results["z_stat"] = round(z, 3)
                results["significant_at_95"] = abs(z) > 1.96

        return results

# Example
import random
random.seed(42)
ab = ABTestManager()
ab.create_experiment("new_ranker", ["control", "treatment"], traffic_pct=0.5)

for i in range(2000):
    uid = f"user_{i}"
    v = ab.assign_variant("new_ranker", uid)
    ctr = random.gauss(0.12 if v == "treatment" else 0.10, 0.05)
    ab.log_event("new_ranker", uid, "ctr", max(0, ctr))

print(ab.analyze("new_ranker", "ctr"))`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always clarify requirements first</strong> &mdash; spend the first 3-5 minutes asking questions. A latency constraint of 50ms vs 5s changes the entire design. 1K users vs 1B users means completely different infrastructure.</li>
          <li><strong>Think about data before models</strong> &mdash; the best model fails without good training data. Ask: where does the data come from? How do we label it? What&apos;s the labeling cost and latency?</li>
          <li><strong>Online vs. offline features</strong> &mdash; offline features (computed in batch) are cheap and reliable. Online features (computed at request time) add latency and complexity. Only go online when freshness matters (e.g., &quot;transactions in last 5 minutes&quot;).</li>
          <li><strong>Batch vs. real-time serving</strong> &mdash; precompute predictions when possible (cheaper, simpler). Real-time inference is needed only when the input changes per request (search queries, fraud scoring, live context).</li>
          <li><strong>Two-stage architecture is standard</strong> &mdash; for large item catalogs, use a fast candidate retrieval stage (ANN, embedding similarity) followed by a precise ranking stage (GBT, deep network). This is the Netflix/YouTube/Instagram pattern.</li>
          <li><strong>Start simple, iterate</strong> &mdash; a logistic regression deployed this week beats a transformer deployed next quarter. Show the interviewer you understand iterative improvement: v1 ships fast, v2 adds complexity where it matters.</li>
          <li><strong>Monitoring is non-negotiable</strong> &mdash; every ML system degrades over time. Monitor input distributions, prediction distributions, and business metrics. Set up alerts for drift and plan retraining triggers.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Jumping straight to model architecture</strong> &mdash; the most common mistake. Interviewers want systems thinking, not &quot;I&apos;d use a transformer.&quot; Cover requirements, data, and serving before discussing the model.</li>
          <li><strong>Ignoring data quality and labeling</strong> &mdash; &quot;We&apos;ll use click logs as labels&quot; &mdash; but clicks are noisy, biased by position, and don&apos;t capture satisfaction. Discuss label quality, potential biases, and how to mitigate them.</li>
          <li><strong>No monitoring or retraining plan</strong> &mdash; a system without monitoring is a ticking time bomb. Data distributions shift, user behavior changes, upstream pipelines break. Always include monitoring in your design.</li>
          <li><strong>Over-engineering for day 1</strong> &mdash; you don&apos;t need a feature store, real-time pipeline, and multi-model ensemble from the start. Design a v1 that&apos;s simple and deployable, then describe the evolution path.</li>
          <li><strong>Forgetting the cold start problem</strong> &mdash; new users and items have no history. How does the system handle them? Fallback to popularity, content-based features, or explicit onboarding flows.</li>
          <li><strong>Training/serving skew</strong> &mdash; if training uses batch-computed features but serving computes features differently, the model sees different distributions in production. A feature store with shared computation logic prevents this.</li>
          <li><strong>Ignoring feedback loops</strong> &mdash; if your model recommends items, users click on them, and those clicks become training data, you have a feedback loop. Popular items get more exposure, more clicks, and more reinforcement &mdash; regardless of actual quality.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> &quot;Design a recommendation system for an e-commerce platform with 100M users and 10M products.&quot;</p>
        <p><strong>Strong answer structure:</strong></p>
        <ol>
          <li>
            <strong>Requirements:</strong> Latency under 200ms. Optimize for purchase conversion (not just clicks). Personalized for logged-in users, popularity-based for anonymous. Handle cold-start for new products added daily.
          </li>
          <li>
            <strong>ML Formulation:</strong> Pointwise ranking &mdash; predict P(purchase | user, item, context). Labels from purchase events (strong positive), add-to-cart (weak positive), negative sampling from impressions without clicks.
          </li>
          <li>
            <strong>Data Pipeline:</strong> Event streaming (Kafka) captures views, clicks, purchases. Daily batch job computes user and item aggregate features. Real-time features (session behavior, cart contents) computed on the fly. Feature store ensures training/serving consistency.
          </li>
          <li>
            <strong>Model:</strong> Two-stage: (a) candidate generation using two-tower DNN with user and item embeddings, retrieving top 200 via approximate nearest neighbors; (b) ranking using XGBoost over detailed features (user history, item attributes, cross-features, real-time context). Start with XGBoost alone, add the neural retrieval stage when catalog grows.
          </li>
          <li>
            <strong>Serving:</strong> Candidate embeddings precomputed nightly. ANN index updated hourly. Ranking model served via model server with 50ms p99 SLA. Redis cache for user features. Fallback to popularity ranking if model is unavailable.
          </li>
          <li>
            <strong>Evaluation:</strong> Offline: NDCG@20, Hit Rate@10, precision/recall. Online A/B test: conversion rate, revenue per session, engagement time. Monitor for popularity bias, category coverage, fairness across user segments. Retrain weekly with sliding window of last 90 days.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Chip Huyen &quot;Designing Machine Learning Systems&quot; (O&apos;Reilly)</strong> &mdash; The definitive book on ML systems. Covers the full lifecycle from data to deployment to monitoring with real-world case studies.</li>
          <li><strong>Stanford CS 329S: Machine Learning Systems Design</strong> &mdash; Chip Huyen&apos;s course with publicly available lecture notes covering production ML patterns.</li>
          <li><strong>Google &quot;Rules of Machine Learning&quot; by Martin Zinkevich</strong> &mdash; 43 practical rules for ML engineering. Rule #1: Don&apos;t be afraid to launch a product without machine learning.</li>
          <li><strong>Eugene Yan &quot;System Design for Recommendations and Search&quot;</strong> &mdash; Excellent deep dive into two-stage retrieval-ranking architectures used at major tech companies.</li>
          <li><strong>Made With ML by Goku Mohandas</strong> &mdash; Comprehensive, code-driven guide to MLOps and system design with end-to-end examples.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
