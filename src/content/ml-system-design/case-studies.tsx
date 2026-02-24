"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function CaseStudies() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          The best way to prepare for ML system design interviews is to study real systems built by companies operating at massive scale. These four case studies cover the most commonly asked categories and reveal patterns that repeat across every domain.
        </p>

        <h3>Case Study 1: Recommendation System (Netflix)</h3>
        <p>
          Netflix personalizes content for 230M+ subscribers across 190 countries. Their recommendation engine drives over 80% of content watched on the platform. The core challenge: rank ~15,000 titles for each user in real time, balancing relevance, diversity, and freshness while handling cold-start for new users and newly added content.
        </p>
        <ul>
          <li><strong>Requirements:</strong> Sub-200ms latency, personalized for every user, must handle cold-start, optimize for watch time (not clicks).</li>
          <li><strong>Data:</strong> Implicit feedback (watch completion, replays, abandonment), viewing context (device, time of day), content metadata (genre, cast, duration).</li>
          <li><strong>Features:</strong> User embedding from watch history, item embedding from metadata + engagement stats, cross features (user genre affinity, recency decay), contextual signals (device type, hour).</li>
          <li><strong>Model:</strong> Two-stage &mdash; candidate generation (two-tower DNN / matrix factorization) retrieves ~500 titles, then a ranking model (gradient-boosted trees or deep network) scores them using rich features.</li>
          <li><strong>Serving:</strong> Precompute candidate embeddings nightly, update ANN index hourly, ranking model served with &lt;50ms p99 latency. Fallback to popularity-based ranking if model is unavailable.</li>
          <li><strong>Evaluation:</strong> Offline: NDCG@K, recall@K, catalog coverage. Online: A/B tests on viewing hours, retention, content diversity. Monitor for popularity bias and filter bubbles.</li>
        </ul>

        <h3>Case Study 2: Fraud Detection (Stripe)</h3>
        <p>
          Stripe&apos;s Radar processes billions of dollars in payments, scoring every transaction in under 100ms. The system must maintain high recall (catch fraud) without excessive false positives (blocking legitimate customers). Fraud is adversarial &mdash; attackers constantly adapt their tactics.
        </p>
        <ul>
          <li><strong>Requirements:</strong> &lt;100ms latency (blocking the transaction), high recall with manageable false positive rate, explainable decisions (regulatory), continuous adaptation to new fraud patterns.</li>
          <li><strong>Data:</strong> Transaction features, user behavioral history, device fingerprints, merchant data. Labels from chargebacks (delayed 30-90 days) and manual investigations. Extreme class imbalance (~0.1% fraud).</li>
          <li><strong>Features:</strong> Velocity features (transaction count/amount in last hour/day), behavioral anomaly features (amount z-score vs user history), network features (shared devices, merchant fraud rate), contextual features (new device, new country).</li>
          <li><strong>Model:</strong> Hybrid rules + ML. Rules engine catches obvious patterns (&gt;10 transactions/hour, known bad IPs). XGBoost for nuanced scoring (fast inference, interpretable with SHAP). Address imbalance with weighted loss or SMOTE.</li>
          <li><strong>Serving:</strong> Real-time scoring per transaction. User features cached in Redis. Rules engine as fast pre-filter (&lt;1ms). ML model scoring (&lt;10ms). Fallback to rules-only if model is unavailable. Canary deployment with shadow scoring.</li>
          <li><strong>Evaluation:</strong> Precision-recall at operating threshold. Online: fraud loss rate, false decline rate. Monitor for adversarial drift (new attack patterns). Retrain weekly with confirmed fraud labels.</li>
        </ul>

        <h3>Case Study 3: Search Ranking (Google)</h3>
        <p>
          Google Search handles 8.5 billion queries per day, returning relevant results within 200ms from an index of hundreds of billions of web pages. The system uses a multi-stage funnel: retrieval, initial ranking, and neural re-ranking.
        </p>
        <ul>
          <li><strong>Requirements:</strong> &lt;200ms total latency, handle navigational (&quot;facebook login&quot;), informational (&quot;how does photosynthesis work&quot;), and transactional (&quot;buy running shoes&quot;) queries differently.</li>
          <li><strong>Data:</strong> Web crawl (content, links), click logs (position-debiased), query logs, human relevance judgments (for evaluation). Query understanding: intent classification, entity recognition, spell correction.</li>
          <li><strong>Features:</strong> BM25 text match, PageRank authority, content freshness, query-document semantic similarity, click-through rate (debiased), site quality signals, mobile-friendliness.</li>
          <li><strong>Model:</strong> Three-stage funnel: (1) BM25 inverted index retrieves top 1000 documents, (2) lightweight LambdaMART ranks down to top 100, (3) cross-encoder (BERT) re-ranks top 10-20 for final display.</li>
          <li><strong>Serving:</strong> Distributed inverted index across thousands of shards. Scatter-gather retrieval. Neural re-ranking only on final candidates (too expensive for all). Query cache for popular queries. Fallback to BM25-only ranking.</li>
          <li><strong>Evaluation:</strong> Offline: NDCG, MRR from human judgments. Online: click satisfaction (long clicks vs short clicks), query reformulation rate, pogo-sticking rate. Side-by-side human evaluations for major ranking changes.</li>
        </ul>

        <h3>Case Study 4: Feed Ranking (Instagram)</h3>
        <p>
          Instagram ranks posts from followed accounts plus suggested content for 2B+ monthly active users. With millions of posts created per minute, the system must score thousands of candidates per feed refresh in under 500ms using multi-objective optimization.
        </p>
        <ul>
          <li><strong>Requirements:</strong> &lt;500ms latency for feed generation, multi-objective (likes, comments, shares, saves, dwell time), diversity constraints (no single author dominating), freshness bias for new posts.</li>
          <li><strong>Data:</strong> Social graph (follows, close friends), interaction history (likes, comments, saves, shares, dwell time), content signals (image embeddings, caption NLP, hashtags), session context (time, device, recent activity).</li>
          <li><strong>Features:</strong> Author-viewer affinity (interaction frequency), content type preference, category affinity, recency decay, post engagement rate, social proof (friends who engaged), session features (items already seen).</li>
          <li><strong>Model:</strong> Multi-task deep network predicting P(like), P(comment), P(share), P(save), E[dwell time] simultaneously. Final score is weighted combination tuned by A/B tests to maximize long-term retention (not just immediate engagement).</li>
          <li><strong>Serving:</strong> Three-stage funnel: candidate sourcing (following + explore + ads), lightweight pre-ranking (&lt;5ms), full ranking model (&lt;50ms). Post-ranking diversity injection to ensure category and author variety. Real-time feature computation from current session.</li>
          <li><strong>Evaluation:</strong> Online: session time, return frequency, content diversity index. Offline: per-objective AUC. Monitor for engagement bait amplification, filter bubbles, and content creator fairness. Regular calibration of objective weights via experiments.</li>
        </ul>
      </TopicSection>

      <TopicSection type="math">
        <h3>NDCG for Ranking Quality (Netflix, Google)</h3>
        <BlockMath math="NDCG@k = \frac{DCG@k}{IDCG@k}, \quad DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}" />
        <p>NDCG penalizes relevant items placed at lower ranks. Values range from 0 to 1, where 1 means perfect ranking. Both Netflix and Google use NDCG as a primary offline metric.</p>

        <h3>Precision-Recall Tradeoff (Stripe)</h3>
        <BlockMath math="\text{Precision}(\tau) = \frac{TP(\tau)}{TP(\tau) + FP(\tau)}, \quad \text{Recall}(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}" />
        <p>At a score threshold <InlineMath math="\tau" />, lowering it catches more fraud (higher recall) but blocks more legitimate transactions (lower precision). Stripe chooses the operating point that maximizes recall subject to a false positive rate constraint.</p>

        <h3>Multi-Objective Scoring (Instagram)</h3>
        <BlockMath math="\text{Score}(post) = \sum_{i=1}^{k} w_i \cdot P_i(engagement_i | user, post, context)" />
        <p>where <InlineMath math="w_i" /> are weights tuned via online experiments. Instagram&apos;s weights are calibrated to maximize <em>long-term retention</em>, not just immediate engagement, to avoid optimizing for clickbait.</p>

        <h3>BM25 Retrieval (Google)</h3>
        <BlockMath math="BM25(q, d) = \sum_{t \in q} IDF(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{avgdl})}" />
        <p>BM25 is the workhorse of first-stage retrieval. It combines term frequency, inverse document frequency, and document length normalization. Parameters <InlineMath math="k_1 \approx 1.5" /> and <InlineMath math="b \approx 0.75" /> are standard defaults.</p>

        <h3>Position Debiasing (Google, Instagram)</h3>
        <BlockMath math="\hat{r}(d, q) = \frac{c(d, q)}{P(\text{examine} \mid \text{position}(d))}" />
        <p>Users are more likely to click higher-positioned items regardless of quality. Inverse propensity weighting corrects this bias when using click data as training labels.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Case Study 1: Netflix-Style Two-Stage Recommender</h3>
        <CodeBlock
          language="python"
          title="netflix_recommender.py"
          code={`"""
Netflix-style recommendation: candidate generation + ranking.
Key patterns: implicit feedback, embedding similarity, contextual re-ranking.
"""
import random
import math
from collections import defaultdict

class CandidateGenerator:
    """Stage 1: Fast retrieval via user-item embedding similarity."""

    def __init__(self, dim: int = 32):
        self.dim = dim
        self.user_embeddings: dict[str, list[float]] = {}
        self.item_embeddings: dict[str, list[float]] = {}

    def train_from_interactions(self, watch_history: dict):
        """Build user embeddings by averaging watched item embeddings.
        Production: two-tower DNN trained on (user, positive_item) pairs.
        """
        for user_id, events in watch_history.items():
            emb = [0.0] * self.dim
            weight_sum = 0.0
            for event in events[-50:]:  # recent 50 interactions
                item_emb = self.item_embeddings.get(event["item_id"])
                if item_emb:
                    w = event["watch_pct"]  # weight by engagement
                    for i in range(self.dim):
                        emb[i] += item_emb[i] * w
                    weight_sum += w
            if weight_sum > 0:
                emb = [e / weight_sum for e in emb]
            self.user_embeddings[user_id] = emb

    def retrieve(self, user_id: str, n: int = 200,
                 exclude: set = None) -> list[tuple[str, float]]:
        """Retrieve top-N items by cosine similarity."""
        user_emb = self.user_embeddings.get(user_id, [0.0] * self.dim)
        exclude = exclude or set()
        scores = []
        for item_id, item_emb in self.item_embeddings.items():
            if item_id in exclude:
                continue
            dot = sum(u * i for u, i in zip(user_emb, item_emb))
            norm_u = math.sqrt(sum(u ** 2 for u in user_emb)) + 1e-8
            norm_i = math.sqrt(sum(i ** 2 for i in item_emb)) + 1e-8
            scores.append((item_id, dot / (norm_u * norm_i)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


class Ranker:
    """Stage 2: Re-rank candidates with rich features + context."""

    def __init__(self):
        self.item_metadata: dict[str, dict] = {}

    def rank(self, user_id: str, candidates: list[tuple[str, float]],
             context: dict, watched_genres: dict) -> list[tuple[str, float]]:
        scored = []
        for item_id, retrieval_score in candidates:
            meta = self.item_metadata.get(item_id, {})
            score = retrieval_score * 0.4  # embedding similarity

            # Content quality
            score += meta.get("avg_completion_rate", 0.5) * 0.2
            # Genre affinity
            genre = meta.get("genre", "unknown")
            genre_weight = watched_genres.get(genre, 0)
            score += min(genre_weight / 10, 0.3) * 0.2
            # Freshness boost (evening = new releases)
            if context.get("hour", 12) >= 18:
                score += meta.get("is_new_release", 0) * 0.15
            # Diversity penalty (same genre as recent watch)
            if genre == context.get("last_watched_genre"):
                score -= 0.05

            scored.append((item_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# --- Demo ---
random.seed(42)
gen = CandidateGenerator(dim=16)
ranker = Ranker()
genres = ["action", "comedy", "drama", "sci-fi", "documentary"]

# Create item catalog
for i in range(200):
    item_id = f"title_{i}"
    gen.item_embeddings[item_id] = [random.gauss(0, 1) for _ in range(16)]
    ranker.item_metadata[item_id] = {
        "genre": genres[i % len(genres)],
        "avg_completion_rate": random.uniform(0.3, 0.95),
        "is_new_release": 1.0 if i > 180 else 0.0,
    }

# Simulate watch history
watch_history = {"user_A": []}
watched_set = set()
genre_counts = defaultdict(int)
for _ in range(40):
    item = f"title_{random.randint(0, 199)}"
    pct = random.uniform(0.2, 1.0)
    watch_history["user_A"].append({"item_id": item, "watch_pct": pct})
    watched_set.add(item)
    genre_counts[ranker.item_metadata[item]["genre"]] += 1

gen.train_from_interactions(watch_history)

# Stage 1: Retrieve candidates
candidates = gen.retrieve("user_A", n=50, exclude=watched_set)
print(f"Stage 1: Retrieved {len(candidates)} candidates")

# Stage 2: Rank with context
context = {"hour": 21, "last_watched_genre": "action", "device": "tv"}
ranked = ranker.rank("user_A", candidates, context, dict(genre_counts))

print("\\nTop 10 recommendations:")
for item_id, score in ranked[:10]:
    meta = ranker.item_metadata[item_id]
    print(f"  {item_id} [{meta['genre']}] score={score:.3f}"
          f" completion={meta['avg_completion_rate']:.0%}")`}
        />

        <h3>Case Study 2: Stripe-Style Fraud Detection</h3>
        <CodeBlock
          language="python"
          title="stripe_fraud_detection.py"
          code={`"""
Stripe-style fraud detection: rules + ML hybrid with real-time features.
Key patterns: velocity features, rules cascade, explainable scoring.
"""
import math
import time
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class Transaction:
    txn_id: str
    user_id: str
    amount: float
    merchant_id: str
    device_id: str
    country: str
    timestamp: float

class FraudDetector:
    """Hybrid rules + ML fraud scoring pipeline."""

    def __init__(self):
        self.user_history: dict[str, list[Transaction]] = defaultdict(list)
        self.device_users: dict[str, set] = defaultdict(set)

    def compute_features(self, txn: Transaction) -> dict:
        """Real-time + historical feature computation."""
        history = self.user_history.get(txn.user_id, [])
        recent_1h = [t for t in history if txn.timestamp - t.timestamp < 3600]
        recent_24h = [t for t in history if txn.timestamp - t.timestamp < 86400]

        # Amount anomaly detection
        amounts = [t.amount for t in history]
        avg_amt = sum(amounts) / len(amounts) if amounts else 0
        std_amt = (sum((a - avg_amt) ** 2 for a in amounts) /
                   max(len(amounts), 1)) ** 0.5 if amounts else 1.0

        return {
            "amount": txn.amount,
            "amount_zscore": (txn.amount - avg_amt) / max(std_amt, 1.0),
            "txn_count_1h": len(recent_1h),
            "txn_count_24h": len(recent_24h),
            "amount_sum_24h": sum(t.amount for t in recent_24h),
            "is_new_device": txn.user_id not in self.device_users.get(
                txn.device_id, set()),
            "is_new_country": txn.country not in set(
                t.country for t in history),
            "device_user_count": len(self.device_users.get(
                txn.device_id, set())),
            "unique_merchants_24h": len(set(
                t.merchant_id for t in recent_24h)),
        }

    def apply_rules(self, features: dict) -> tuple:
        """Stage 1: Fast deterministic rules (<1ms)."""
        if features["txn_count_1h"] > 10:
            return "BLOCK", "high_velocity"
        if features["amount"] > 10000 and features["is_new_device"]:
            return "BLOCK", "high_amount_new_device"
        if features["device_user_count"] > 5:
            return "BLOCK", "shared_device"
        return "PASS", None

    def ml_score(self, features: dict) -> tuple:
        """Stage 2: ML model scoring (<10ms).
        Production: XGBoost with SHAP explanations.
        """
        # Weighted feature scoring (placeholder for trained model)
        weights = {
            "amount_zscore": 0.20,
            "txn_count_1h": 0.05,
            "is_new_device": 0.25,
            "is_new_country": 0.20,
            "device_user_count": 0.04,
            "unique_merchants_24h": 0.03,
        }
        raw = sum(weights.get(k, 0) * float(v) for k, v in features.items())
        prob = 1.0 / (1.0 + math.exp(-(raw - 1.0)))  # sigmoid

        # Top contributing feature (like SHAP top feature)
        contributions = {k: weights.get(k, 0) * float(features[k])
                         for k in weights}
        top_reason = max(contributions, key=contributions.get)
        return round(prob, 4), top_reason

    def score(self, txn: Transaction) -> dict:
        """Full pipeline: features -> rules -> ML -> decision."""
        features = self.compute_features(txn)

        rule_decision, rule_reason = self.apply_rules(features)
        if rule_decision == "BLOCK":
            return {"decision": "BLOCK", "stage": "rules",
                    "reason": rule_reason, "score": 1.0}

        prob, reason = self.ml_score(features)
        decision = "BLOCK" if prob > 0.8 else "REVIEW" if prob > 0.5 else "ALLOW"
        return {"decision": decision, "stage": "ml",
                "score": prob, "reason": reason}

    def record(self, txn: Transaction):
        self.user_history[txn.user_id].append(txn)
        self.device_users[txn.device_id].add(txn.user_id)

# --- Demo ---
detector = FraudDetector()
now = time.time()

# Build normal history
for i in range(30):
    t = Transaction(f"txn_{i}", "alice", 40 + i * 3, "shop_A",
                    "dev_1", "US", now - 86400 * (30 - i))
    detector.record(t)

# Normal transaction
normal = Transaction("txn_ok", "alice", 55.0, "shop_A", "dev_1", "US", now)
print("Normal:", detector.score(normal))

# Suspicious: new device, new country, high amount
fraud = Transaction("txn_bad", "alice", 2800.0, "shop_X",
                    "dev_NEW", "NG", now)
print("Suspicious:", detector.score(fraud))`}
        />

        <h3>Case Study 3: Google-Style Search Ranking</h3>
        <CodeBlock
          language="python"
          title="google_search_ranking.py"
          code={`"""
Google-style search ranking: BM25 retrieval + neural re-ranking.
Key patterns: multi-stage funnel, query understanding, position debiasing.
"""
import math
from collections import Counter, defaultdict

class BM25Index:
    """Stage 1: Inverted index with BM25 scoring."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.df: dict[str, int] = defaultdict(int)
        self.avg_dl = 0

    def add_documents(self, documents: list[str]):
        self.docs = documents
        for doc_id, doc in enumerate(documents):
            tokens = doc.lower().split()
            self.doc_tokens.append(tokens)
            counts = Counter(tokens)
            for term, freq in counts.items():
                self.df[term] += 1
                self.index[term].append((doc_id, freq))
        self.avg_dl = sum(len(t) for t in self.doc_tokens) / len(documents)

    def search(self, query: str, top_k: int = 100) -> list[tuple[int, float]]:
        terms = query.lower().split()
        scores = defaultdict(float)
        n = len(self.docs)
        for term in terms:
            idf = math.log((n - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
            for doc_id, tf in self.index.get(term, []):
                dl = len(self.doc_tokens[doc_id])
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
                scores[doc_id] += idf * tf_norm
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class NeuralReranker:
    """Stage 3: Cross-encoder re-ranking (BERT-style)."""

    def rerank(self, query: str, doc_ids: list[int],
               doc_tokens: list[list[str]], doc_metadata: list[dict],
               top_k: int = 10) -> list[tuple[int, float]]:
        q_terms = set(query.lower().split())
        scored = []
        for doc_id in doc_ids:
            tokens = doc_tokens[doc_id]
            meta = doc_metadata[doc_id]
            # Semantic features (placeholder for cross-encoder)
            overlap = len(q_terms & set(tokens)) / max(len(q_terms), 1)
            title_match = len(q_terms & set(
                meta.get("title", "").lower().split())) / max(len(q_terms), 1)
            score = (overlap * 0.3 + title_match * 0.3 +
                     meta.get("authority", 0.5) * 0.2 +
                     meta.get("freshness", 0.5) * 0.2)
            scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def evaluate_ndcg(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    dcg = sum((1.0 if ranked_ids[i] in relevant else 0.0) / math.log2(i + 2)
              for i in range(min(k, len(ranked_ids))))
    ideal = sorted([1] * min(len(relevant), k) + [0] * max(k - len(relevant), 0),
                    reverse=True)
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(len(ideal)))
    return dcg / idcg if idcg > 0 else 0.0

# --- Demo ---
documents = [
    "machine learning algorithms for classification tasks",
    "deep learning neural networks image recognition computer vision",
    "natural language processing transformers attention mechanism",
    "gradient boosted trees xgboost for tabular data prediction",
    "convolutional neural networks for image classification",
    "transformer architecture for machine translation tasks",
    "random forests ensemble methods for robust prediction",
    "reinforcement learning for game playing and robotics control",
    "support vector machines for binary classification problems",
    "neural network optimization training deep models effectively",
]
metadata = [
    {"title": "ML Algorithms Guide", "authority": 0.8, "freshness": 0.6},
    {"title": "Deep Learning for Vision", "authority": 0.9, "freshness": 0.7},
    {"title": "NLP with Transformers", "authority": 0.85, "freshness": 0.9},
    {"title": "XGBoost Tutorial", "authority": 0.7, "freshness": 0.5},
    {"title": "CNN for Image Classification", "authority": 0.9, "freshness": 0.6},
    {"title": "Transformer Architecture", "authority": 0.95, "freshness": 0.8},
    {"title": "Random Forests Guide", "authority": 0.6, "freshness": 0.4},
    {"title": "Reinforcement Learning", "authority": 0.75, "freshness": 0.7},
    {"title": "SVM Classification", "authority": 0.5, "freshness": 0.3},
    {"title": "Neural Network Training", "authority": 0.8, "freshness": 0.8},
]

idx = BM25Index()
idx.add_documents(documents)

query = "neural networks image classification"
print(f"Query: '{query}'\\n")

# Stage 1: BM25
bm25_results = idx.search(query, top_k=7)
print("Stage 1 - BM25 Retrieval:")
for doc_id, score in bm25_results:
    print(f"  [{score:.2f}] {metadata[doc_id]['title']}")

# Stage 3: Neural re-rank
reranker = NeuralReranker()
candidate_ids = [d for d, _ in bm25_results]
reranked = reranker.rerank(query, candidate_ids, idx.doc_tokens, metadata, top_k=5)
print("\\nStage 3 - Neural Re-ranking:")
for doc_id, score in reranked:
    print(f"  [{score:.2f}] {metadata[doc_id]['title']}")

# Evaluate
relevant = {1, 4, 9}
ndcg = evaluate_ndcg([d for d, _ in reranked], relevant, k=5)
print(f"\\nNDCG@5: {ndcg:.3f}")`}
        />

        <h3>Case Study 4: Instagram-Style Feed Ranking</h3>
        <CodeBlock
          language="python"
          title="instagram_feed_ranking.py"
          code={`"""
Instagram-style feed ranking: multi-objective optimization with diversity.
Key patterns: multi-task scoring, candidate sourcing, diversity injection.
"""
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class Post:
    post_id: str
    author_id: str
    content_type: str   # "photo", "video", "reel", "carousel"
    category: str       # "food", "travel", "fitness", "tech"
    timestamp: float
    engagement: dict = field(default_factory=dict)

class FeedRanker:
    """Multi-objective feed ranking with diversity constraints."""

    def __init__(self):
        self.following: dict[str, set] = defaultdict(set)
        self.interests: dict[str, dict] = defaultdict(lambda: defaultdict(float))
        self.posts: dict[str, Post] = {}
        # Weights tuned by A/B tests (optimize long-term retention)
        self.weights = {
            "p_like": 1.0, "p_comment": 1.5,
            "p_share": 2.0, "p_save": 1.8, "e_dwell": 0.5,
        }

    def predict_engagement(self, user_id: str, post: Post) -> dict:
        """Multi-task engagement prediction.
        Production: multi-tower DNN with shared bottom layers.
        """
        interests = self.interests.get(user_id, {})
        is_following = post.author_id in self.following.get(user_id, set())

        # Content type baseline
        type_base = {"reel": 0.14, "video": 0.11, "carousel": 0.09, "photo": 0.07}
        base = type_base.get(post.content_type, 0.07)

        # User-content affinity
        total = max(sum(interests.values()), 1)
        affinity = interests.get(post.category, 0) / total

        # Recency decay
        age_h = max((time.time() - post.timestamp) / 3600, 0.1)
        recency = 1.0 / (1.0 + age_h / 24)

        # Social signal
        social = 0.1 if is_following else 0.0

        return {
            "p_like": min(base + affinity * 0.2 + social, 0.5) * recency,
            "p_comment": min(base * 0.3 + affinity * 0.15, 0.2) * recency,
            "p_share": min(base * 0.1 + affinity * 0.08, 0.1) * recency,
            "p_save": min(base * 0.15 + affinity * 0.12, 0.15) * recency,
            "e_dwell": (3.0 if post.content_type in ("reel", "video")
                       else 1.5) * (1 + affinity),
        }

    def score(self, predictions: dict) -> float:
        """Weighted multi-objective score."""
        return sum(self.weights[k] * predictions[k] for k in self.weights)

    def source_candidates(self, user_id: str, n: int = 300) -> list[str]:
        """Gather posts from following + explore."""
        following = self.following.get(user_id, set())
        from_following = [pid for pid, p in self.posts.items()
                          if p.author_id in following]
        from_explore = [pid for pid, p in self.posts.items()
                        if p.author_id not in following]
        random.shuffle(from_explore)
        # 70% following, 30% explore
        limit_explore = int(n * 0.3)
        return (from_following + from_explore[:limit_explore])[:n]

    def inject_diversity(self, ranked: list[tuple[str, float]],
                         n: int = 20) -> list[str]:
        """Ensure author and category diversity."""
        selected, author_count = [], defaultdict(int)
        cat_count = defaultdict(int)
        for post_id, _ in ranked:
            post = self.posts[post_id]
            if author_count[post.author_id] >= 2:
                continue
            if cat_count[post.category] >= 5:
                continue
            selected.append(post_id)
            author_count[post.author_id] += 1
            cat_count[post.category] += 1
            if len(selected) >= n:
                break
        return selected

    def rank_feed(self, user_id: str, n: int = 20) -> list[dict]:
        """Full feed ranking pipeline."""
        candidates = self.source_candidates(user_id)
        scored = []
        for pid in candidates:
            post = self.posts.get(pid)
            if not post:
                continue
            preds = self.predict_engagement(user_id, post)
            scored.append((pid, self.score(preds)))
        scored.sort(key=lambda x: x[1], reverse=True)

        diverse = self.inject_diversity(scored, n)
        return [{"post_id": p, "author": self.posts[p].author_id,
                 "type": self.posts[p].content_type,
                 "category": self.posts[p].category} for p in diverse]

# --- Demo ---
random.seed(42)
ranker = FeedRanker()
authors = [f"creator_{i}" for i in range(15)]
categories = ["food", "travel", "fitness", "tech", "fashion"]
types = ["photo", "video", "reel", "carousel"]

# Social graph
for a in random.sample(authors, 8):
    ranker.following["viewer_1"].add(a)

# Create posts
for i in range(80):
    post = Post(f"post_{i}", random.choice(authors), random.choice(types),
                random.choice(categories),
                time.time() - random.randint(0, 86400 * 2),
                {"likes": random.randint(5, 500)})
    ranker.posts[post.post_id] = post

# Build interest profile
for _ in range(40):
    p = random.choice(list(ranker.posts.values()))
    ranker.interests["viewer_1"][p.category] += 1.0

feed = ranker.rank_feed("viewer_1", n=10)
print("Feed for viewer_1:")
for item in feed:
    print(f"  {item['post_id']} by {item['author']}"
          f" ({item['type']}, {item['category']})")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Two-stage architecture is universal</strong> &mdash; Netflix (candidate gen + ranking), Google (retrieval + ranking + reranking), Instagram (sourcing + pre-rank + full rank), Stripe (rules + ML). The first stage is fast and broad; the second is slow and precise. This pattern appears in virtually every ML system operating at scale.</li>
          <li><strong>Implicit feedback dominates explicit ratings</strong> &mdash; Netflix uses watch completion rate, Google uses clicks and dwell time, Instagram uses likes/saves/shares. Explicit ratings are rare and biased. Design your label strategy around implicit signals but account for noise and position bias.</li>
          <li><strong>Feature stores prevent training/serving skew</strong> &mdash; all four systems compute features centrally and share them between training and serving. Without this, subtle numerical differences between batch and online feature computation cause silent model degradation.</li>
          <li><strong>Multi-objective optimization is the norm for engagement</strong> &mdash; Instagram and Netflix don&apos;t optimize a single metric. They predict multiple engagement signals and combine them with learned weights. The weights themselves are tuned via A/B tests, creating a two-level optimization.</li>
          <li><strong>Adversarial systems need different patterns</strong> &mdash; fraud detection (Stripe) faces active adversaries who adapt to the model. This requires more frequent retraining, diverse feature types, and rule-based guardrails. Recommendation and search systems don&apos;t face this challenge.</li>
          <li><strong>Cold start demands explicit fallback strategies</strong> &mdash; Netflix uses content-based features for new titles, Google uses document quality signals for new pages, Instagram boosts content from new creators. Never assume every entity has behavioral history.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Using a single-stage architecture at scale</strong> &mdash; scoring 100M items with a heavy model is infeasible. If you describe a system that runs a transformer on every candidate, the interviewer will push back. Always use a funnel: cheap retrieval first, expensive ranking on a small candidate set.</li>
          <li><strong>Ignoring position bias in click data</strong> &mdash; for search and feed ranking, users click higher-positioned items more regardless of quality. Training on raw click data reinforces the existing ranking. Mention inverse propensity weighting or position features to debias training labels.</li>
          <li><strong>Forgetting feedback loops</strong> &mdash; recommendations determine what users see, which generates the clicks that train the next model. This self-reinforcing loop narrows diversity over time. Mention exploration strategies (epsilon-greedy, Thompson sampling, explore/exploit allocation) to break the loop.</li>
          <li><strong>Assuming labels are immediate and clean</strong> &mdash; fraud labels (chargebacks) arrive 30-90 days after the transaction. Recommendation &quot;labels&quot; (did the user enjoy the content?) may never be fully observed. Discuss label latency and how it constrains retraining frequency.</li>
          <li><strong>Not discussing diversity and fairness</strong> &mdash; a feed that maximizes engagement will converge to clickbait. A fraud system with geographic bias will disproportionately block legitimate transactions from certain countries. Always include diversity injection and fairness monitoring in your design.</li>
          <li><strong>Skipping the rules/heuristic baseline</strong> &mdash; every production system has a rules layer. Stripe uses rules for obvious fraud patterns, Google uses rules for spam. Rules give you a deployable v0 and handle edge cases that ML models miss.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> &quot;Compare the system design for a recommendation system vs a fraud detection system. What are the three most important differences?&quot;</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Latency and error cost:</strong> Recommendations tolerate 200-500ms latency and a bad suggestion is merely annoying. Fraud detection must respond in under 100ms (blocking the payment) and a missed fraud case costs real money while a false positive blocks a legitimate customer. This asymmetry makes fraud systems favor high recall with a carefully tuned precision threshold.
          </li>
          <li>
            <strong>Label availability:</strong> Recommendation systems get abundant implicit feedback (clicks, watches) within seconds. Fraud labels arrive 30-90 days later via chargebacks, and there is extreme class imbalance (~0.1% fraud). This means fraud models train on older, sparser data and must handle imbalance explicitly (weighted loss, SMOTE, stratified sampling).
          </li>
          <li>
            <strong>Adversarial dynamics:</strong> Users of a recommendation system don&apos;t try to fool the model. Fraudsters actively reverse-engineer detection logic and change tactics. This requires fraud systems to use diverse feature types (harder to simultaneously spoof), retrain more frequently, and maintain rule-based guardrails that are independent of the ML model.
          </li>
          <li>
            <strong>Explainability:</strong> Recommendations need no explanation (&quot;Because you watched X&quot; is nice but optional). Fraud decisions must be explainable &mdash; regulatory compliance requires telling blocked merchants why, and support teams need to review flagged transactions. This favors tree-based models with SHAP over opaque deep networks.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Netflix Tech Blog &mdash; &quot;System Architectures for Personalization and Recommendation&quot;</strong> &mdash; Deep dive into Netflix&apos;s multi-algorithm recommendation architecture, candidate generation strategies, and online/offline evaluation pipeline.</li>
          <li><strong>Stripe Engineering Blog &mdash; &quot;How we built Radar&quot;</strong> &mdash; Details on the hybrid rules + ML approach, network-level features, and how they handle adversarial drift in fraud patterns.</li>
          <li><strong>Google Research &mdash; &quot;Wide and Deep Learning for Recommender Systems&quot; (2016)</strong> &mdash; Foundational paper on combining memorization (wide/linear) with generalization (deep) for ranking, used across Google products.</li>
          <li><strong>Instagram Engineering &mdash; &quot;Powered by AI: Instagram&apos;s Explore Recommender System&quot;</strong> &mdash; Three-stage funnel, multi-objective optimization, balancing engagement quality with content diversity.</li>
          <li><strong>YouTube &mdash; &quot;Deep Neural Networks for YouTube Recommendations&quot; (2016)</strong> &mdash; The paper that defined the two-stage retrieval-ranking pattern adopted across the industry. Required reading for any system design interview.</li>
          <li><strong>Chip Huyen &mdash; &quot;Designing Machine Learning Systems&quot; (O&apos;Reilly)</strong> &mdash; Chapters 7-10 cover deployment, serving, distribution shifts, and monitoring with detailed case studies from Netflix, Uber, and others.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
