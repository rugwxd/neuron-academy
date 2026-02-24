"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function MLSystemDesignQuestions() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          ML system design interviews test your ability to take a vague business problem and architect a
          complete machine learning system — from data collection through model training to serving and
          monitoring. These interviews evaluate breadth (do you think about all the pieces?) and depth
          (can you dive deep when asked?).
        </p>
        <p>
          The framework for answering any ML system design question is: (1) <strong>Clarify requirements</strong> —
          what metric matters, latency constraints, scale. (2) <strong>Data</strong> — what data exists, how to
          label it, feature engineering. (3) <strong>Model</strong> — baseline, iterate, offline evaluation.
          (4) <strong>Serving</strong> — online vs batch, latency, throughput. (5) <strong>Monitoring</strong> —
          drift detection, A/B testing, feedback loops.
        </p>
        <p>
          Below are 8 full case studies covering the most commonly asked ML system design questions. Each includes
          a detailed walkthrough you could deliver in a 45-minute interview.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Key Metrics for System Design</h3>
        <p><strong>Recommendation quality:</strong></p>
        <BlockMath math="\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}, \quad \text{DCG@k} = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}" />

        <p><strong>Retrieval quality:</strong></p>
        <BlockMath math="\text{Precision@k} = \frac{|\text{relevant items in top-k}|}{k}, \quad \text{Recall@k} = \frac{|\text{relevant items in top-k}|}{|\text{all relevant items}|}" />

        <p><strong>Online experiment (A/B test) sample size:</strong></p>
        <BlockMath math="n \geq \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}" />
        <p>
          where <InlineMath math="\delta" /> is the minimum detectable effect and <InlineMath math="\sigma^2" /> is the variance of the metric.
        </p>

        <p><strong>Approximate Nearest Neighbors (ANN) tradeoff:</strong></p>
        <BlockMath math="\text{Recall} = \frac{|\text{ANN result} \cap \text{exact kNN result}|}{k}" />
        <p>Higher recall = better quality, but slower. Tune <code>nprobe</code> (FAISS) or <code>ef_search</code> (HNSW) to balance.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Case Study 1: News Feed Ranking (Facebook/Meta Style)</h3>
        <CodeBlock
          language="python"
          title="case_1_newsfeed_ranking.py"
          code={`"""
PROBLEM: Design a system that ranks posts in a social media news feed.
Users see hundreds of potential posts; show the most relevant ones first.

REQUIREMENTS CLARIFICATION:
- 1B daily active users, each with ~1000 candidate posts
- Latency: < 200ms for ranking
- Optimize for engagement (likes, comments, shares, time spent)
- Must balance relevance, diversity, and content policy

ARCHITECTURE:
1. Candidate Generation (reduce 1000 -> ~200 posts)
   - Recent posts from followed accounts
   - Posts from groups/pages
   - "Friends of friends" viral content

2. Ranking Model (score 200 posts)
   - Multi-task learning: predict P(like), P(comment), P(share), P(hide)
   - Final score = weighted combination:
"""

# Scoring formula
def compute_feed_score(predictions: dict, weights: dict) -> float:
    """
    Multi-objective ranking score.
    predictions: {'p_like': 0.3, 'p_comment': 0.1, 'p_share': 0.05, 'p_hide': 0.02}
    weights: {'like': 1.0, 'comment': 3.0, 'share': 5.0, 'hide': -10.0}
    """
    score = sum(predictions[f"p_{action}"] * weights[action]
                for action in weights)
    return score

"""
3. Features:
   - User features: age, country, past engagement rates, active hours
   - Post features: type (photo/video/text), age, creator stats, text embedding
   - Cross features: user-creator affinity, topic match, time since last interaction
   - Context: time of day, device, network speed

4. Model Architecture:
   - Two-tower for candidate retrieval (user tower + post tower)
   - Deep & Cross Network (DCN) or DeepFM for ranking
   - Serve with TorchServe or Triton with dynamic batching

5. Online Serving Pipeline:
   Candidate Gen (50ms) -> Feature Fetch (30ms) -> Ranking (50ms)
   -> Re-ranking/Policy (20ms) -> Response (total < 200ms)

6. Monitoring:
   - Engagement rate (likes/impressions), time spent
   - Diversity metrics (how many unique creators shown)
   - Content policy violations surfaced
   - A/B test new ranking models with 1% traffic holdout
"""`}
        />

        <h3>Case Study 2: Fraud Detection System (Stripe/PayPal Style)</h3>
        <CodeBlock
          language="python"
          title="case_2_fraud_detection.py"
          code={`"""
PROBLEM: Build a real-time fraud detection system for online payments.
Flag fraudulent transactions before they are processed.

REQUIREMENTS:
- 10K transactions/second at peak
- Latency: < 100ms per transaction decision
- Precision > 95% (minimize false positives that block good users)
- Recall > 80% (catch most fraud)
- Fraud rate: ~0.1% of transactions

ARCHITECTURE:

1. RULE ENGINE (first layer, <5ms):
   - Hard rules: blocked countries, velocity checks (>5 txns in 1 min)
   - Known fraud patterns from investigations
   - Catches ~30% of fraud, near-zero false positives

2. ML MODEL (second layer, <50ms):
   - Gradient boosted trees (XGBoost/LightGBM) for tabular features
   - Why not deep learning? GBTs are faster, more interpretable,
     and work better on structured/tabular data
"""

# Feature engineering for fraud detection
features = {
    # Transaction features
    "amount": "Transaction amount",
    "amount_zscore": "(amount - user_avg) / user_std",
    "is_round_amount": "amount % 100 == 0",

    # Velocity features (real-time aggregation)
    "txn_count_1h": "Number of transactions by this user in last 1 hour",
    "txn_count_24h": "Number of transactions in last 24 hours",
    "unique_merchants_1h": "Distinct merchants in last 1 hour",
    "total_amount_24h": "Total spend in last 24 hours",

    # Device/location features
    "new_device": "First time seeing this device fingerprint",
    "distance_from_last": "Geo distance from last transaction",
    "impossible_travel": "distance / time_delta > 900 km/h",

    # Behavioral features
    "hour_of_day": "Local hour of transaction",
    "is_typical_merchant_category": "User has bought from this category before",
    "days_since_last_txn": "Recency",
}

"""
3. HANDLING CLASS IMBALANCE (0.1% fraud):
   - Do NOT oversample for training - use scale_pos_weight in XGBoost
   - Evaluate with precision-recall curve, NOT accuracy or ROC
   - Optimize threshold for business constraint: precision >= 95%

4. REAL-TIME FEATURE STORE:
   - Redis for velocity features (increment counters per user)
   - Precomputed user profiles updated hourly in batch
   - Feature latency budget: 20ms

5. HUMAN-IN-THE-LOOP:
   - Medium-confidence predictions (0.3 < score < 0.7) go to manual review
   - Reviewer labels feed back into training data (active learning)

6. MONITORING:
   - Precision/recall on labeled transactions (delayed by chargeback window)
   - Feature drift on transaction amounts, velocity patterns
   - Alert if fraud rate changes by > 2x in any 1-hour window
"""`}
        />

        <h3>Case Study 3: Search Ranking (Google/Bing Style)</h3>
        <CodeBlock
          language="python"
          title="case_3_search_ranking.py"
          code={`"""
PROBLEM: Design the ML system behind a web search engine ranking.
Given a query, rank billions of web pages by relevance.

MULTI-STAGE ARCHITECTURE:

Stage 1: RETRIEVAL (<10ms, billions -> 10K documents)
- Inverted index (BM25) for keyword matching
- Embedding-based retrieval (dual encoder) for semantic matching
- Union of both candidate sets

Stage 2: PRE-RANKING (<20ms, 10K -> 500 documents)
- Lightweight model (small GBRT or distilled neural model)
- Fast features: BM25 score, page authority, click-through rate

Stage 3: RANKING (<50ms, 500 -> ~20 documents)
- Full-featured neural ranker (cross-encoder)
- BERT-based model that jointly encodes query + document
"""

# Cross-encoder scoring
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-12-v2"
)

def score_query_doc_pairs(query: str, documents: list[str]) -> list[float]:
    """Score query-document relevance using cross-encoder."""
    pairs = [[query, doc] for doc in documents]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)
    return scores.tolist()

"""
KEY FEATURES:
- Query features: length, intent classification, entity detection
- Document features: PageRank, freshness, domain authority, content quality
- Query-document features: BM25 score, semantic similarity, title match
- User features: location, language, search history
- Click features: historical CTR for this query-URL pair

TRAINING DATA:
- Click logs (biased but abundant)
- Human relevance judgments (expensive but unbiased)
- Use inverse propensity weighting to debias click data

EVALUATION:
- Offline: NDCG@10 on human-judged query sets
- Online: A/B test measuring click-through rate, time-to-click,
  query reformulation rate (lower = better)
"""`}
        />

        <h3>Case Study 4: Recommendation System (Netflix/Spotify Style)</h3>
        <CodeBlock
          language="python"
          title="case_4_recommendation.py"
          code={`"""
PROBLEM: Build a recommendation system for a streaming platform.
Recommend movies/shows that users will enjoy and engage with.

TWO-STAGE ARCHITECTURE:

Stage 1: CANDIDATE GENERATION (millions -> 500 items)
- Collaborative filtering: users who watched X also watched Y
- Content-based: similar genre, director, cast
- Embedding retrieval: two-tower model (user embedding + item embedding)
"""

import numpy as np

class TwoTowerRetrieval:
    """Simplified two-tower model for candidate generation."""
    def __init__(self, user_embeddings, item_embeddings):
        # user_embeddings: (n_users, dim)
        # item_embeddings: (n_items, dim)
        self.user_emb = user_embeddings
        self.item_emb = item_embeddings

    def get_candidates(self, user_id: int, k: int = 500) -> list[int]:
        """Retrieve top-k items by dot-product similarity."""
        user_vec = self.user_emb[user_id]  # (dim,)
        scores = self.item_emb @ user_vec  # (n_items,)
        top_k = np.argpartition(scores, -k)[-k:]
        return top_k[np.argsort(scores[top_k])[::-1]]

"""
Stage 2: RANKING (500 -> 20-50 items)
- Neural ranking model with rich features:

  User features: watch history embedding, genre preferences,
                 time-of-day patterns, device type
  Item features: content embedding, popularity, recency,
                 average rating, genre one-hot
  Cross features: user-genre affinity, "users like you watched this",
                  collaborative filtering score

- Multi-task: predict P(click), P(watch > 50%), P(complete), P(add to list)
- Final score: weighted combination tuned to optimize retention

Stage 3: RE-RANKING (business logic)
- Diversity: don't show 10 action movies in a row
- Freshness: boost new releases
- Business rules: promote original content, contractual obligations

COLD START PROBLEM:
- New users: use demographic-based recommendations, popularity,
  or an onboarding quiz
- New items: use content features (genre, description embedding),
  seed with editorial picks

SERVING:
- Pre-compute candidate sets hourly in batch (user embeddings are stable)
- Real-time ranking on request (captures current context)
- Cache top recommendations for popular user segments

EVALUATION:
- Offline: recall@k for retrieval, NDCG@k for ranking
- Online A/B test: watch time, completion rate, retention (D7, D30)
"""`}
        />

        <h3>Case Study 5: Estimated Time of Arrival (Uber/Lyft Style)</h3>
        <CodeBlock
          language="python"
          title="case_5_eta_prediction.py"
          code={`"""
PROBLEM: Predict estimated time of arrival (ETA) for ride-hailing.
Given pickup and dropoff locations, predict trip duration.

REQUIREMENTS:
- Accuracy: mean absolute error < 2 minutes for trips under 30 min
- Latency: < 50ms (shown to user while booking)
- Scale: 10M+ predictions/day

FEATURES:
"""
features = {
    # Route features
    "haversine_distance": "Straight-line distance between points",
    "road_distance": "Actual road network distance (OSRM/Google)",
    "num_turns": "Number of turns on optimal route",
    "road_type_mix": "% highway vs local roads",

    # Spatial features
    "pickup_h3_cell": "H3 hexagonal grid cell for pickup location",
    "dropoff_h3_cell": "H3 cell for dropoff",
    "pickup_poi_density": "Points of interest near pickup (proxy for congestion)",

    # Temporal features
    "hour_of_day": "Cyclical encoding: sin(2*pi*hour/24), cos(...)",
    "day_of_week": "Cyclical encoding",
    "is_rush_hour": "7-9am or 5-7pm on weekdays",
    "is_holiday": "Public holiday flag",

    # Real-time features
    "current_speed_on_route": "Average speed from GPS traces in last 5 min",
    "surge_multiplier": "Proxy for current demand (correlated with traffic)",
    "weather_condition": "Rain, snow, clear (from weather API)",
    "num_active_trips_in_area": "Proxy for local congestion",

    # Historical features
    "historical_avg_speed_this_route_hour": "Average speed on this OD pair at this hour",
    "historical_trip_time_p50": "Median historical trip time for this OD pair",
}

"""
MODEL:
- Gradient boosted trees (LightGBM) as primary model
  - Handles tabular data well, fast inference, interpretable
  - Train on completed trip data (millions of training examples)
  - Target: log(trip_duration_minutes) — log transform for better distribution

- Graph neural network for road segments (advanced):
  - Model traffic as a graph where nodes = intersections, edges = road segments
  - Aggregate real-time speed information along the route

LOSS FUNCTION:
- Quantile regression for prediction intervals:
  - Predict 10th, 50th, 90th percentiles
  - Show user the 50th percentile as ETA
  - Use spread (p90 - p10) as confidence indicator

TRAINING:
- Billions of completed trips with actual duration
- Train/val/test split by time (never leak future data!)
- Retrain weekly to capture seasonal patterns

MONITORING:
- MAE and MAPE by city, time of day, distance bucket
- Alert if MAE degrades > 10% for any segment
- A/B test model updates: does better ETA improve booking completion rate?
"""`}
        />

        <h3>Case Study 6: Content Moderation (TikTok/YouTube Style)</h3>
        <CodeBlock
          language="python"
          title="case_6_content_moderation.py"
          code={`"""
PROBLEM: Automatically detect harmful content (hate speech, violence,
misinformation) across text, images, and video at scale.

REQUIREMENTS:
- Process 500M+ pieces of content/day
- Latency: < 500ms for blocking before publication (pre-screening)
- Recall > 95% for severe violations (must catch almost everything)
- Precision > 90% (minimize wrongful takedowns)

MULTI-MODAL ARCHITECTURE:

1. TEXT MODERATION:
   - Fine-tuned BERT/RoBERTa for toxicity classification
   - Multi-label: hate_speech, harassment, violence, self_harm, sexual, spam
   - Train on human-labeled examples + synthetic data augmentation

2. IMAGE MODERATION:
   - Fine-tuned ViT or EfficientNet
   - Detect: nudity, violence, graphic content, manipulated media
   - OCR + text classifier for text-in-image (memes, screenshots)

3. VIDEO MODERATION:
   - Sample keyframes (1 per second) through image classifier
   - Audio transcription (Whisper) -> text classifier
   - Temporal model for context (a medical procedure vs violence)

4. MULTI-MODAL FUSION:
   - Late fusion: combine text, image, video scores
   - Cross-modal: CLIP-style model detects mismatches
     (benign text + harmful image)

DECISION FRAMEWORK:
- High confidence harmful (score > 0.95): auto-remove
- Medium confidence (0.5 - 0.95): send to human review queue
- Low confidence (< 0.5): allow with monitoring

ADVERSARIAL ROBUSTNESS:
- Users actively try to evade detection (leetspeak, unicode tricks)
- Character-level models + normalization preprocessing
- Regularly red-team the system with new evasion techniques
- Retrain monthly with new adversarial examples

FAIRNESS:
- Audit for disparate impact across languages and dialects
- AAE (African American English) often falsely flagged by toxicity models
- Separate thresholds or additional features to reduce bias
"""`}
        />

        <h3>Case Study 7: Dynamic Pricing (Airbnb/Amazon Style)</h3>
        <CodeBlock
          language="python"
          title="case_7_dynamic_pricing.py"
          code={`"""
PROBLEM: Build a system that suggests optimal listing prices for hosts
on a rental platform to maximize bookings and revenue.

REQUIREMENTS:
- Update price suggestions daily for 7M+ listings
- Accuracy: within 10% of market-clearing price
- Must explain pricing to hosts ("your price is X because...")

MODEL ARCHITECTURE:

1. DEMAND FORECASTING:
   - Predict booking probability as a function of price
   - Features: location, property type, amenities, photos quality score,
     host rating, seasonality, local events, competitor prices
   - Model: LightGBM for demand curve estimation

2. PRICE OPTIMIZATION:
   - Given demand curve, find price that maximizes expected revenue:
"""

import numpy as np
from scipy.optimize import minimize_scalar

def optimal_price(demand_model, features: dict, min_price: float,
                  max_price: float) -> float:
    """
    Find price that maximizes expected revenue = price * P(booking|price).
    """
    def neg_revenue(price):
        features_with_price = {**features, "price": price}
        p_booking = demand_model.predict_proba(features_with_price)
        return -(price * p_booking)

    result = minimize_scalar(neg_revenue, bounds=(min_price, max_price),
                             method="bounded")
    return result.x

"""
3. COMPETITIVE ANALYSIS:
   - Scrape comparable listings within 2km, similar size/amenities
   - Compute percentile of suggested price vs competitors
   - Warn hosts if they are significantly above/below market

4. EXPLAINABILITY (critical for host adoption):
   - SHAP values for top price drivers
   - "Your price is $150/night because:
     - Location premium: +$30 (downtown)
     - Weekend: +$20
     - Below average rating: -$15
     - Large event nearby: +$25"

5. A/B TESTING:
   - Randomly assign hosts to treatment (ML pricing) vs control (own pricing)
   - Primary metric: revenue per available night
   - Secondary: booking rate, host satisfaction (survey)
   - Run for 4+ weeks to capture weekly patterns

6. GUARDRAILS:
   - Never suggest price below host's minimum
   - Cap daily price changes at 20% to avoid shocking hosts
   - Human review for top-revenue listings
"""`}
        />

        <h3>Case Study 8: Autocomplete / Query Suggestions (Google Style)</h3>
        <CodeBlock
          language="python"
          title="case_8_autocomplete.py"
          code={`"""
PROBLEM: As a user types a search query, suggest completions in real-time.
Suggestions should be relevant, popular, and safe.

REQUIREMENTS:
- Latency: < 50ms (must feel instant)
- Update suggestions with each keystroke
- Handle 100K+ queries per second
- No offensive or harmful suggestions

ARCHITECTURE:

1. OFFLINE: Build suggestion index (daily)
   - Source: past query logs (billions of queries)
   - Aggregate: count queries by prefix, weighted by recency
   - Filter: remove queries below minimum frequency threshold
   - Safety: block harmful queries (profanity filter + ML classifier)
   - Build trie or sorted prefix index

2. ONLINE: Serve suggestions (per keystroke)
"""

class TrieNode:
    def __init__(self):
        self.children = {}
        self.suggestions = []  # top-k suggestions for this prefix

class AutocompleteTrie:
    def __init__(self):
        self.root = TrieNode()

    def build(self, query_counts: dict, k: int = 10):
        """Build trie with top-k suggestions at each node."""
        # Sort queries by count
        sorted_queries = sorted(query_counts.items(), key=lambda x: -x[1])

        for query, count in sorted_queries:
            node = self.root
            for char in query.lower():
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                # Keep top-k suggestions at each prefix node
                if len(node.suggestions) < k:
                    node.suggestions.append((query, count))

    def get_suggestions(self, prefix: str) -> list[str]:
        """O(len(prefix)) lookup — no traversal of subtree needed."""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        return [q for q, _ in node.suggestions]

"""
3. PERSONALIZATION:
   - Blend global popularity with user's search history
   - score = 0.7 * global_popularity + 0.3 * user_affinity
   - User history stored in a fast key-value store (Redis)

4. REAL-TIME TRENDING:
   - Detect queries spiking in last 1 hour (e.g., breaking news)
   - Inject trending queries into suggestions with a boost factor
   - Use streaming aggregation (Kafka + Flink) for real-time counts

5. SERVING:
   - Trie sharded by first character(s) across multiple servers
   - Each shard fits in memory (trie is compact)
   - CDN caching for the most common prefixes ("how to", "what is")

6. EVALUATION:
   - Offline: MRR (Mean Reciprocal Rank) — is the user's final query
     in the top suggestions?
   - Online: suggestion acceptance rate, queries-per-session (lower = better)
"""`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always start with requirements clarification</strong>: Latency budget, scale (QPS), what metric to optimize, and constraints. This frames the entire design.</li>
          <li><strong>Propose a baseline first</strong>: &quot;We could start with logistic regression on handcrafted features&quot; shows maturity. Then iterate toward more complex solutions.</li>
          <li><strong>Data is usually the bottleneck</strong>: Spend time discussing data collection, labeling strategies, and handling class imbalance. Models are easy; good data is hard.</li>
          <li><strong>Think in stages</strong>: Most production ML systems are multi-stage (retrieval → ranking → re-ranking). Single monolithic models rarely work at scale.</li>
          <li><strong>Discuss tradeoffs explicitly</strong>: &quot;We could use a larger model for +2% accuracy but it would triple latency. Given our 50ms budget, I recommend...&quot;</li>
          <li><strong>Don&apos;t forget monitoring and iteration</strong>: Mention A/B testing, drift detection, and the feedback loop. This separates senior from junior candidates.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Jumping straight to the model</strong>: Interviewers want to see you think about the full system. Spending 30 minutes on model architecture and 0 minutes on data, serving, and monitoring is a red flag.</li>
          <li><strong>Ignoring latency constraints</strong>: A BERT cross-encoder is great for ranking but takes 50ms per document. If you have 1000 candidates and a 100ms budget, the math does not work. Always check feasibility.</li>
          <li><strong>Not discussing offline vs online evaluation</strong>: Good offline metrics (AUC, NDCG) do not guarantee online improvement. Always mention A/B testing.</li>
          <li><strong>Forgetting about cold start</strong>: New users and new items break collaborative filtering. Always discuss fallback strategies.</li>
          <li><strong>Over-engineering</strong>: For a startup with 10K users, you do not need a two-tower model with FAISS indexing. Match complexity to scale.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Meta-Question:</strong> How do you structure a 45-minute ML system design interview?</p>
        <p><strong>Framework:</strong></p>
        <ol>
          <li><strong>Minutes 0-5: Clarify requirements</strong>
            <ul>
              <li>What is the business objective? What metric matters most?</li>
              <li>Scale: QPS, data size, number of users/items?</li>
              <li>Latency and throughput requirements?</li>
              <li>What data is available? Are labels available?</li>
            </ul>
          </li>
          <li><strong>Minutes 5-10: High-level architecture</strong>
            <ul>
              <li>Draw the system diagram: data pipeline → feature store → model → serving → monitoring</li>
              <li>Identify the stages (retrieval → ranking → re-ranking if applicable)</li>
            </ul>
          </li>
          <li><strong>Minutes 10-20: Data and features</strong>
            <ul>
              <li>Training data sources, labeling strategy, class balance</li>
              <li>Key features and feature engineering</li>
              <li>Online vs offline features, feature store design</li>
            </ul>
          </li>
          <li><strong>Minutes 20-30: Model</strong>
            <ul>
              <li>Baseline (simple model), then iterate</li>
              <li>Model architecture, loss function, training strategy</li>
              <li>Offline evaluation metrics</li>
            </ul>
          </li>
          <li><strong>Minutes 30-40: Serving and monitoring</strong>
            <ul>
              <li>Batch vs real-time serving, caching, latency optimization</li>
              <li>A/B testing strategy, guardrail metrics</li>
              <li>Monitoring: data drift, model performance, business metrics</li>
            </ul>
          </li>
          <li><strong>Minutes 40-45: Deep dive on interviewer&apos;s choice</strong>
            <ul>
              <li>Be ready to go deep on any component</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>&quot;Designing Machine Learning Systems&quot; by Chip Huyen</strong> — The best book on production ML system design. Covers all the topics interviewers care about.</li>
          <li><strong>Stanford CS 329S: Machine Learning Systems Design</strong> — Chip Huyen&apos;s course materials are freely available online.</li>
          <li><strong>&quot;System Design Interview - An Insider&apos;s Guide Vol 2&quot; by Alex Xu</strong> — Several ML system design chapters (nearby search, notification system).</li>
          <li><strong>Meta, Google, Netflix engineering blogs</strong> — Real-world descriptions of production ML systems at scale.</li>
          <li><strong>Educative &quot;Grokking the Machine Learning Interview&quot;</strong> — Structured practice problems with solutions.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
