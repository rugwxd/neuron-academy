"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function FeatureStores() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A feature store is infrastructure that manages the <strong>entire lifecycle of features</strong> — from
          computation to storage to serving. It solves the most painful problem in production ML: the
          <strong> training-serving skew</strong>. During training, you compute features from historical data
          using batch jobs. During inference, you need the exact same features computed in real-time.
          Without a feature store, teams end up duplicating feature logic between Python training scripts
          and Java/Go serving code, leading to subtle bugs that silently degrade model performance.
        </p>
        <p>
          The architecture has two sides. The <strong>offline store</strong> holds historical feature values
          for training (typically backed by a data warehouse like BigQuery, Snowflake, or S3/Parquet files).
          The <strong>online store</strong> holds the latest feature values for real-time inference (backed by
          a low-latency key-value store like Redis, DynamoDB, or Bigtable). A materialization pipeline
          keeps them in sync: batch features are computed in the offline store and pushed to the online store;
          streaming features flow through Kafka/Kinesis and update both stores.
        </p>
        <p>
          <strong>Feast</strong> is the leading open-source feature store (used by companies like Shopify, Gojek,
          and Salesforce). <strong>Tecton</strong> (founded by the creators of Uber&apos;s Michelangelo) is the
          enterprise leader, powering feature pipelines at Atlassian, Plaid, and others.
          Other major platforms include Hopsworks (open-source, full MLOps) and built-in feature stores
          in Databricks and SageMaker.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Point-in-Time Joins (Avoiding Data Leakage)</h3>
        <p>
          The most critical operation in a feature store. When constructing a training dataset, each
          example has a timestamp <InlineMath math="t_i" />. For each feature, we must retrieve
          the <strong>most recent value at or before <InlineMath math="t_i" /></strong>:
        </p>
        <BlockMath math="\text{feature}(entity, t_i) = \arg\max_{t \leq t_i} \text{value}(entity, t)" />
        <p>
          This prevents <strong>look-ahead bias</strong> — using information from the future during training. Example: if you&apos;re
          predicting fraud for a transaction at 2pm, you must not use the user&apos;s spending pattern that
          includes transactions after 2pm.
        </p>

        <h3>Feature Freshness and Staleness</h3>
        <p>
          For a feature computed at time <InlineMath math="t_c" /> and served at
          time <InlineMath math="t_s" />, the <strong>staleness</strong> is:
        </p>
        <BlockMath math="\text{staleness} = t_s - t_c" />
        <p>
          Acceptable staleness depends on the feature type:
        </p>
        <ul>
          <li><strong>Batch features</strong> (user lifetime value, 30-day average): staleness of hours is fine. Recompute daily/hourly.</li>
          <li><strong>Near-real-time features</strong> (session click count, items in cart): staleness of seconds. Use streaming (Kafka + Flink).</li>
          <li><strong>Real-time features</strong> (current request context, device type): zero staleness — computed at request time, not stored.</li>
        </ul>

        <h3>Feature Versioning</h3>
        <p>
          Features are often improved over time (better aggregation, new data source). A feature store must
          track versions to ensure reproducibility:
        </p>
        <BlockMath math="\text{model}_v \rightarrow \text{feature\_set}_v = \{f_1^{v_1}, f_2^{v_3}, f_3^{v_1}, \ldots\}" />
        <p>
          Each model version is pinned to specific feature versions, enabling exact reproduction of training data.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Feast: Open-Source Feature Store</h3>
        <CodeBlock
          language="python"
          title="feast_feature_store.py"
          code={`# --- Step 1: Define feature views (feature_repo/features.py) ---
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from datetime import timedelta

# Data source (Parquet file, BigQuery, Redshift, etc.)
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
)

# Entity: the primary key for feature lookup
user = Entity(name="user_id", join_keys=["user_id"])

# Feature view: a group of related features from one source
user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=1),  # Feature freshness: recompute daily
    schema=[
        Field(name="total_purchases", dtype=Int64),
        Field(name="avg_order_value", dtype=Float32),
        Field(name="days_since_last_order", dtype=Int64),
        Field(name="lifetime_value", dtype=Float32),
    ],
    source=user_stats_source,
    online=True,  # Materialize to online store for serving
)

# --- Step 2: Apply definitions ---
# $ feast apply
# This registers feature views and creates online/offline stores

# --- Step 3: Materialize to online store ---
# $ feast materialize 2024-01-01T00:00:00 2024-12-31T23:59:59

# --- Step 4: Get training data (offline, point-in-time correct) ---
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo/")

# Entity dataframe: who + when (point-in-time join keys)
entity_df = pd.DataFrame({
    "user_id": [101, 102, 103, 101],
    "event_timestamp": pd.to_datetime([
        "2024-06-15 10:00:00",
        "2024-06-15 11:00:00",
        "2024-06-16 09:00:00",
        "2024-06-20 14:00:00",  # Same user, different time!
    ]),
})

# Point-in-time join: gets features as they were at each timestamp
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_stats:total_purchases",
        "user_stats:avg_order_value",
        "user_stats:lifetime_value",
    ],
).to_df()

print(training_df)
# Each row has features as of its event_timestamp — no leakage!

# --- Step 5: Get features for online serving ---
feature_vector = store.get_online_features(
    features=[
        "user_stats:total_purchases",
        "user_stats:avg_order_value",
        "user_stats:lifetime_value",
    ],
    entity_rows=[{"user_id": 101}],
).to_dict()

print(feature_vector)  # Latest feature values for real-time inference`}
        />

        <h3>Streaming Features with Kafka</h3>
        <CodeBlock
          language="python"
          title="streaming_features.py"
          code={`# Streaming feature computation with Flink-style processing
# This pattern is used at Uber, Stripe, DoorDash for near-real-time features

from datetime import datetime, timedelta
from collections import defaultdict
import time

class StreamingFeatureComputer:
    """
    Computes sliding window aggregations from an event stream.
    In production, this runs in Apache Flink, Spark Streaming, or Tecton.
    """
    def __init__(self, window_size_minutes=15):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.events = defaultdict(list)  # entity -> [(timestamp, value)]

    def process_event(self, entity_id, value, timestamp=None):
        """Process a new event from Kafka/Kinesis."""
        ts = timestamp or datetime.now()
        self.events[entity_id].append((ts, value))
        self._cleanup(entity_id, ts)

    def _cleanup(self, entity_id, current_time):
        """Remove events outside the window."""
        cutoff = current_time - self.window_size
        self.events[entity_id] = [
            (ts, v) for ts, v in self.events[entity_id] if ts > cutoff
        ]

    def get_features(self, entity_id):
        """Get current aggregated features for serving."""
        events = self.events.get(entity_id, [])
        if not events:
            return {"count": 0, "sum": 0, "mean": 0, "max": 0}

        values = [v for _, v in events]
        return {
            "count_15m": len(values),
            "sum_15m": sum(values),
            "mean_15m": sum(values) / len(values),
            "max_15m": max(values),
        }

# Example: real-time transaction monitoring
computer = StreamingFeatureComputer(window_size_minutes=15)

# Simulate transaction stream
now = datetime.now()
computer.process_event("user_101", 29.99, now - timedelta(minutes=10))
computer.process_event("user_101", 149.99, now - timedelta(minutes=5))
computer.process_event("user_101", 899.99, now - timedelta(minutes=1))

features = computer.get_features("user_101")
print(f"User 101 (last 15 min):")
print(f"  Transaction count: {features['count_15m']}")
print(f"  Total spent: \${features['sum_15m']:.2f}")
print(f"  Max transaction: \${features['max_15m']:.2f}")
# These features feed into a fraud detection model in real-time`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Start simple, add a feature store when you feel the pain</strong>: If you have 1-2 models and a small team, a well-organized set of SQL queries or dbt models is fine. Feature stores add value when you have 5+ models sharing features, or when training-serving skew becomes a real problem.</li>
          <li><strong>Online store latency budget</strong>: For real-time inference, feature retrieval should take &lt;10ms at p99. Redis and DynamoDB achieve this. A model serving 1000 QPS with 50 features needs ~50K key-value lookups per second.</li>
          <li><strong>Feature reuse saves engineering time</strong>: At Uber, the feature store holds ~10,000 features shared across ~100 models. Without it, each team would reimplement common features (user lifetime value, driver rating, etc.) independently.</li>
          <li><strong>Feast vs. Tecton</strong>: Feast is open-source and great for getting started — you control the infrastructure. Tecton is a managed service with built-in streaming, monitoring, and enterprise features. Choose based on your team size and operational maturity.</li>
          <li><strong>Feature monitoring is critical</strong>: Track feature distributions over time. A shift in feature values (data drift) is often the first signal that model performance is degrading.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Training-serving skew</strong>: The #1 problem. Training features computed in Python/Pandas, serving features computed in Java/SQL. Even small differences (different rounding, different null handling) silently degrade model performance by 2-10%. The feature store solves this by using the same definition for both.</li>
          <li><strong>Look-ahead bias in training data</strong>: Computing features without respecting timestamps. If you train on June data but compute &quot;user lifetime value&quot; using all data through December, you&apos;re leaking future information. Always use point-in-time joins.</li>
          <li><strong>Over-engineering streaming features</strong>: Not every feature needs sub-second freshness. If a user&apos;s &quot;average order value over 30 days&quot; is updated hourly instead of in real-time, the model accuracy difference is negligible. Streaming infrastructure is expensive — use it only where freshness actually matters (fraud detection, session-based recommendations).</li>
          <li><strong>No feature validation</strong>: Features can silently break — a null rate jumps from 1% to 50%, a numeric feature shifts by 10x. Without automated checks, bad features flow into models undetected.</li>
          <li><strong>Storing raw data as features</strong>: A feature store should hold <strong>engineered</strong> features, not raw events. Store &quot;user_transaction_count_30d,&quot; not every individual transaction.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You are building a fraud detection system that needs to make decisions within 100ms of a transaction arriving. Design the feature serving architecture. What features would be online vs. offline? How do you prevent training-serving skew?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Feature categories and serving strategy</strong>:
            <ul>
              <li><strong>Real-time (computed at request time)</strong>: Transaction amount, merchant category, time of day, device fingerprint. These come directly from the request payload — no storage needed.</li>
              <li><strong>Near-real-time (streaming, online store)</strong>: Transaction count in last 15 min, unique merchants in last hour, spending velocity. Computed via Kafka + Flink, pushed to Redis. Staleness: &lt;5 seconds.</li>
              <li><strong>Batch (offline store → online store)</strong>: User lifetime value, 30-day spending average, account age, historical fraud rate for merchant. Computed daily via Spark/dbt, materialized to DynamoDB. Staleness: hours.</li>
            </ul>
          </li>
          <li><strong>Architecture</strong>:
            <ul>
              <li>Transaction arrives → API gateway → feature service fetches batch features from Redis/DynamoDB (~5ms) + streaming features from Redis (~3ms) + extracts real-time features from request (~1ms).</li>
              <li>Combined feature vector → model inference (~10ms) → decision returned within 100ms budget.</li>
            </ul>
          </li>
          <li><strong>Preventing training-serving skew</strong>:
            <ul>
              <li>Define all features once in the feature store (Feast/Tecton). Training uses <code>get_historical_features()</code> with point-in-time joins. Serving uses <code>get_online_features()</code>. Same transformation code, same feature definitions.</li>
              <li>Log all served features alongside predictions. Periodically compare the distribution of served features vs. training features to detect drift.</li>
              <li>Run integration tests that compute the same feature both ways and assert equality.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Feast documentation (feast.dev)</strong> — Getting started with the leading open-source feature store.</li>
          <li><strong>Tecton blog: &quot;What is a Feature Store?&quot;</strong> — In-depth explanation from the creators of Uber&apos;s Michelangelo.</li>
          <li><strong>Hermann &amp; Del Balso (2017) &quot;Meet Michelangelo: Uber&apos;s ML Platform&quot;</strong> — The paper that popularized the feature store concept.</li>
          <li><strong>Chip Huyen &quot;Designing Machine Learning Systems&quot; Chapter 7</strong> — Feature engineering and feature stores in production.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
