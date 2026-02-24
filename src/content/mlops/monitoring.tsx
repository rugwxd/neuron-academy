"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function ModelMonitoring() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A model that performs brilliantly at deployment will inevitably degrade over time. The world
          changes — user behavior shifts, new product categories appear, economic conditions fluctuate —
          and the patterns your model learned become stale. <strong>Model monitoring</strong> is the
          practice of continuously tracking model health to detect problems before they impact business outcomes.
        </p>
        <p>
          There are three types of degradation to watch for. <strong>Data drift</strong> means the
          distribution of incoming features has changed from what the model was trained on (e.g., your
          e-commerce model trained on US users suddenly receives traffic from a new country).
          <strong> Concept drift</strong> means the relationship between features and the target has
          changed (e.g., what constitutes a &quot;spam&quot; email evolves as spammers adapt). <strong>Performance
          degradation</strong> is a direct drop in model accuracy, precision, or other business metrics.
        </p>
        <p>
          Data drift is the early warning signal — you can detect it without ground truth labels.
          Concept drift and performance degradation require labeled data, which often arrives with a delay.
          A robust monitoring system tracks all three layers and triggers retraining pipelines when thresholds
          are breached.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Detecting Data Drift</h3>
        <p>
          <strong>Population Stability Index (PSI)</strong> quantifies how much a feature distribution
          has shifted between a reference (training) and current (production) dataset:
        </p>
        <BlockMath math="\text{PSI} = \sum_{i=1}^{B} (p_i - q_i) \cdot \ln\left(\frac{p_i}{q_i}\right)" />
        <p>
          where <InlineMath math="p_i" /> is the proportion of data in bin <InlineMath math="i" /> for the
          reference distribution and <InlineMath math="q_i" /> for the current distribution. Rules of thumb:
        </p>
        <ul>
          <li><InlineMath math="\text{PSI} < 0.1" />: No significant drift</li>
          <li><InlineMath math="0.1 \leq \text{PSI} < 0.25" />: Moderate drift — investigate</li>
          <li><InlineMath math="\text{PSI} \geq 0.25" />: Significant drift — retrain</li>
        </ul>

        <h3>Kolmogorov-Smirnov Test</h3>
        <p>
          For continuous features, the KS test measures the maximum distance between two cumulative
          distribution functions:
        </p>
        <BlockMath math="D = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|" />
        <p>
          If <InlineMath math="D" /> exceeds a critical value (or the p-value is below 0.05), the
          distributions are statistically significantly different.
        </p>

        <h3>Jensen-Shannon Divergence</h3>
        <p>
          A symmetric, bounded alternative to KL divergence for comparing distributions:
        </p>
        <BlockMath math="\text{JSD}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M), \quad M = \frac{1}{2}(P + Q)" />
        <p>
          <InlineMath math="\text{JSD} \in [0, 1]" /> (when using base-2 logarithm), making it easier
          to set universal thresholds.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Computing PSI from Scratch</h3>
        <CodeBlock
          language="python"
          title="psi_drift.py"
          code={`import numpy as np

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    Args:
        reference: Feature values from training data
        current: Feature values from production data
        bins: Number of bins for discretization

    Returns:
        PSI value (0 = identical, >0.25 = major drift)
    """
    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Compute proportions in each bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    curr_counts = np.histogram(current, bins=breakpoints)[0]

    # Avoid division by zero
    ref_pct = (ref_counts + 1e-6) / ref_counts.sum()
    curr_pct = (curr_counts + 1e-6) / curr_counts.sum()

    # PSI formula
    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return psi

# Example: No drift
np.random.seed(42)
ref_data = np.random.normal(0, 1, 10000)
curr_no_drift = np.random.normal(0, 1, 10000)
print(f"No drift PSI: {compute_psi(ref_data, curr_no_drift):.4f}")  # ~0.002

# Example: Moderate drift (mean shifted)
curr_moderate = np.random.normal(0.5, 1, 10000)
print(f"Moderate drift PSI: {compute_psi(ref_data, curr_moderate):.4f}")  # ~0.12

# Example: Major drift
curr_major = np.random.normal(2, 1.5, 10000)
print(f"Major drift PSI: {compute_psi(ref_data, curr_major):.4f}")  # ~1.5`}
        />

        <h3>Comprehensive Monitoring with Evidently</h3>
        <CodeBlock
          language="python"
          title="monitoring_evidently.py"
          code={`import pandas as pd
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    ClassificationPreset,
    TargetDriftPreset,
)

# Load reference (training) and current (production) data
reference_df = pd.read_csv("data/reference.csv")
current_df = pd.read_csv("data/current_week.csv")

# ---- 1. Data Drift Report ----
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(
    reference_data=reference_df,
    current_data=current_df,
)
drift_report.save_html("reports/data_drift.html")

# Programmatic access to drift results
drift_results = drift_report.as_dict()
dataset_drift = drift_results["metrics"][0]["result"]["dataset_drift"]
drifted_features = drift_results["metrics"][0]["result"]["number_of_drifted_columns"]
print(f"Dataset drift detected: {dataset_drift}")
print(f"Drifted features: {drifted_features}")

# ---- 2. Performance Monitoring (requires labels) ----
perf_report = Report(metrics=[ClassificationPreset()])
perf_report.run(
    reference_data=reference_df,
    current_data=current_df,
)
perf_report.save_html("reports/performance.html")

# ---- 3. Target/Prediction Drift ----
target_report = Report(metrics=[TargetDriftPreset()])
target_report.run(
    reference_data=reference_df,
    current_data=current_df,
)
target_report.save_html("reports/target_drift.html")`}
        />

        <h3>Automated Alerting Pipeline</h3>
        <CodeBlock
          language="python"
          title="monitoring_pipeline.py"
          code={`import json
import logging
from datetime import datetime
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_monitor")

class ModelMonitor:
    def __init__(self, reference_data, feature_names, psi_threshold=0.2, ks_alpha=0.05):
        self.reference = reference_data
        self.feature_names = feature_names
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.alerts = []

    def check_data_drift(self, current_data):
        """Run drift detection on all features."""
        drifted = []
        for i, name in enumerate(self.feature_names):
            ref_col = self.reference[:, i]
            curr_col = current_data[:, i]

            # KS test
            ks_stat, p_value = stats.ks_2samp(ref_col, curr_col)

            # PSI
            psi = compute_psi(ref_col, curr_col)

            if psi > self.psi_threshold or p_value < self.ks_alpha:
                drifted.append({
                    "feature": name,
                    "psi": round(psi, 4),
                    "ks_stat": round(ks_stat, 4),
                    "ks_pvalue": round(p_value, 6),
                })

        if drifted:
            self.alerts.append({
                "type": "DATA_DRIFT",
                "timestamp": datetime.utcnow().isoformat(),
                "drifted_features": drifted,
            })
            logger.warning(f"Data drift detected in {len(drifted)} features!")

        return drifted

    def check_performance(self, y_true, y_pred, baseline_metric=0.85):
        """Monitor model accuracy against baseline."""
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")

        if acc < baseline_metric * 0.95:  # 5% degradation threshold
            self.alerts.append({
                "type": "PERFORMANCE_DEGRADATION",
                "timestamp": datetime.utcnow().isoformat(),
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "baseline": baseline_metric,
            })
            logger.critical(f"Performance degradation! Acc: {acc:.4f} < {baseline_metric * 0.95:.4f}")

        return {"accuracy": acc, "f1": f1}

    def check_prediction_drift(self, current_preds, reference_preds):
        """Detect drift in model outputs."""
        ks_stat, p_value = stats.ks_2samp(reference_preds, current_preds)
        if p_value < self.ks_alpha:
            self.alerts.append({
                "type": "PREDICTION_DRIFT",
                "timestamp": datetime.utcnow().isoformat(),
                "ks_stat": round(ks_stat, 4),
                "p_value": round(p_value, 6),
            })
        return {"ks_stat": ks_stat, "p_value": p_value}

# Usage
# monitor = ModelMonitor(reference_data, feature_names)
# drifted = monitor.check_data_drift(current_batch)
# perf = monitor.check_performance(labels, predictions)`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Monitor inputs before outputs</strong>: Data drift detection does not require ground truth labels and gives you the earliest warning signal. Track feature distributions daily.</li>
          <li><strong>Set up tiered alerts</strong>: Warning at PSI &gt; 0.1 (investigate), critical at PSI &gt; 0.25 (trigger retraining). Do not page engineers for mild drift.</li>
          <li><strong>Log everything</strong>: Store all model inputs, outputs, and (eventually) ground truth labels. You need this data to diagnose why a model degraded and to build the next training set.</li>
          <li><strong>Use Evidently, WhyLabs, or Arize</strong>: Do not build a full monitoring stack from scratch. These tools provide dashboards, statistical tests, and alerting out of the box.</li>
          <li><strong>Shadow deployment</strong>: Run a new model alongside the old one, sending both the same traffic but only serving predictions from the old model. Compare outputs to catch regressions before they affect users.</li>
          <li><strong>Track business metrics, not just ML metrics</strong>: A 2% drop in AUC might not matter. A 5% drop in conversion rate definitely does. Connect model performance to business KPIs.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Only monitoring accuracy</strong>: Accuracy requires ground truth labels which are often delayed (days to weeks). By the time you see an accuracy drop, the model has been serving bad predictions for a while. Monitor data drift for early warning.</li>
          <li><strong>Using the wrong drift test</strong>: PSI works for binned data, KS test for continuous features, chi-squared for categorical features. Using the wrong test gives unreliable results.</li>
          <li><strong>Not establishing baselines</strong>: Without a reference distribution (from your training/validation set), you have nothing to compare against. Always save a reference snapshot at deployment time.</li>
          <li><strong>Alerting on every feature independently</strong>: With 200 features and a 5% significance level, you expect ~10 false alarms. Use correction methods (Bonferroni) or monitor aggregate drift scores.</li>
          <li><strong>Assuming retraining fixes everything</strong>: If concept drift is caused by a fundamental change (new regulation, market shift), retraining on old patterns will not help. You need new labeled data reflecting the new reality.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Your fraud detection model was deployed 6 months ago with 95% precision. The business team reports that false positives have increased. Walk through your debugging process.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Confirm the problem quantitatively</strong>: Pull recent predictions and ground truth labels. Compute precision, recall, and F1 weekly over the last 6 months. Is precision actually dropping, or is volume increasing (same rate, more absolute false positives)?</li>
          <li><strong>Check data drift</strong>: Compare the feature distributions of recent transactions to the training data using PSI. If features like transaction_amount, merchant_category, or device_type have drifted significantly, that&apos;s likely the root cause.</li>
          <li><strong>Check concept drift</strong>: Has the fraud rate itself changed? If new fraud patterns have emerged (new attack vectors), the model&apos;s learned decision boundary is no longer correct. Look at false positives — are they a new &quot;type&quot; of transaction the model hasn&apos;t seen?</li>
          <li><strong>Segment the analysis</strong>: Break down performance by customer segment, geography, transaction type. Often the degradation is localized (e.g., a new product line or market).</li>
          <li><strong>Remediation</strong>: Short-term — adjust the classification threshold to restore precision (at the cost of recall). Medium-term — retrain on recent data. Long-term — set up automated drift monitoring with retraining triggers, and add the new fraud patterns to your feature engineering pipeline.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Evidently AI documentation</strong> — Open-source ML monitoring with drift detection, performance tracking, and dashboards.</li>
          <li><strong>&quot;Monitoring Machine Learning Models in Production&quot; by Christopher Samiullah</strong> — Practical guide to production ML monitoring patterns.</li>
          <li><strong>&quot;Designing Machine Learning Systems&quot; by Chip Huyen, Ch. 8-9</strong> — Data distribution shifts and continual learning.</li>
          <li><strong>Google &quot;ML Test Score&quot; paper</strong> — A rubric for ML production readiness, including monitoring requirements.</li>
          <li><strong>NannyML documentation</strong> — Performance estimation without ground truth labels using confidence-based approaches.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
