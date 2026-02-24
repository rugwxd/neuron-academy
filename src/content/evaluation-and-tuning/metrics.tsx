"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function EvaluationMetrics() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Accuracy is the most intuitive metric — the fraction of predictions you got right. It&apos;s also the most dangerous. If 99% of credit card transactions are legitimate, a model that predicts &quot;not fraud&quot; for every single transaction achieves 99% accuracy while catching exactly zero fraud cases. <strong>The choice of metric determines what your model optimizes for.</strong>
        </p>
        <p>
          For classification, you need to understand the <strong>confusion matrix</strong>: true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN). From these four numbers, you can derive every metric. <strong>Precision</strong> asks &quot;of all the things I flagged as positive, how many actually were?&quot; <strong>Recall</strong> asks &quot;of all the actual positives, how many did I find?&quot; They trade off against each other, and <strong>F1</strong> is their harmonic mean.
        </p>
        <p>
          Threshold-independent metrics like <strong>ROC-AUC</strong> and <strong>PR-AUC</strong> evaluate your model across all possible decision thresholds. ROC-AUC measures how well your model ranks positive examples above negative ones. PR-AUC is more informative when classes are heavily imbalanced — which is most real-world problems.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Confusion Matrix</h3>
        <BlockMath math="\begin{array}{c|cc} & \text{Predicted +} & \text{Predicted -} \\ \hline \text{Actual +} & \text{TP} & \text{FN} \\ \text{Actual -} & \text{FP} & \text{TN} \end{array}" />

        <h3>Core Metrics</h3>
        <BlockMath math="\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}" />
        <BlockMath math="\text{Precision} = \frac{TP}{TP + FP} \quad \text{(of predicted positives, how many correct?)}" />
        <BlockMath math="\text{Recall (Sensitivity)} = \frac{TP}{TP + FN} \quad \text{(of actual positives, how many found?)}" />
        <BlockMath math="\text{Specificity} = \frac{TN}{TN + FP} \quad \text{(of actual negatives, how many correct?)}" />

        <h3>F1 Score</h3>
        <p>The harmonic mean of precision and recall:</p>
        <BlockMath math="F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}" />
        <p>The generalized <InlineMath math="F_\beta" /> score weights recall <InlineMath math="\beta" /> times more than precision:</p>
        <BlockMath math="F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}" />

        <h3>ROC-AUC</h3>
        <p>The ROC curve plots TPR (recall) vs FPR (<InlineMath math="1 - \text{specificity}" />) at all thresholds:</p>
        <BlockMath math="\text{TPR} = \frac{TP}{TP + FN}, \quad \text{FPR} = \frac{FP}{FP + TN}" />
        <p><strong>AUC</strong> = area under this curve. Interpretation: the probability that a random positive example is scored higher than a random negative example.</p>
        <BlockMath math="\text{AUC} = P(\hat{p}_{+} > \hat{p}_{-})" />

        <h3>Log Loss (Cross-Entropy)</h3>
        <BlockMath math="\text{Log Loss} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i)\right]" />
        <p>Penalizes confident wrong predictions harshly. Requires well-calibrated probabilities.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Comprehensive Metric Evaluation</h3>
        <CodeBlock
          language="python"
          title="evaluation_metrics.py"
          code={`import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

# Imbalanced dataset (5% positive)
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=10,
    weights=[0.95, 0.05], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --- All metrics ---
print("=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

print(f"\\nAccuracy:   {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:  {precision_score(y_test, y_pred):.4f}")
print(f"Recall:     {recall_score(y_test, y_pred):.4f}")
print(f"F1:         {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:    {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC:     {average_precision_score(y_test, y_prob):.4f}")
print(f"Log Loss:   {log_loss(y_test, y_prob):.4f}")

print("\\n" + classification_report(y_test, y_pred))`}
        />

        <h3>Threshold Optimization</h3>
        <CodeBlock
          language="python"
          title="threshold_optimization.py"
          code={`from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

# Find optimal threshold for F1
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Compute F1 at each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Default threshold (0.5): F1 = {f1_score(y_test, y_pred):.4f}")
y_pred_opt = (y_prob >= best_threshold).astype(int)
print(f"Optimal threshold ({best_threshold:.3f}): F1 = {f1_score(y_test, y_pred_opt):.4f}")

# For business cost: define cost(FP) and cost(FN), minimize total cost
cost_fp, cost_fn = 1, 10  # false negatives are 10x more costly
total_costs = []
for t in thresholds:
    y_t = (y_prob >= t).astype(int)
    fp = np.sum((y_t == 1) & (y_test == 0))
    fn = np.sum((y_t == 0) & (y_test == 1))
    total_costs.append(cost_fp * fp + cost_fn * fn)

best_cost_idx = np.argmin(total_costs)
print(f"Cost-optimal threshold: {thresholds[best_cost_idx]:.3f}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Never use accuracy alone on imbalanced data</strong>: A dataset with 1% fraud means a &quot;predict all normal&quot; model gets 99% accuracy. Use F1, PR-AUC, or a cost-sensitive metric.</li>
          <li><strong>ROC-AUC vs PR-AUC</strong>: ROC-AUC can be misleading with heavy imbalance — a model with many false positives can still look good because TN dominates. PR-AUC focuses on the positive class and is more informative.</li>
          <li><strong>Match the metric to the business problem</strong>: Medical screening = maximize recall (don&apos;t miss diseases). Spam filter = maximize precision (don&apos;t block legitimate email). Ad click prediction = optimize log loss (calibrated probabilities matter for bidding).</li>
          <li><strong>Threshold tuning is free performance</strong>: The default 0.5 threshold is almost never optimal. Use the precision-recall curve to find the threshold that maximizes your chosen metric.</li>
          <li><strong>Use macro/weighted F1 for multiclass</strong>: <code>average=&quot;macro&quot;</code> treats all classes equally (good for balanced data). <code>average=&quot;weighted&quot;</code> accounts for class size.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Optimizing accuracy when classes are imbalanced</strong>: The model will just predict the majority class. Always check per-class metrics.</li>
          <li><strong>Comparing ROC-AUC across different datasets</strong>: ROC-AUC depends on the class balance. A model with 0.95 AUC on balanced data is not necessarily better than one with 0.85 AUC on highly imbalanced data.</li>
          <li><strong>Using log loss with uncalibrated models</strong>: Tree-based models often output poorly calibrated probabilities. Calibrate with <code>CalibratedClassifierCV</code> before optimizing log loss.</li>
          <li><strong>Confusing micro and macro averaging</strong>: Micro averaging pools all predictions (dominated by majority classes). Macro averaging computes per-class metrics and averages (gives equal weight to all classes).</li>
          <li><strong>Reporting metrics without confidence intervals</strong>: A single F1 score is meaningless without variance. Use cross-validation or bootstrap to estimate uncertainty.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You&apos;re building a cancer screening model. The current model has 95% precision and 40% recall. The doctors want to improve recall to 80%. What are the consequences, and how would you approach this?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>The tradeoff</strong>: Precision and recall are inversely related at any fixed model capacity. Pushing recall from 40% to 80% will <em>necessarily</em> reduce precision — more false positives (healthy patients flagged for follow-up).</li>
          <li><strong>Lower the threshold</strong>: The quickest fix. Instead of predicting positive at probability &gt; 0.5, lower it to whatever threshold achieves 80% recall. Plot the precision-recall curve to see the exact tradeoff.</li>
          <li><strong>Cost-benefit analysis</strong>: A false negative (missed cancer) has much higher cost than a false positive (unnecessary biopsy). At 80% recall, if precision drops to, say, 60%, that means 40% of flagged patients are healthy — is this acceptable given the clinical setting?</li>
          <li><strong>Improve the model</strong>: To get higher recall without sacrificing as much precision, improve the model itself: more/better features, more training data, better algorithms, or ensemble methods.</li>
          <li><strong>Use <InlineMath math="F_2" /> score</strong>: <InlineMath math="F_\beta" /> with <InlineMath math="\beta = 2" /> weights recall twice as much as precision, aligning the metric with the clinical priority.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Saito &amp; Rehmsmeier (2015) &quot;The Precision-Recall Plot Is More Informative than the ROC Plot&quot;</strong> — Why PR-AUC should be preferred for imbalanced data.</li>
          <li><strong>Flach (2019) &quot;Performance Evaluation in Machine Learning&quot;</strong> — Comprehensive survey of metrics, proper scoring rules, and calibration.</li>
          <li><strong>scikit-learn Metrics guide</strong> — Complete reference for all available metrics with examples.</li>
          <li><strong>Google ML Crash Course: Classification metrics</strong> — Interactive visualizations of ROC and PR curves.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
