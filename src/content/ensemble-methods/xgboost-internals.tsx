"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function XGBoostInternals() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          XGBoost (eXtreme Gradient Boosting) isn&apos;t just &quot;gradient boosting but faster.&quot;
          It&apos;s a carefully engineered system that combines three innovations: a <strong>regularized
          objective</strong> that prevents overfitting at the tree-building level, an <strong>approximate
          split-finding algorithm</strong> that makes it scalable to billions of rows, and <strong>system-level
          optimizations</strong> (column-block storage, cache-aware access, out-of-core computation) that
          make it fast on real hardware.
        </p>
        <p>
          Where standard gradient boosting fits trees to pseudo-residuals and hopes for the best,
          XGBoost formulates the exact objective function including a regularization term on tree
          complexity, then derives the <strong>optimal leaf weights</strong> and a <strong>scoring function</strong>
          for evaluating splits — all in closed form. This means every split decision is mathematically
          optimal given the objective.
        </p>
        <p>
          The result: XGBoost (and its successors LightGBM and CatBoost) dominate structured/tabular
          data tasks. If your data lives in a table (not images or text), gradient boosted trees are
          almost certainly your best model.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Regularized Objective</h3>
        <p>
          At round <InlineMath math="t" />, XGBoost minimizes:
        </p>
        <BlockMath math="\mathcal{L}^{(t)} = \sum_{i=1}^{n} L\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)" />
        <p>
          where <InlineMath math="\Omega" /> penalizes tree complexity:
        </p>
        <BlockMath math="\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2" />
        <p>
          Here <InlineMath math="T" /> is the number of leaves, <InlineMath math="w_j" /> is the weight
          of leaf <InlineMath math="j" />, <InlineMath math="\gamma" /> penalizes tree size, and <InlineMath math="\lambda" /> is
          L2 regularization on leaf weights.
        </p>

        <h3>Second-Order Taylor Approximation</h3>
        <p>
          XGBoost takes a second-order Taylor expansion of the loss around <InlineMath math="\hat{y}^{(t-1)}" />:
        </p>
        <BlockMath math="\mathcal{L}^{(t)} \approx \sum_{i=1}^{n}\left[g_i f_t(x_i) + \frac{1}{2}h_i f_t(x_i)^2\right] + \Omega(f_t)" />
        <p>where the gradient and Hessian statistics are:</p>
        <BlockMath math="g_i = \frac{\partial L(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}, \quad h_i = \frac{\partial^2 L(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}" />

        <h3>Optimal Leaf Weights</h3>
        <p>
          Grouping samples by their leaf assignment <InlineMath math="I_j = \{i \mid x_i \text{ falls in leaf } j\}" />,
          the optimal weight for leaf <InlineMath math="j" /> is:
        </p>
        <BlockMath math="w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}" />

        <h3>Split Gain (Scoring Function)</h3>
        <p>
          The gain from splitting a node into left and right children:
        </p>
        <BlockMath math="\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma" />
        <p>
          where <InlineMath math="G_L = \sum_{i \in I_L} g_i" />, <InlineMath math="H_L = \sum_{i \in I_L} h_i" />,
          and similarly for the right child. The <InlineMath math="\gamma" /> term acts as a minimum gain
          threshold — splits that don&apos;t improve the objective by at least <InlineMath math="\gamma" /> are
          pruned. This is <strong>pre-pruning with a principled threshold</strong>.
        </p>

        <h3>Approximate Split Finding (Histogram Method)</h3>
        <p>
          Instead of evaluating every possible threshold (exact greedy), XGBoost can bucket feature values
          into quantiles and only evaluate splits at bucket boundaries. LightGBM takes this further with
          its histogram-based algorithm, reducing split complexity from <InlineMath math="O(n \cdot d)" /> to <InlineMath math="O(\text{bins} \cdot d)" />.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>XGBoost with Proper Tuning</h3>
        <CodeBlock
          language="python"
          title="xgboost_tuned.py"
          code={`import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=10000, n_features=20,
    n_informative=12, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use DMatrix for performance
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,              # Tree depth (3-10)
    "learning_rate": 0.05,       # aka eta — lower = more trees needed
    "subsample": 0.8,            # Row sampling per tree
    "colsample_bytree": 0.8,     # Feature sampling per tree
    "lambda": 1.0,               # L2 regularization on leaf weights
    "alpha": 0.0,                # L1 regularization on leaf weights
    "gamma": 0.1,                # Min split gain threshold
    "min_child_weight": 5,       # Min sum of Hessians in a leaf
    "tree_method": "hist",       # Histogram-based (fast)
    "seed": 42,
}

# Train with early stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, "train"), (dtest, "val")],
    early_stopping_rounds=50,
    verbose_eval=100,
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best val logloss: {model.best_score:.4f}")

# Feature importance
importance = model.get_score(importance_type="gain")
for feat, score in sorted(importance.items(), key=lambda x: -x[1])[:5]:
    print(f"  {feat}: {score:.1f}")`}
        />

        <h3>XGBoost Split Gain — Manual Computation</h3>
        <CodeBlock
          language="python"
          title="split_gain_manual.py"
          code={`import numpy as np

def compute_split_gain(g_left, h_left, g_right, h_right, lambda_reg, gamma):
    """
    Compute the XGBoost split gain formula.
    g = sum of gradients, h = sum of hessians.
    """
    def leaf_score(G, H):
        return G**2 / (H + lambda_reg)

    gain = 0.5 * (
        leaf_score(g_left, h_left)
        + leaf_score(g_right, h_right)
        - leaf_score(g_left + g_right, h_left + h_right)
    ) - gamma

    return gain

# Example: binary classification (log loss)
# Suppose left child has 60 positives, 40 negatives
# Current predictions are all 0.5
# g_i = pred - y_i, h_i = pred * (1 - pred) = 0.25
g_left = 40 * 0.5 + 60 * (-0.5)   # = -10
h_left = 100 * 0.25                # = 25
g_right = 30 * 0.5 + 70 * (-0.5)  # = -20
h_right = 100 * 0.25              # = 25

gain = compute_split_gain(g_left, h_left, g_right, h_right,
                          lambda_reg=1.0, gamma=0.0)
print(f"Split gain: {gain:.4f}")

# Optimal leaf weights
w_left = -g_left / (h_left + 1.0)
w_right = -g_right / (h_right + 1.0)
print(f"Optimal left weight:  {w_left:.4f}")
print(f"Optimal right weight: {w_right:.4f}")`}
        />

        <h3>LightGBM — Often Faster Than XGBoost</h3>
        <CodeBlock
          language="python"
          title="lightgbm_example.py"
          code={`import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=10000, n_features=20,
                           n_informative=12, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,            # No limit — controlled by num_leaves
    num_leaves=31,           # LightGBM uses leaf-wise growth
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_samples=20,
    random_state=42,
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

print(f"Accuracy: {accuracy_score(y_te, model.predict(X_te)):.4f}")

# Key difference: LightGBM grows trees leaf-wise (best-first)
# XGBoost grows level-wise (breadth-first) by default
# Leaf-wise is faster and often more accurate, but can overfit
# if num_leaves is too large`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Start with LightGBM for speed</strong>: LightGBM is typically 5-10x faster than
            XGBoost on large datasets due to histogram-based splitting and leaf-wise growth. Use
            XGBoost when you need exact split finding or GPU training.
          </li>
          <li>
            <strong>Tuning priority order</strong>: (1) learning_rate + n_estimators with early stopping,
            (2) max_depth / num_leaves, (3) subsample + colsample_bytree, (4) regularization
            (lambda, alpha, gamma), (5) min_child_weight.
          </li>
          <li>
            <strong>Use CatBoost for categorical features</strong>: it handles categoricals natively
            with ordered target statistics, avoiding the need for one-hot encoding or label encoding.
          </li>
          <li>
            <strong>SHAP values for interpretability</strong>: XGBoost&apos;s built-in feature importance
            (gain, cover, frequency) can be misleading. Use SHAP for rigorous, theoretically grounded
            feature attribution.
          </li>
          <li>
            <strong>Monotonic constraints</strong>: XGBoost and LightGBM support monotonic constraints
            to enforce domain knowledge (e.g., higher income should never decrease credit score).
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Not using early stopping</strong>: this is the number one mistake. Without it,
            you will overfit. Set <code>early_stopping_rounds=50</code> and monitor a validation set.
          </li>
          <li>
            <strong>Setting learning_rate too high</strong>: a learning rate of 0.3 (the default) is
            almost always too aggressive. Use 0.01-0.1 with more trees.
          </li>
          <li>
            <strong>One-hot encoding high-cardinality categoricals</strong>: this creates sparse, high-dimensional
            data that trees handle poorly. Use label encoding, target encoding, or CatBoost.
          </li>
          <li>
            <strong>Confusing num_leaves with max_depth</strong>: in LightGBM, <code>num_leaves</code> controls
            complexity more directly than <code>max_depth</code>. A tree of depth 10 can have up to
            1024 leaves — set <code>num_leaves</code> lower than <InlineMath math="2^{\text{max\_depth}}" />.
          </li>
          <li>
            <strong>Ignoring class imbalance</strong>: use <code>scale_pos_weight</code> in XGBoost
            or <code>is_unbalance=True</code> in LightGBM. Or better yet, use appropriate metrics
            (AUC-PR, F1) instead of accuracy.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Walk me through how XGBoost decides where to split a node.
          Why does it use second-order gradients? What role does regularization play?
        </p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Objective setup</strong>: XGBoost takes a second-order Taylor expansion of the loss
            function around the current predictions, yielding gradient (<InlineMath math="g_i" />) and
            Hessian (<InlineMath math="h_i" />) statistics for each sample.
          </li>
          <li>
            <strong>Split evaluation</strong>: for each candidate split, XGBoost sums the gradients and
            Hessians on each side (<InlineMath math="G_L, H_L, G_R, H_R" />) and computes the gain
            formula. This gives the exact improvement in the regularized objective.
          </li>
          <li>
            <strong>Why second-order</strong>: using only first-order gradients (like vanilla gradient
            boosting) requires line search to find the optimal step size. The Hessian provides curvature
            information, allowing XGBoost to compute the <strong>optimal leaf weight in closed form</strong>: <InlineMath math="w^* = -G/(H + \lambda)" />.
            This is analogous to Newton&apos;s method vs. gradient descent.
          </li>
          <li>
            <strong>Regularization</strong>: <InlineMath math="\lambda" /> (L2 on leaf weights) shrinks
            predictions toward zero and appears in the denominator of the gain formula, reducing the
            impact of small groups. <InlineMath math="\gamma" /> (leaf count penalty) is subtracted from
            the gain, acting as a pre-pruning threshold that prevents splits that don&apos;t improve
            the objective enough.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Chen &amp; Guestrin (2016) &quot;XGBoost: A Scalable Tree Boosting System&quot;</strong> — The XGBoost paper. Read sections 2-3 for the math, section 4 for system design.</li>
          <li><strong>Ke et al. (2017) &quot;LightGBM: A Highly Efficient Gradient Boosting Decision Tree&quot;</strong> — Histogram-based splitting, leaf-wise growth, GOSS, EFB.</li>
          <li><strong>Prokhorenkova et al. (2018) &quot;CatBoost: unbiased boosting with categorical features&quot;</strong> — Ordered boosting to prevent target leakage.</li>
          <li><strong>Lundberg &amp; Lee (2017) &quot;A Unified Approach to Interpreting Model Predictions&quot;</strong> — SHAP values, with fast TreeSHAP for tree-based models.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
