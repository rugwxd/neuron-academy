"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function SHAP() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          When a model predicts that a loan application should be rejected, the applicant deserves an
          explanation. When a model flags a transaction as fraudulent, the analyst needs to know <em>why</em>.
          <strong> SHAP (SHapley Additive exPlanations)</strong> provides a principled, mathematically
          grounded way to explain individual predictions.
        </p>
        <p>
          The core idea comes from cooperative game theory. Imagine each feature is a &quot;player&quot;
          in a game where the &quot;payout&quot; is the model&apos;s prediction. SHAP assigns each feature
          a <strong>Shapley value</strong> — its fair contribution to the prediction, accounting for
          interactions with all other features. A feature that consistently pushes the prediction higher
          gets a positive SHAP value; one that pulls it lower gets a negative value.
        </p>
        <p>
          What makes SHAP special is that it is the <strong>only method</strong> that satisfies three
          desirable properties simultaneously: <em>local accuracy</em> (SHAP values sum to the
          model&apos;s output), <em>missingness</em> (absent features get zero attribution), and
          <em>consistency</em> (if a feature&apos;s contribution increases, its SHAP value never decreases).
          No other explanation method (LIME, permutation importance, etc.) satisfies all three.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Shapley Value Formula</h3>
        <p>
          For a model <InlineMath math="f" /> with feature set <InlineMath math="N = \{1, 2, \ldots, p\}" />,
          the Shapley value of feature <InlineMath math="i" /> is:
        </p>
        <BlockMath math="\phi_i(f) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\;(|N| - |S| - 1)!}{|N|!} \left[ f(S \cup \{i\}) - f(S) \right]" />
        <p>
          where <InlineMath math="f(S)" /> is the model&apos;s expected prediction when only the features
          in <InlineMath math="S" /> are known (features not in <InlineMath math="S" /> are marginalized out).
        </p>

        <h3>Intuition Behind the Formula</h3>
        <p>
          The Shapley value considers <strong>every possible ordering</strong> of features. For each ordering,
          it computes how much the prediction changes when feature <InlineMath math="i" /> is added. The
          Shapley value is the average marginal contribution across all orderings:
        </p>
        <BlockMath math="\phi_i = \frac{1}{|N|!} \sum_{\pi \in \Pi(N)} \left[ f(S_\pi^i \cup \{i\}) - f(S_\pi^i) \right]" />
        <p>
          where <InlineMath math="S_\pi^i" /> is the set of features that come before <InlineMath math="i" /> in
          permutation <InlineMath math="\pi" />.
        </p>

        <h3>SHAP Properties (Axioms)</h3>
        <p><strong>1. Local Accuracy (Efficiency):</strong></p>
        <BlockMath math="f(x) = \phi_0 + \sum_{i=1}^{p} \phi_i" />
        <p>
          where <InlineMath math="\phi_0 = E[f(X)]" /> is the base value (average prediction). The SHAP
          values exactly decompose the prediction.
        </p>

        <p><strong>2. Missingness:</strong> If feature <InlineMath math="i" /> is absent (does not change the
          output regardless of coalition), then <InlineMath math="\phi_i = 0" />.</p>

        <p><strong>3. Consistency:</strong> If a model changes so that feature <InlineMath math="i" />&apos;s
          marginal contribution increases (or stays the same) for every coalition, then <InlineMath math="\phi_i" /> does not decrease.</p>

        <h3>Computational Complexity</h3>
        <p>
          The exact Shapley value requires evaluating <InlineMath math="2^p" /> coalitions, which is
          exponential in the number of features. For <InlineMath math="p = 20" /> features, that is
          over 1 million coalitions. Approximation algorithms are essential:
        </p>
        <ul>
          <li><strong>KernelSHAP</strong>: Weighted linear regression approximation, works for any model — <InlineMath math="O(2^p)" /> worst case but often converges faster.</li>
          <li><strong>TreeSHAP</strong>: Exact computation for tree-based models in <InlineMath math="O(TLD^2)" /> where <InlineMath math="T" /> = number of trees, <InlineMath math="L" /> = max leaves, <InlineMath math="D" /> = max depth.</li>
          <li><strong>DeepSHAP</strong>: Backpropagation-based approximation for neural networks.</li>
        </ul>
      </TopicSection>

      <TopicSection type="code">
        <h3>SHAP for Tree-Based Models</h3>
        <CodeBlock
          language="python"
          title="shap_trees.py"
          code={`import shap
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# ---- TreeSHAP (exact and fast for tree models) ----
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# shap_values shape: (n_samples, n_features)
# Each value = contribution of that feature to that prediction

# Base value (average prediction in log-odds for classification)
print(f"Base value: {explainer.expected_value:.4f}")

# Explain a single prediction
i = 0
print(f"\\nPrediction for sample {i}:")
print(f"  Model output (log-odds): {model.predict(X_test.iloc[[i]], output_margin=True)[0]:.4f}")
print(f"  Base value + sum(SHAP) = {explainer.expected_value + shap_values[i].sum():.4f}")
print(f"\\nTop contributing features:")
feature_effects = list(zip(X_test.columns, shap_values[i], X_test.iloc[i]))
feature_effects.sort(key=lambda x: abs(x[1]), reverse=True)
for name, shap_val, feat_val in feature_effects[:5]:
    direction = "+" if shap_val > 0 else ""
    print(f"  {name} = {feat_val}: {direction}{shap_val:.4f}")

# ---- Visualization ----
# Force plot for single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Summary plot (global feature importance)
shap.summary_plot(shap_values, X_test)

# Dependence plot (how one feature affects predictions)
shap.dependence_plot("Age", shap_values, X_test)`}
        />

        <h3>SHAP for Deep Learning</h3>
        <CodeBlock
          language="python"
          title="shap_deep_learning.py"
          code={`import shap
import torch
import torch.nn as nn
import numpy as np

# Define a simple neural network
class TabularNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

model = TabularNet(X_train.shape[1])
# ... (training code omitted for brevity)

# ---- DeepSHAP ----
# Use training data as background distribution
background = torch.tensor(X_train.values[:100], dtype=torch.float32)
test_data = torch.tensor(X_test.values[:50], dtype=torch.float32)

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_data)

# ---- KernelSHAP (model-agnostic, works for ANY model) ----
def model_predict(X_array):
    """Wrapper: numpy array -> model predictions."""
    with torch.no_grad():
        tensor = torch.tensor(X_array, dtype=torch.float32)
        return model(tensor).numpy().flatten()

# Use k-means to summarize background data (faster than using all training data)
background_summary = shap.kmeans(X_train.values, 50)
explainer = shap.KernelExplainer(model_predict, background_summary)

# Compute SHAP values (slower but model-agnostic)
shap_values = explainer.shap_values(X_test.values[:20], nsamples=500)

shap.summary_plot(shap_values, X_test.iloc[:20])`}
        />

        <h3>Shapley Values from Scratch</h3>
        <CodeBlock
          language="python"
          title="shapley_scratch.py"
          code={`import numpy as np
from itertools import combinations
from math import factorial

def exact_shapley(model_fn, x, X_background, feature_names):
    """
    Compute exact Shapley values (exponential complexity).
    Only practical for small number of features (< 15).

    Args:
        model_fn: callable, takes array of shape (n, p) -> (n,)
        x: single instance to explain, shape (p,)
        X_background: background dataset, shape (n_bg, p)
        feature_names: list of feature names
    """
    p = len(x)
    N = set(range(p))
    shapley_values = np.zeros(p)

    for i in range(p):
        phi_i = 0.0
        others = N - {i}

        for size in range(0, p):
            # All coalitions of this size (without feature i)
            for S in combinations(others, size):
                S = set(S)

                # f(S union {i}) - f(S)
                # Marginalize out features NOT in the coalition
                def expected_prediction(coalition):
                    """E[f(x) | x_coalition = observed values]."""
                    X_eval = X_background.copy()
                    for j in coalition:
                        X_eval[:, j] = x[j]
                    return model_fn(X_eval).mean()

                marginal = (
                    expected_prediction(S | {i}) - expected_prediction(S)
                )

                # Shapley weight
                weight = (
                    factorial(len(S)) * factorial(p - len(S) - 1)
                    / factorial(p)
                )
                phi_i += weight * marginal

        shapley_values[i] = phi_i

    # Verify: base_value + sum(shapley) should equal f(x)
    base_value = model_fn(X_background).mean()
    print(f"Base value: {base_value:.4f}")
    print(f"f(x): {model_fn(x.reshape(1, -1))[0]:.4f}")
    print(f"Base + sum(SHAP): {base_value + shapley_values.sum():.4f}")

    return dict(zip(feature_names, shapley_values))`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Use TreeSHAP for tree models</strong>: It is exact and fast. For XGBoost, LightGBM, and random forests, there is no reason to use the slower KernelSHAP.</li>
          <li><strong>KernelSHAP for everything else</strong>: It works on any model (neural networks, SVMs, ensembles) but is slower. Use <code>shap.kmeans()</code> to summarize the background data and keep <code>nsamples</code> reasonable.</li>
          <li><strong>SHAP summary plot is the best global importance plot</strong>: Unlike a bar chart, it shows feature importance AND direction of effect AND the distribution of effects. Always use it.</li>
          <li><strong>Watch the base value</strong>: SHAP values are relative to the base value (average prediction). A SHAP value of +0.3 means &quot;this feature pushes the prediction 0.3 above average.&quot;</li>
          <li><strong>For classification, use log-odds</strong>: SHAP values in probability space are harder to interpret because probabilities are bounded. Work in log-odds space where effects are additive.</li>
          <li><strong>SHAP interaction values</strong>: <code>shap_interaction = explainer.shap_interaction_values(X)</code> reveals pairwise feature interactions. Expensive but very insightful.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Confusing SHAP importance with feature importance</strong>: Tree feature importance (gain) measures how much a feature reduces impurity across all splits. SHAP importance measures the average impact on predictions. They can disagree, and SHAP is more reliable.</li>
          <li><strong>Using too small a background dataset</strong>: The background data approximates the marginal distribution. Too few samples (e.g., 10) gives noisy SHAP values. Use at least 100-500 background samples.</li>
          <li><strong>Interpreting SHAP values as causal</strong>: SHAP values are <em>attributions</em>, not causal effects. A feature with high SHAP value may be correlated with the true cause but changing it would not change the outcome.</li>
          <li><strong>Ignoring feature correlation</strong>: When features are correlated, marginalizing by independently sampling features (KernelSHAP default) creates unrealistic data points. Use <code>shap.TreeExplainer(model, feature_perturbation=&quot;interventional&quot;)</code> for tree models to handle this.</li>
          <li><strong>Averaging SHAP values to get global importance</strong>: Use <code>mean(|SHAP|)</code> (mean absolute value), not <code>mean(SHAP)</code>. A feature that is +5 for half the data and -5 for the other half has zero mean SHAP but is extremely important.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain the difference between SHAP and LIME. When would you use each?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>LIME (Local Interpretable Model-agnostic Explanations)</strong>: Fits a local linear model around the prediction by perturbing the input and observing how the prediction changes. Fast but the explanations depend on the perturbation strategy and kernel width — different runs can give different explanations.</li>
          <li><strong>SHAP</strong>: Computes exact Shapley values from game theory. The explanations are unique (the Shapley value is the only attribution satisfying efficiency, symmetry, dummy, and additivity). They are consistent and additive (they sum to the prediction).</li>
          <li><strong>When to use LIME</strong>: Quick exploration, very large models where SHAP is too slow, or when you need explanations for text/image data (LIME handles superpixels well).</li>
          <li><strong>When to use SHAP</strong>: When you need rigorous, reproducible explanations — regulatory compliance (banking, healthcare), model debugging, or whenever TreeSHAP is available (tree models). SHAP is preferred in most production settings.</li>
          <li><strong>Key difference</strong>: LIME explanations can be inconsistent (feature A appears important in one explanation but not another). SHAP provides a unique, theoretically grounded attribution for each feature.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Lundberg &amp; Lee (2017) &quot;A Unified Approach to Interpreting Model Predictions&quot;</strong> — The original SHAP paper, introducing the connection between Shapley values and model explanations.</li>
          <li><strong>Christoph Molnar &quot;Interpretable Machine Learning&quot; (free online book)</strong> — Excellent chapters on SHAP, LIME, and other interpretability methods.</li>
          <li><strong>SHAP GitHub repository</strong> — Documentation, tutorials, and the shap Python package.</li>
          <li><strong>Lundberg et al. (2020) &quot;From Local Explanations to Global Understanding with Explainable AI for Trees&quot;</strong> — TreeSHAP algorithm and global explanation patterns.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
