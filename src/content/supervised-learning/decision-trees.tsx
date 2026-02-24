"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";
import DecisionTreeViz from "@/components/viz/DecisionTreeViz";

export default function DecisionTrees() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A decision tree is the machine learning equivalent of a game of <strong>20 questions</strong>.
          It asks a series of yes/no questions about your features, and each answer narrows down the prediction.
        </p>
        <p>
          &quot;Is the house bigger than 1500 sq ft?&quot; → Yes → &quot;Is it in a good school district?&quot; → Yes → Predicted price: $450K.
        </p>
        <p>
          The tree learns <em>which questions to ask</em> and <em>in what order</em> by finding the feature
          and threshold at each step that best separates the data. &quot;Best separates&quot; is measured by
          <strong> impurity</strong> — how mixed the classes are on each side of the split.
        </p>
        <p>
          Decision trees are powerful because they&apos;re <strong>interpretable</strong> (you can literally
          read the rules), handle nonlinear relationships, require no feature scaling, and work with both
          numerical and categorical data. Their weakness: they tend to overfit. That&apos;s why we use
          <em> random forests</em> and <em>gradient boosting</em> — ensembles of many trees.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Impurity Measures</h3>
        <p><strong>Gini Impurity</strong> (used by CART, scikit-learn default):</p>
        <BlockMath math="G = 1 - \sum_{k=1}^{K} p_k^2" />
        <p>
          where <InlineMath math="p_k" /> is the proportion of class <InlineMath math="k" /> in the node.
          Pure node → G = 0. Maximally impure (50/50 binary) → G = 0.5.
        </p>

        <p><strong>Entropy / Information Gain</strong> (used by ID3, C4.5):</p>
        <BlockMath math="H = -\sum_{k=1}^{K} p_k \log_2 p_k" />

        <h3>Split Selection</h3>
        <p>At each node, find the feature and threshold that maximizes the <strong>information gain</strong>:</p>
        <BlockMath math="\text{Gain} = H(\text{parent}) - \frac{n_L}{n} H(\text{left}) - \frac{n_R}{n} H(\text{right})" />
        <p>
          The algorithm greedily picks the split with the highest gain at each level.
        </p>

        <h3>Regression Trees</h3>
        <p>For regression, use <strong>variance reduction</strong> instead of impurity. Each leaf predicts the mean of its training points:</p>
        <BlockMath math="\text{MSE}_{node} = \frac{1}{n}\sum_{i \in \text{node}} (y_i - \bar{y}_{node})^2" />
      </TopicSection>

      <TopicSection type="code">
        <h3>From Scratch — Simplified Decision Tree</h3>
        <CodeBlock
          language="python"
          title="decision_tree_scratch.py"
          code={`import numpy as np

def gini(y):
    """Gini impurity for binary classification."""
    if len(y) == 0:
        return 0
    p = np.mean(y)
    return 1 - p**2 - (1 - p)**2

def best_split(X, y):
    """Find the best feature and threshold to split on."""
    best_gain = -1
    best_feature, best_threshold = None, None
    parent_gini = gini(y)

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            left_mask = X[:, feature] <= t
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            # Weighted Gini after split
            n = len(y)
            weighted = (
                left_mask.sum() / n * gini(y[left_mask])
                + right_mask.sum() / n * gini(y[right_mask])
            )
            gain = parent_gini - weighted

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold, best_gain

def build_tree(X, y, depth=0, max_depth=5):
    """Recursively build a decision tree."""
    # Stopping conditions
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 2:
        return {"prediction": int(np.round(np.mean(y)))}

    feature, threshold, gain = best_split(X, y)
    if gain <= 0:
        return {"prediction": int(np.round(np.mean(y)))}

    left_mask = X[:, feature] <= threshold
    return {
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(X[left_mask], y[left_mask], depth + 1, max_depth),
        "right": build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth),
    }

def predict_one(tree, x):
    if "prediction" in tree:
        return tree["prediction"]
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_one(tree["left"], x)
    return predict_one(tree["right"], x)

# Example
X = np.random.randn(200, 2)
y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
tree = build_tree(X, y, max_depth=4)
preds = [predict_one(tree, x) for x in X]
print(f"Accuracy: {np.mean(preds == y):.2%}")`}
        />

        <h3>With scikit-learn</h3>
        <CodeBlock
          language="python"
          title="decision_tree_sklearn.py"
          code={`from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train with controlled depth to avoid overfitting
clf = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=5,
    random_state=42,
)
clf.fit(X_train, y_train)

print(f"Train accuracy: {clf.score(X_train, y_train):.2%}")
print(f"Test accuracy: {clf.score(X_test, y_test):.2%}")

# Print the tree rules (interpretability!)
print(export_text(clf, feature_names=["x1", "x2"]))`}
        />
      </TopicSection>

      <TopicSection type="see-it">
        <p className="mb-4">
          Adjust the <strong>max depth</strong> slider to see how the decision boundary changes.
          At depth 1, the tree makes a single split. At higher depths, it creates increasingly complex
          (potentially overfit) boundaries. The dashed lines show where splits occur.
        </p>
        <DecisionTreeViz />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Single trees overfit</strong> — always use them inside ensembles (Random Forest or Gradient Boosting) for production.</li>
          <li><strong>Control depth for interpretability</strong> — a depth-3 tree has at most 8 leaves, which is very readable. Use this for explaining to stakeholders.</li>
          <li><strong>No feature scaling needed</strong> — trees split on thresholds, so scale doesn&apos;t matter.</li>
          <li><strong>Handle missing values</strong> — XGBoost and LightGBM have built-in handling. scikit-learn trees require imputation.</li>
          <li><strong>Feature importance is a side effect</strong> — trees naturally rank features by how much they reduce impurity.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Not setting max_depth</strong> — an unrestricted tree will memorize the training data (100% train accuracy, poor generalization).</li>
          <li><strong>Using a single tree in production</strong> — use Random Forest (variance reduction) or Gradient Boosting (bias reduction).</li>
          <li><strong>Interpreting feature importance as causal</strong> — it just means the tree used that feature for splitting, not that the feature causes the outcome.</li>
          <li><strong>Assuming axis-aligned boundaries are sufficient</strong> — trees can only split perpendicular to axes. For diagonal boundaries, you need many splits (or feature engineering).</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> What is the time complexity of training a decision tree? How does Random Forest improve on a single tree?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Training complexity</strong>: <InlineMath math="O(n \cdot d \cdot \log n)" /> at each node (n samples, d features, sorting takes log n). Total: <InlineMath math="O(n \cdot d \cdot \log^2 n)" /> for a balanced tree.</li>
          <li><strong>Random Forest improvement</strong>: Trains many trees (typically 100-500) on bootstrap samples (bagging), and each tree only considers a random subset of <InlineMath math="\sqrt{d}" /> features at each split.
            <ul>
              <li>Bagging reduces <strong>variance</strong> (averaging many noisy, uncorrelated trees).</li>
              <li>Feature subsampling decorrelates the trees (if one feature is dominant, not every tree will find it first).</li>
              <li>Result: much lower variance, only slightly higher bias → better generalization.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Breiman (1984)</strong> — CART: The original decision tree paper.</li>
          <li><strong>ESL Chapter 9</strong> — Additive Models, Trees, and Related Methods.</li>
          <li><strong>scikit-learn Decision Tree docs</strong> — Visualization, feature importance, pruning options.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
