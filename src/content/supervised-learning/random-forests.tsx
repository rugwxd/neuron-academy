"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function RandomForests() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A single decision tree is smart but unreliable — it memorizes quirks in your training data (high variance). A <strong>random forest</strong> fixes this with the &quot;wisdom of crowds&quot; approach: train hundreds of decision trees, each on a slightly different version of the data, and let them vote.
        </p>
        <p>
          The key insight is <strong>bagging</strong> (bootstrap aggregating). Each tree is trained on a random sample <em>with replacement</em> from the training set. This means each tree sees a different subset of the data, so they make different mistakes. When you average their predictions (regression) or take a majority vote (classification), the individual errors tend to cancel out.
        </p>
        <p>
          But there&apos;s a twist: if one feature is extremely powerful, every tree will split on it first, making all trees highly correlated. Averaging correlated trees doesn&apos;t reduce variance much. So random forests add a second layer of randomness: at each split, only a random subset of features is considered (typically <InlineMath math="\sqrt{d}" /> for classification, <InlineMath math="d/3" /> for regression). This <strong>decorrelates</strong> the trees, making the ensemble much more powerful than bagging alone.
        </p>
        <p>
          The result is a model that is hard to overfit, requires almost no hyperparameter tuning, handles missing values gracefully (in some implementations), and provides a built-in measure of feature importance. It&apos;s the default &quot;first thing to try&quot; for tabular data.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Bagging: Why Averaging Reduces Variance</h3>
        <p>If you have <InlineMath math="B" /> trees, each with variance <InlineMath math="\sigma^2" /> and pairwise correlation <InlineMath math="\rho" />:</p>
        <BlockMath math="\text{Var}\left[\frac{1}{B}\sum_{b=1}^{B} f_b(x)\right] = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2" />
        <p>
          As <InlineMath math="B \to \infty" />, the second term vanishes, but the first term <InlineMath math="\rho\sigma^2" /> stays. That&apos;s why decorrelating trees (reducing <InlineMath math="\rho" />) matters so much — it lowers the floor on variance.
        </p>

        <h3>Feature Subsampling</h3>
        <p>At each split, randomly sample <InlineMath math="m" /> features from <InlineMath math="d" /> total:</p>
        <BlockMath math="m = \lfloor\sqrt{d}\rfloor \quad \text{(classification)}, \qquad m = \lfloor d/3 \rfloor \quad \text{(regression)}" />

        <h3>Out-of-Bag (OOB) Error</h3>
        <p>Each bootstrap sample leaves out about 37% of the data (the OOB samples). For each data point, predict using only the trees that did <em>not</em> train on it:</p>
        <BlockMath math="P(\text{sample not in bootstrap}) = \left(1 - \frac{1}{n}\right)^n \xrightarrow{n \to \infty} \frac{1}{e} \approx 0.368" />
        <p>The OOB error is a free cross-validation estimate — no separate validation set needed.</p>

        <h3>Feature Importance (Mean Decrease in Impurity)</h3>
        <BlockMath math="\text{Importance}(j) = \frac{1}{B}\sum_{b=1}^{B}\sum_{t \in T_b} \frac{n_t}{n} \Delta G(t) \cdot \mathbf{1}[\text{feature}(t) = j]" />
        <p>where <InlineMath math="\Delta G(t)" /> is the impurity decrease at node <InlineMath math="t" />.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>From Scratch</h3>
        <CodeBlock
          language="python"
          title="random_forest_scratch.py"
          code={`import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForestScratch:
    def __init__(self, n_trees=100, max_depth=10, max_features="sqrt"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.oob_score_ = None

    def fit(self, X, y):
        n, d = X.shape
        oob_votes = np.zeros((n, len(np.unique(y))))
        oob_counts = np.zeros(n)

        for _ in range(self.n_trees):
            # Bootstrap sample
            idx = np.random.randint(0, n, size=n)
            oob_idx = np.setdiff1d(np.arange(n), np.unique(idx))

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

            # Accumulate OOB predictions
            if len(oob_idx) > 0:
                oob_probs = tree.predict_proba(X[oob_idx])
                oob_votes[oob_idx] += oob_probs
                oob_counts[oob_idx] += 1

        # Compute OOB score
        mask = oob_counts > 0
        oob_preds = np.argmax(oob_votes[mask], axis=1)
        self.oob_score_ = np.mean(oob_preds == y[mask])
        return self

    def predict(self, X):
        all_preds = np.array([t.predict(X) for t in self.trees])
        # Majority vote
        from scipy.stats import mode
        return mode(all_preds, axis=0, keepdims=False).mode

# Example
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    random_state=42
)

rf = RandomForestScratch(n_trees=100, max_depth=10)
rf.fit(X, y)
print(f"OOB accuracy: {rf.oob_score_:.2%}")`}
        />

        <h3>With scikit-learn</h3>
        <CodeBlock
          language="python"
          title="random_forest_sklearn.py"
          code={`from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=500,       # number of trees
    max_depth=None,         # grow full trees (let bagging handle variance)
    max_features="sqrt",    # random feature subset at each split
    min_samples_leaf=2,     # slight regularization
    oob_score=True,         # free validation estimate
    n_jobs=-1,              # parallelize across all cores
    random_state=42,
)
rf.fit(X_train, y_train)

print(f"OOB accuracy:  {rf.oob_score_:.2%}")
print(f"Test accuracy: {rf.score(X_test, y_test):.2%}")

# Feature importance
importances = rf.feature_importances_
top_5 = np.argsort(importances)[-5:][::-1]
for i in top_5:
    print(f"  Feature {i}: {importances[i]:.4f}")

print(classification_report(y_test, rf.predict(X_test)))`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>n_estimators: more is (almost) always better</strong>: Unlike boosting, random forests don&apos;t overfit with more trees. Use 500-1000 and stop when OOB error plateaus.</li>
          <li><strong>max_depth=None is fine</strong>: Let each tree grow fully. Bagging handles variance. Only limit depth for speed or memory.</li>
          <li><strong>Feature importance has caveats</strong>: MDI (default) is biased toward high-cardinality features. Use <strong>permutation importance</strong> (<code>sklearn.inspection.permutation_importance</code>) for a more reliable estimate.</li>
          <li><strong>No feature scaling needed</strong>: Like single trees, random forests are invariant to monotone transformations of features.</li>
          <li><strong>Embarrassingly parallel</strong>: Each tree is independent. Set <code>n_jobs=-1</code> to use all CPU cores.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Using random forests on very high-dimensional sparse data</strong>: For text (bag-of-words with 50K features), linear models or gradient boosting are typically better. Random feature subsets may miss the signal.</li>
          <li><strong>Treating MDI feature importance as ground truth</strong>: It favors continuous features and features with many categories. Always validate with permutation importance.</li>
          <li><strong>Expecting random forests to extrapolate</strong>: Trees can only predict values within the range seen during training. For time series with trends, they fail at forecasting beyond historical values.</li>
          <li><strong>Using too few trees</strong>: With 10 trees, variance is still high. Use at least 100, ideally 500+.</li>
          <li><strong>Ignoring class imbalance</strong>: Use <code>class_weight=&quot;balanced&quot;</code> or <code>class_weight=&quot;balanced_subsample&quot;</code> for imbalanced datasets.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain the difference between bagging and random forests. Why does adding feature subsampling on top of bagging help?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Bagging</strong> trains each tree on a bootstrap sample of the data but uses <em>all</em> features at each split. If there is one dominant feature, every bagged tree will split on it first, making the trees highly correlated.</li>
          <li><strong>Random forests</strong> add feature subsampling: at each split, only <InlineMath math="m \ll d" /> random features are considered. This forces trees to explore different feature subsets, <strong>decorrelating</strong> them.</li>
          <li>
            <strong>Why this matters</strong>: The variance of the ensemble average is <InlineMath math="\rho\sigma^2 + \frac{(1-\rho)}{B}\sigma^2" />. Reducing <InlineMath math="\rho" /> (correlation between trees) directly reduces ensemble variance. With bagging alone, <InlineMath math="\rho" /> can be high (e.g., 0.8); with random forests, it drops to 0.1-0.3.
          </li>
          <li><strong>The tradeoff</strong>: Each individual tree is slightly worse (higher bias, since it can&apos;t always pick the best feature), but the ensemble is much better because the variance reduction outweighs the bias increase.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Breiman (2001) &quot;Random Forests&quot;</strong> — The original paper. Introduces OOB, variable importance, and the connection to adaptive nearest neighbors.</li>
          <li><strong>ESL Chapter 15</strong> — Random Forests, with the variance formula and analysis of decorrelation.</li>
          <li><strong>Probst et al. (2019) &quot;Hyperparameters and Tuning Strategies for Random Forest&quot;</strong> — Comprehensive empirical study of which hyperparameters matter.</li>
          <li><strong>Strobl et al. (2007)</strong> — Why default variable importance can be biased, and conditional permutation alternatives.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
