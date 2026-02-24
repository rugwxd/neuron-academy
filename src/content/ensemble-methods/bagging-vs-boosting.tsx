"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function BaggingVsBoosting() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A single decision tree is like asking <strong>one expert</strong> for advice — they might be brilliant
          but they&apos;re also biased by their experience. Ensemble methods are the idea that a <strong>committee
          of experts</strong> will outperform any individual, as long as they make different kinds of mistakes.
        </p>
        <p>
          There are two fundamentally different strategies for building that committee:
        </p>
        <ul>
          <li>
            <strong>Bagging (Bootstrap Aggregating)</strong>: Train many models <em>independently</em> on
            different random samples of the data, then <strong>average</strong> their predictions. Each model
            is a &quot;strong learner&quot; — fully grown trees that overfit. Averaging cancels out the
            overfitting. <strong>Random Forest</strong> is the most famous example.
          </li>
          <li>
            <strong>Boosting</strong>: Train models <em>sequentially</em>, where each new model focuses on
            correcting the <strong>mistakes</strong> of the previous ones. Each model is a &quot;weak learner&quot;
            — a shallow tree (stump or depth 3-6). Together they form a powerful ensemble. <strong>XGBoost</strong>,
            <strong> LightGBM</strong>, and <strong>AdaBoost</strong> are the big names.
          </li>
        </ul>
        <p>
          The key difference: bagging reduces <strong>variance</strong> (averaging smooths out noise),
          while boosting reduces <strong>bias</strong> (iteratively fixing systematic errors). In practice,
          boosting usually wins on tabular data, but bagging is more forgiving — it&apos;s harder to overfit
          a Random Forest than to overtune XGBoost.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Bagging: Variance Reduction Through Averaging</h3>
        <p>
          If you have <InlineMath math="B" /> independent models, each with variance <InlineMath math="\sigma^2" />,
          the variance of their average is:
        </p>
        <BlockMath math="\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B} f_b(x)\right) = \frac{\sigma^2}{B}" />
        <p>
          In practice, the models aren&apos;t truly independent (they&apos;re trained on overlapping bootstrap samples),
          so the actual reduction depends on the correlation <InlineMath math="\rho" /> between trees:
        </p>
        <BlockMath math="\text{Var}_{\text{RF}} = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2" />
        <p>
          Random Forest reduces <InlineMath math="\rho" /> by randomly subsetting features at each split
          (typically <InlineMath math="\sqrt{d}" /> for classification, <InlineMath math="d/3" /> for regression).
        </p>

        <h3>AdaBoost: Reweighting Misclassified Samples</h3>
        <p>At each round <InlineMath math="t" />:</p>
        <ol>
          <li>Train weak learner <InlineMath math="h_t" /> on weighted data</li>
          <li>Compute weighted error: <BlockMath math="\epsilon_t = \frac{\sum_{i: h_t(x_i) \neq y_i} w_i}{\sum_i w_i}" /></li>
          <li>Compute learner weight: <BlockMath math="\alpha_t = \frac{1}{2}\ln\frac{1 - \epsilon_t}{\epsilon_t}" /></li>
          <li>Update sample weights: <BlockMath math="w_i \leftarrow w_i \cdot \exp(-\alpha_t y_i h_t(x_i))" /></li>
        </ol>
        <p>Final prediction:</p>
        <BlockMath math="H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)" />

        <h3>Gradient Boosting: Fitting Residuals</h3>
        <p>
          Instead of reweighting samples, gradient boosting fits each new tree to the <strong>negative gradient</strong> (pseudo-residuals) of the loss:
        </p>
        <BlockMath math="F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)" />
        <p>where <InlineMath math="h_m" /> is trained to predict:</p>
        <BlockMath math="r_{im} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}" />
        <p>
          For MSE loss, the pseudo-residuals are just the actual residuals: <InlineMath math="r_i = y_i - F_{m-1}(x_i)" />.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Bagging vs Boosting — Head to Head</h3>
        <CodeBlock
          language="python"
          title="bagging_vs_boosting.py"
          code={`import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

# Create a moderately complex dataset
X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=10,
    n_redundant=5, random_state=42
)

models = {
    "Single Tree (depth=None)": DecisionTreeClassifier(random_state=42),
    "Bagging (50 trees)": BaggingClassifier(
        n_estimators=50, random_state=42
    ),
    "Random Forest (50 trees)": RandomForestClassifier(
        n_estimators=50, random_state=42
    ),
    "AdaBoost (50 stumps)": AdaBoostClassifier(
        n_estimators=50, random_state=42
    ),
    "Gradient Boosting (50 trees)": GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        random_state=42,
    ),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"{name:35s}  {scores.mean():.4f} +/- {scores.std():.4f}")

# Typical output:
# Single Tree (depth=None)             0.8790 +/- 0.0180
# Bagging (50 trees)                   0.9265 +/- 0.0120
# Random Forest (50 trees)             0.9340 +/- 0.0098
# AdaBoost (50 stumps)                 0.9055 +/- 0.0105
# Gradient Boosting (50 trees)         0.9420 +/- 0.0085`}
        />

        <h3>Gradient Boosting from Scratch</h3>
        <CodeBlock
          language="python"
          title="gradient_boosting_scratch.py"
          code={`import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingScratch:
    """Simplified gradient boosting for regression (MSE loss)."""
    def __init__(self, n_estimators=100, lr=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = None

    def fit(self, X, y):
        # Initialize with the mean
        self.init_pred = np.mean(y)
        current_pred = np.full(len(y), self.init_pred)

        for _ in range(self.n_estimators):
            # Pseudo-residuals (negative gradient of MSE)
            residuals = y - current_pred

            # Fit a tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions with learning rate
            current_pred += self.lr * tree.predict(X)

        return self

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred

# Test it
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
model = GradientBoostingScratch(n_estimators=100, lr=0.1, max_depth=3)
model.fit(X, y)
print(f"MSE: {mean_squared_error(y, model.predict(X)):.2f}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Use Random Forest as a baseline</strong>: it&apos;s hard to screw up. Out-of-the-box
            with 500 trees and default hyperparameters, it&apos;s competitive on most tabular problems.
          </li>
          <li>
            <strong>Gradient boosting wins competitions</strong>: XGBoost, LightGBM, or CatBoost dominate
            Kaggle and real-world tabular tasks. But they require careful hyperparameter tuning
            (learning rate, max depth, number of rounds, regularization).
          </li>
          <li>
            <strong>Learning rate and n_estimators are coupled</strong>: lower learning rate needs more trees.
            The common recipe is to set a low learning rate (0.01-0.1) and use early stopping to find the
            optimal number of rounds.
          </li>
          <li>
            <strong>Feature importance comes for free</strong>: both Random Forest and gradient boosting
            provide feature importance scores based on how often (and how effectively) each feature is used
            for splitting.
          </li>
          <li>
            <strong>OOB score is a free validation set</strong>: in bagging, ~37% of samples are left out of
            each bootstrap sample. Their predictions form the out-of-bag (OOB) estimate — no need for a
            separate validation split.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Not using early stopping with boosting</strong>: without it, gradient boosting will
            overfit as you add more trees. Always monitor validation loss and stop when it plateaus.
          </li>
          <li>
            <strong>Tuning n_estimators without fixing learning rate</strong>: these two parameters
            are tightly coupled. Set a learning rate first, then use early stopping for n_estimators.
          </li>
          <li>
            <strong>Thinking more trees always helps in Random Forest</strong>: variance keeps decreasing
            with more trees, but there are diminishing returns after ~300-500. It never hurts, but it
            wastes compute.
          </li>
          <li>
            <strong>Ignoring tree depth in boosting</strong>: gradient boosting uses shallow trees
            (depth 3-8) by design. Using deep trees defeats the purpose — you want many weak learners,
            not a few strong ones.
          </li>
          <li>
            <strong>Confusing bagging with boosting conceptually</strong>: bagging trains trees
            in parallel (independent), boosting trains them sequentially (dependent). This is a
            favorite interview question.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Explain the difference between bagging and boosting. When would
          you choose one over the other? Can boosting overfit?
        </p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Bagging</strong> trains <InlineMath math="B" /> models independently on bootstrap samples
            and averages them. It reduces <strong>variance</strong>. Random Forest adds feature subsampling
            to further decorrelate trees.
          </li>
          <li>
            <strong>Boosting</strong> trains models sequentially, each correcting the previous ensemble&apos;s
            errors. It reduces <strong>bias</strong>. The learning rate controls how much each tree contributes.
          </li>
          <li>
            <strong>When to choose</strong>:
            <ul>
              <li>Random Forest: when you want robust results with minimal tuning, when data is noisy, or when
              interpretability of feature importances matters.</li>
              <li>Gradient boosting: when you need maximum predictive performance on tabular data and can invest
              time in hyperparameter search.</li>
            </ul>
          </li>
          <li>
            <strong>Can boosting overfit?</strong> Yes. Unlike bagging (where adding more trees never hurts),
            boosting can overfit with too many rounds, too high a learning rate, or trees that are too deep.
            Early stopping on a validation set is essential.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Breiman (1996) &quot;Bagging Predictors&quot;</strong> — The original bagging paper.</li>
          <li><strong>Freund &amp; Schapire (1997) &quot;A Decision-Theoretic Generalization of Online Learning&quot;</strong> — The AdaBoost paper.</li>
          <li><strong>Friedman (2001) &quot;Greedy Function Approximation: A Gradient Boosting Machine&quot;</strong> — The gradient boosting framework.</li>
          <li><strong>ESL Chapters 8, 10, 15</strong> — Comprehensive treatment of ensemble methods.</li>
          <li><strong>Louppe (2014) &quot;Understanding Random Forests&quot;</strong> — PhD thesis with the clearest explanation of why Random Forests work.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
