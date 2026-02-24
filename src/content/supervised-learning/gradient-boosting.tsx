"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function GradientBoosting() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Random forests reduce <em>variance</em> by averaging many independent trees. <strong>Gradient boosting</strong> takes the opposite approach: it reduces <em>bias</em> by building trees <em>sequentially</em>, where each new tree focuses on correcting the mistakes of all previous trees combined.
        </p>
        <p>
          Here&apos;s the intuition: you fit a tree and look at the residuals (errors). Some points are way off. You fit a <em>second</em> tree specifically to predict those residuals. Now your combined model is the first tree&apos;s prediction plus the second tree&apos;s correction. Still some errors left? Fit a third tree to the remaining residuals. Keep going for hundreds of rounds, each time making a small correction.
        </p>
        <p>
          The word &quot;gradient&quot; comes from the fact that the residuals are actually the <strong>negative gradient</strong> of the loss function. For squared error, the negative gradient is literally the residual <InlineMath math="y_i - \hat{y}_i" />. For other losses (log loss, Huber loss), the &quot;residuals&quot; are generalized gradients. This is what makes gradient boosting so flexible — it works with any differentiable loss function.
        </p>
        <p>
          Today, three implementations dominate: <strong>XGBoost</strong> (2016, regularized and optimized), <strong>LightGBM</strong> (2017, histogram-based for speed), and <strong>CatBoost</strong> (2018, handles categoricals natively). For structured/tabular data, gradient boosting is the reigning champion and wins the majority of Kaggle competitions.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Additive Model</h3>
        <p>The ensemble prediction after <InlineMath math="M" /> rounds:</p>
        <BlockMath math="\hat{y}_i = \sum_{m=1}^{M} \eta \cdot f_m(\mathbf{x}_i)" />
        <p>where <InlineMath math="\eta" /> is the learning rate and each <InlineMath math="f_m" /> is a small tree.</p>

        <h3>The Gradient Boosting Algorithm</h3>
        <p>Initialize with a constant: <InlineMath math="F_0(x) = \arg\min_c \sum L(y_i, c)" /></p>
        <p>For each round <InlineMath math="m = 1, \dots, M" />:</p>
        <BlockMath math="r_{im} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)} \quad \text{(pseudo-residuals)}" />
        <BlockMath math="f_m = \text{fit a tree to } \{(x_i, r_{im})\}" />
        <BlockMath math="F_m(x) = F_{m-1}(x) + \eta \cdot f_m(x)" />

        <h3>Common Loss Functions</h3>
        <p><strong>Regression (MSE):</strong></p>
        <BlockMath math="L = \frac{1}{2}(y - \hat{y})^2 \quad \Rightarrow \quad r_i = y_i - \hat{y}_i" />
        <p><strong>Classification (Log Loss):</strong></p>
        <BlockMath math="L = -[y\log p + (1-y)\log(1-p)] \quad \Rightarrow \quad r_i = y_i - p_i" />

        <h3>XGBoost&apos;s Regularized Objective</h3>
        <p>XGBoost adds explicit regularization to the tree structure:</p>
        <BlockMath math="\mathcal{L} = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{m=1}^{M}\left[\gamma T_m + \frac{1}{2}\lambda\sum_{j=1}^{T_m} w_{mj}^2\right]" />
        <p>
          where <InlineMath math="T_m" /> is the number of leaves in tree <InlineMath math="m" />, <InlineMath math="w_{mj}" /> are leaf weights, <InlineMath math="\gamma" /> penalizes tree complexity, and <InlineMath math="\lambda" /> is L2 regularization on leaf values.
        </p>
        <p>XGBoost also uses a second-order Taylor approximation of the loss for faster, more accurate splits:</p>
        <BlockMath math="\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma" />
        <p>where <InlineMath math="G = \sum g_i" /> (sum of first derivatives) and <InlineMath math="H = \sum h_i" /> (sum of second derivatives).</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>From Scratch — Gradient Boosting for Regression</h3>
        <CodeBlock
          language="python"
          title="gradient_boosting_scratch.py"
          code={`import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = None

    def fit(self, X, y):
        # Initialize with mean
        self.init_pred = np.mean(y)
        F = np.full(len(y), self.init_pred)

        for _ in range(self.n_estimators):
            # Pseudo-residuals (negative gradient of MSE)
            residuals = y - F

            # Fit a small tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            F += self.lr * tree.predict(X)

        return self

    def predict(self, X):
        F = np.full(len(X), self.init_pred)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return F

# Example
np.random.seed(42)
X = np.sort(np.random.uniform(0, 10, 500)).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.normal(0, 0.2, 500)

gb = GradientBoostingScratch(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X, y)
y_pred = gb.predict(X)
mse = np.mean((y - y_pred) ** 2)
print(f"MSE: {mse:.4f}")`}
        />

        <h3>XGBoost, LightGBM, and CatBoost</h3>
        <CodeBlock
          language="python"
          title="boosting_frameworks.py"
          code={`import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=5000, n_features=20,
                           n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- XGBoost ---
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,            # row sampling
    colsample_bytree=0.8,     # feature sampling
    reg_lambda=1.0,           # L2 regularization
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=42,
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)
print(f"XGBoost:  {accuracy_score(y_test, xgb_model.predict(X_test)):.2%}")

# --- LightGBM ---
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=-1,             # no limit (leaf-wise growth)
    num_leaves=31,            # controls complexity
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)
lgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
print(f"LightGBM: {accuracy_score(y_test, lgb_model.predict(X_test)):.2%}")

# --- CatBoost ---
cb_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=0,
)
cb_model.fit(X_train, y_train, eval_set=(X_test, y_test))
print(f"CatBoost: {accuracy_score(y_test, cb_model.predict(X_test)):.2%}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Learning rate and n_estimators are coupled</strong>: Lower learning rate needs more trees. Use <InlineMath math="\eta = 0.01\text{-}0.1" /> with early stopping to find the right number of trees automatically.</li>
          <li><strong>Always use early stopping</strong>: Unlike random forests, boosting <em>can</em> overfit with too many rounds. Monitor validation loss and stop when it plateaus.</li>
          <li><strong>XGBoost vs LightGBM vs CatBoost</strong>: LightGBM is fastest (histogram-based, leaf-wise growth). CatBoost handles categorical features natively (no one-hot encoding needed). XGBoost is the most mature and widely deployed.</li>
          <li><strong>Subsample and colsample_bytree add stochasticity</strong>: Setting these to 0.7-0.9 reduces overfitting and speeds up training (stochastic gradient boosting).</li>
          <li><strong>Shallow trees are key</strong>: Use max_depth 3-8. Deep trees overfit quickly and negate the bias-reduction benefit of boosting.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Not using early stopping</strong>: Boosting will keep adding trees and overfit if you don&apos;t stop. Always monitor validation loss.</li>
          <li><strong>Setting learning rate too high</strong>: A learning rate of 1.0 means each tree fully corrects the previous errors, leading to overfitting. Use 0.01-0.1.</li>
          <li><strong>Growing trees too deep</strong>: Gradient boosting works best with <em>weak learners</em> (shallow trees, depth 3-6). Deep trees overfit and reduce the benefit of sequential correction.</li>
          <li><strong>Ignoring feature interactions</strong>: Boosting captures interactions up to the tree depth. If you need higher-order interactions, increase depth or engineer interaction features.</li>
          <li><strong>Leaking test data through early stopping</strong>: If your early stopping uses the test set, your test error is optimistic. Use a separate validation set or nested cross-validation.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Compare Random Forest and Gradient Boosting. When would you choose one over the other?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Random Forest</strong> reduces <strong>variance</strong> via averaging independent trees. <strong>Gradient Boosting</strong> reduces <strong>bias</strong> via sequential correction.</li>
          <li><strong>Overfitting</strong>: RF is very hard to overfit (more trees = more averaging). GB can overfit if you add too many rounds — it requires early stopping and careful tuning.</li>
          <li><strong>Tuning effort</strong>: RF works well out-of-the-box (main knob: n_estimators). GB has more hyperparameters (learning rate, max_depth, subsample, regularization).</li>
          <li><strong>Performance ceiling</strong>: GB typically achieves lower error on tabular data when properly tuned, because it directly attacks bias.</li>
          <li><strong>Training speed</strong>: RF is embarrassingly parallel (all trees at once). GB is inherently sequential (each tree depends on the previous), though LightGBM/XGBoost are highly optimized.</li>
          <li><strong>Choose RF</strong> when: you want a quick, robust baseline with minimal tuning, or you have noisy data. <strong>Choose GB</strong> when: you need the best possible accuracy on tabular data and have time to tune.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Friedman (2001) &quot;Greedy Function Approximation: A Gradient Boosting Machine&quot;</strong> — The foundational paper. Introduces the gradient view of boosting.</li>
          <li><strong>Chen &amp; Guestrin (2016) &quot;XGBoost&quot;</strong> — Regularized objective, approximate split finding, system design.</li>
          <li><strong>Ke et al. (2017) &quot;LightGBM&quot;</strong> — Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB).</li>
          <li><strong>Prokhorenkova et al. (2018) &quot;CatBoost&quot;</strong> — Ordered boosting to prevent target leakage with categorical features.</li>
          <li><strong>ESL Chapter 10</strong> — Boosting and additive trees.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
