"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function LogisticRegression() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Logistic regression answers the question: <strong>what&apos;s the probability that this data point belongs to class 1?</strong> Despite its name, it&apos;s a <em>classification</em> algorithm, not a regression one. The &quot;regression&quot; part refers to the fact that it fits a linear model under the hood — but then squashes the output through a sigmoid function to produce a probability between 0 and 1.
        </p>
        <p>
          Think of it this way: linear regression predicts any number from <InlineMath math="-\infty" /> to <InlineMath math="+\infty" />. That&apos;s useless for classification — you need a number between 0 and 1. The <strong>sigmoid function</strong> <InlineMath math="\sigma(z) = \frac{1}{1+e^{-z}}" /> does exactly this: it takes any real number and maps it to (0, 1). If the output is above 0.5, predict class 1; otherwise, predict class 0.
        </p>
        <p>
          The <strong>decision boundary</strong> is the surface where the predicted probability equals 0.5. For logistic regression, this boundary is always a straight line (or hyperplane in higher dimensions). This means logistic regression works well when classes are roughly linearly separable, and struggles when the true boundary is curved. That&apos;s when you reach for SVMs with kernels, decision trees, or neural networks.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>The Model</h3>
        <p>Compute a linear combination, then apply the sigmoid:</p>
        <BlockMath math="z = \mathbf{w}^T \mathbf{x} + b" />
        <BlockMath math="P(y=1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}" />

        <h3>Log-Odds (Logit)</h3>
        <p>The inverse of the sigmoid gives the <strong>logit</strong> — the log-odds of the positive class:</p>
        <BlockMath math="\log \frac{P(y=1)}{P(y=0)} = \mathbf{w}^T \mathbf{x} + b" />
        <p>
          This is why it&apos;s called <em>logistic</em> regression: we&apos;re modeling the log-odds as a linear function of the features.
        </p>

        <h3>Loss Function: Binary Cross-Entropy</h3>
        <p>We maximize the likelihood, or equivalently minimize the <strong>negative log-likelihood</strong>:</p>
        <BlockMath math="L(\mathbf{w}, b) = -\frac{1}{n}\sum_{i=1}^{n}\left[ y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i) \right]" />
        <p>
          There is no closed-form solution. We optimize using <strong>gradient descent</strong> (or more commonly Newton&apos;s method / L-BFGS, which converges faster since the loss is convex).
        </p>

        <h3>Gradient</h3>
        <BlockMath math="\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n} X^T (\hat{\mathbf{p}} - \mathbf{y})" />
        <p>
          Remarkably, this has the same form as the linear regression gradient — the difference is that <InlineMath math="\hat{\mathbf{p}}" /> passes through the sigmoid.
        </p>

        <h3>Multiclass Extension: Softmax</h3>
        <p>For <InlineMath math="K" /> classes, replace the sigmoid with softmax:</p>
        <BlockMath math="P(y=k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}" />
        <p>The loss becomes <strong>categorical cross-entropy</strong>.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>From Scratch</h3>
        <CodeBlock
          language="python"
          title="logistic_regression_scratch.py"
          code={`import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.losses = []

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p_hat = self.sigmoid(z)

            # Binary cross-entropy loss
            loss = -np.mean(y * np.log(p_hat + 1e-12)
                          + (1 - y) * np.log(1 - p_hat + 1e-12))
            self.losses.append(loss)

            # Gradients
            error = p_hat - y
            self.w -= self.lr * (X.T @ error) / n
            self.b -= self.lr * error.mean()
        return self

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Example: classify two Gaussian blobs
np.random.seed(42)
X = np.vstack([np.random.randn(100, 2) + [2, 2],
               np.random.randn(100, 2) + [-2, -2]])
y = np.array([1] * 100 + [0] * 100)

model = LogisticRegressionScratch(lr=0.1, epochs=500)
model.fit(X, y)
preds = model.predict(X)
print(f"Accuracy: {np.mean(preds == y):.2%}")
print(f"Final loss: {model.losses[-1]:.4f}")`}
        />

        <h3>With scikit-learn</h3>
        <CodeBlock
          language="python"
          title="logistic_regression_sklearn.py"
          code={`from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features — important for logistic regression convergence
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# C is inverse of regularization strength (smaller C = more regularization)
model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Coefficients: {model.coef_[0]}")
print(f"Intercept: {model.intercept_[0]:.4f}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always scale features</strong>: Logistic regression with L-BFGS or SGD is sensitive to feature magnitudes. Use <code>StandardScaler</code> or <code>MinMaxScaler</code>.</li>
          <li><strong>Regularization is on by default in sklearn</strong>: The <code>C</code> parameter controls it (default 1.0). For high-dimensional data (text, genomics), use L1 for sparsity or ElasticNet.</li>
          <li><strong>Coefficients are interpretable</strong>: <InlineMath math="e^{w_j}" /> is the odds ratio — a one-unit increase in feature <InlineMath math="j" /> multiplies the odds of class 1 by <InlineMath math="e^{w_j}" />.</li>
          <li><strong>Adjust the threshold</strong>: The default 0.5 threshold is not always optimal. Use precision-recall curves to find the best threshold for your use case (e.g., 0.3 if false negatives are costly).</li>
          <li><strong>Great baseline</strong>: Always start with logistic regression before trying complex models. If it works well, you may not need anything fancier.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Forgetting to scale features</strong>: Without scaling, the optimizer may converge slowly or not at all, and coefficients are not comparable.</li>
          <li><strong>Using accuracy on imbalanced data</strong>: If 95% of samples are class 0, predicting all zeros gives 95% accuracy. Use ROC-AUC, precision, recall, or F1 instead.</li>
          <li><strong>Applying logistic regression to nonlinear boundaries</strong>: It can only learn linear decision boundaries. Add polynomial or interaction features, or switch to a nonlinear model.</li>
          <li><strong>Ignoring multicollinearity</strong>: Highly correlated features make coefficients unstable and uninterpretable. Use L2 (Ridge) regularization or drop correlated features.</li>
          <li><strong>Interpreting outputs as calibrated probabilities</strong>: While logistic regression outputs probabilities, they&apos;re only well-calibrated if the model assumptions hold. Use calibration plots to verify.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Why do we use cross-entropy loss for logistic regression instead of MSE? What goes wrong with MSE?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>MSE is non-convex for logistic regression</strong>: If you plug the sigmoid into the MSE loss <InlineMath math="(y - \sigma(z))^2" />, you get a non-convex function with multiple local minima. Gradient descent can get stuck.</li>
          <li><strong>Cross-entropy is convex</strong>: The negative log-likelihood <InlineMath math="-[y\log\sigma(z) + (1-y)\log(1-\sigma(z))]" /> is convex in <InlineMath math="z" />, guaranteeing a unique global minimum.</li>
          <li><strong>Better gradients</strong>: With MSE and sigmoid, the gradient includes <InlineMath math="\sigma'(z)" />, which vanishes when predictions are confident (the &quot;flat tails&quot; of the sigmoid). Cross-entropy avoids this — the gradient is simply <InlineMath math="(\hat{p} - y)" />, so learning never stalls due to saturated sigmoids.</li>
          <li><strong>Probabilistic justification</strong>: Cross-entropy is the maximum likelihood estimator under a Bernoulli model, making it the statistically principled choice.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>ESL Chapter 4.4</strong> — Logistic regression in the context of linear classifiers.</li>
          <li><strong>Bishop PRML Chapter 4</strong> — Bayesian treatment of logistic regression.</li>
          <li><strong>scikit-learn Logistic Regression docs</strong> — Solver comparison, multiclass strategies (OvR vs multinomial).</li>
          <li><strong>Platt Scaling (1999)</strong> — Calibrating probabilities from classifiers.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
