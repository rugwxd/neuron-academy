"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";
import LinearRegressionViz from "@/components/viz/LinearRegressionViz";

export default function LinearRegression() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Linear regression answers the simplest question in machine learning: <strong>what&apos;s the best straight line through my data?</strong>
        </p>
        <p>
          You have some input <InlineMath math="x" /> (like house size) and you want to predict some output <InlineMath math="y" /> (like price).
          Linear regression finds the line <InlineMath math="y = mx + b" /> that gets as close to all the data points as possible.
          &quot;As close as possible&quot; means minimizing the sum of squared vertical distances from each point to the line — these are called <strong>residuals</strong>.
        </p>
        <p>
          Despite its simplicity, linear regression is incredibly powerful. It&apos;s interpretable (the slope tells you exactly how much <InlineMath math="y" /> changes per unit of <InlineMath math="x" />), it&apos;s fast, and it forms the foundation for understanding more complex models.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>The Model</h3>
        <BlockMath math="\hat{y} = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b" />

        <h3>Objective: Minimize Mean Squared Error</h3>
        <BlockMath math="L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\|\mathbf{y} - X\mathbf{w}\|^2" />

        <h3>Closed-Form Solution (Normal Equation)</h3>
        <p>Setting the gradient to zero and solving:</p>
        <BlockMath math="\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}" />
        <p>
          This is the <strong>OLS (Ordinary Least Squares)</strong> solution. It gives you the exact answer in one step — no iteration needed.
        </p>

        <h3>Gradient Descent Alternative</h3>
        <p>When <InlineMath math="X^TX" /> is too large to invert (millions of features), use gradient descent:</p>
        <BlockMath math="\frac{\partial L}{\partial \mathbf{w}} = -\frac{2}{n}X^T(\mathbf{y} - X\mathbf{w})" />
        <BlockMath math="\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{\partial L}{\partial \mathbf{w}}" />

        <h3>R² (Coefficient of Determination)</h3>
        <BlockMath math="R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}" />
        <p>
          <InlineMath math="R^2 = 1" /> means perfect fit. <InlineMath math="R^2 = 0" /> means no better than predicting the mean.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>From Scratch</h3>
        <CodeBlock
          language="python"
          title="linear_regression_scratch.py"
          code={`import numpy as np

class LinearRegressionScratch:
    def __init__(self):
        self.w = None
        self.b = None

    def fit_closed_form(self, X, y):
        """OLS closed-form solution."""
        # Add bias column
        X_b = np.column_stack([np.ones(len(X)), X])
        # Normal equation: w = (X'X)^-1 X'y
        w = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)
        self.b = w[0]
        self.w = w[1:]
        return self

    def fit_gradient_descent(self, X, y, lr=0.01, epochs=1000):
        """Gradient descent solution."""
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(epochs):
            y_hat = X @ self.w + self.b
            residuals = y_hat - y

            # Gradients
            dw = (2 / n) * X.T @ residuals
            db = (2 / n) * residuals.sum()

            self.w -= lr * dw
            self.b -= lr * db
        return self

    def predict(self, X):
        return X @ self.w + self.b

    def r_squared(self, X, y):
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot

# Example
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.squeeze() + 3 + np.random.randn(100) * 2

model = LinearRegressionScratch()
model.fit_closed_form(X, y)
print(f"w = {model.w[0]:.4f}, b = {model.b:.4f}")
print(f"R² = {model.r_squared(X, y):.4f}")
# w ≈ 2.5, b ≈ 3.0, R² ≈ 0.94`}
        />

        <h3>With scikit-learn (Use This)</h3>
        <CodeBlock
          language="python"
          title="linear_regression_sklearn.py"
          code={`from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")`}
        />
      </TopicSection>

      <TopicSection type="see-it">
        <p className="mb-4">
          <strong>Click</strong> anywhere on the plot to add data points. <strong>Drag</strong> existing points to move them.
          Watch the regression line (blue) and residuals (pink dashed) update in real-time.
          Try adding an outlier far from the line to see how it pulls the fit.
        </p>
        <LinearRegressionViz />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always check residuals</strong> — plot residuals vs predicted values. They should look random. Patterns mean your model is missing something (try polynomial features).</li>
          <li><strong>Multicollinearity</strong> — if features are highly correlated, coefficients become unstable. Check VIF (Variance Inflation Factor) or use Ridge regression.</li>
          <li><strong>Feature scaling matters for gradient descent</strong> — standardize features (zero mean, unit variance) before training. The normal equation doesn&apos;t need this.</li>
          <li><strong>Use regularization for many features</strong> — Ridge (L2) or Lasso (L1) prevents overfitting when <InlineMath math="p \gg n" />.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Evaluating on training data</strong> — always use a held-out test set. Training R² is always optimistic.</li>
          <li><strong>Adding R² values of individual features</strong> — R² is not additive. A feature alone might have low R² but contribute a lot in combination with others.</li>
          <li><strong>Interpreting coefficients without scaling</strong> — a coefficient of 1000 doesn&apos;t mean that feature is more important if it&apos;s measured in millimeters while another is in kilometers.</li>
          <li><strong>Extrapolating beyond training data range</strong> — linear regression is only valid within the range of your training data.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> What are the assumptions of linear regression? What happens when they&apos;re violated?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Linearity</strong>: The relationship between X and y is linear. Violated → add polynomial/interaction features.</li>
          <li><strong>Independence</strong>: Observations are independent. Violated (e.g., time series) → use appropriate models or add lag features.</li>
          <li><strong>Homoscedasticity</strong>: Constant error variance. Violated → use weighted least squares or log-transform y.</li>
          <li><strong>Normality of errors</strong>: Errors are normally distributed. Violated → still works for prediction (OLS is BLUE by Gauss-Markov), but confidence intervals and p-values are unreliable.</li>
          <li><strong>No perfect multicollinearity</strong>: No feature is a perfect linear combination of others. Violated → matrix is singular, use Ridge.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>ESL Chapter 3</strong> — Hastie, Tibshirani, Friedman. The definitive treatment of linear methods.</li>
          <li><strong>scikit-learn Linear Models guide</strong> — Practical comparison of OLS, Ridge, Lasso, ElasticNet.</li>
          <li><strong>Gauss-Markov Theorem</strong> — Proves OLS is the best linear unbiased estimator (BLUE).</li>
        </ul>
      </TopicSection>
    </div>
  );
}
