"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function SupportVectorMachines() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Most classifiers just try to separate classes — they find <em>some</em> boundary that works. Support Vector Machines find the <strong>best</strong> boundary: the one that maximizes the <strong>margin</strong>, which is the distance between the decision boundary and the closest data points from each class. These closest points are called <strong>support vectors</strong> — they &quot;support&quot; the boundary like tent poles holding up a canvas.
        </p>
        <p>
          Why does maximizing the margin matter? Intuitively, a wider margin means the classifier is more confident and more robust to noise. A new data point that&apos;s slightly off from the training distribution is less likely to be misclassified. Formally, margin maximization is connected to generalization bounds in statistical learning theory — wider margins correspond to lower VC dimension and better generalization.
        </p>
        <p>
          But what if the data isn&apos;t linearly separable? Two solutions: <strong>soft margins</strong> allow some points to violate the boundary (controlled by the <InlineMath math="C" /> parameter), and the <strong>kernel trick</strong> implicitly maps data into a higher-dimensional space where it <em>becomes</em> linearly separable. The RBF kernel, for example, maps to an infinite-dimensional space — and SVMs can work in it efficiently without ever computing the actual coordinates.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Hard-Margin SVM</h3>
        <p>For linearly separable data with labels <InlineMath math="y_i \in \{-1, +1\}" />:</p>
        <BlockMath math="\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 \; \forall \, i" />
        <p>
          The margin width is <InlineMath math="\frac{2}{\|\mathbf{w}\|}" />, so minimizing <InlineMath math="\|\mathbf{w}\|^2" /> maximizes the margin.
        </p>

        <h3>Soft-Margin SVM</h3>
        <p>Allow violations with slack variables <InlineMath math="\xi_i \geq 0" />:</p>
        <BlockMath math="\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\xi_i" />
        <BlockMath math="\text{subject to} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0" />
        <p>
          <InlineMath math="C" /> controls the tradeoff: large <InlineMath math="C" /> penalizes violations heavily (narrow margin, low bias), small <InlineMath math="C" /> allows more violations (wide margin, high bias).
        </p>

        <h3>Hinge Loss Formulation</h3>
        <p>The soft-margin SVM is equivalent to minimizing:</p>
        <BlockMath math="\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))" />
        <p>The <InlineMath math="\max(0, 1 - y_i f(\mathbf{x}_i))" /> term is the <strong>hinge loss</strong>. Points correctly classified with margin &gt; 1 contribute zero loss.</p>

        <h3>The Dual Problem and the Kernel Trick</h3>
        <p>The dual formulation expresses the problem in terms of dot products <InlineMath math="\mathbf{x}_i^T \mathbf{x}_j" />:</p>
        <BlockMath math="\max_{\boldsymbol{\alpha}} \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j" />
        <BlockMath math="\text{subject to} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0" />
        <p>
          The <strong>kernel trick</strong>: replace <InlineMath math="\mathbf{x}_i^T\mathbf{x}_j" /> with a kernel function <InlineMath math="K(\mathbf{x}_i, \mathbf{x}_j)" /> that computes the dot product in a higher-dimensional space <em>without explicitly mapping there</em>:
        </p>
        <ul>
          <li><strong>Linear</strong>: <InlineMath math="K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T\mathbf{z}" /></li>
          <li><strong>Polynomial</strong>: <InlineMath math="K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T\mathbf{z} + r)^d" /></li>
          <li><strong>RBF (Gaussian)</strong>: <InlineMath math="K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma\|\mathbf{x} - \mathbf{z}\|^2)" /> — maps to infinite dimensions!</li>
        </ul>
      </TopicSection>

      <TopicSection type="code">
        <h3>From Scratch — Linear SVM with Hinge Loss</h3>
        <CodeBlock
          language="python"
          title="svm_scratch.py"
          code={`import numpy as np

class LinearSVMScratch:
    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        """Train via subgradient descent on hinge loss."""
        # Convert labels to {-1, +1}
        y_svm = np.where(y == 0, -1, y)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.epochs):
            margins = y_svm * (X @ self.w + self.b)

            # Subgradient of hinge loss
            violated = margins < 1  # points inside margin or misclassified
            dw = self.w - self.C * (y_svm[violated] @ X[violated])
            db = -self.C * y_svm[violated].sum()

            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

# Example: two concentric circles won't work (not linearly separable)
# Two blobs will
np.random.seed(42)
X = np.vstack([np.random.randn(100, 2) + [2, 2],
               np.random.randn(100, 2) + [-2, -2]])
y = np.array([1] * 100 + [-1] * 100)

svm = LinearSVMScratch(C=1.0, lr=0.0001, epochs=1000)
svm.fit(X, y)
preds = svm.predict(X)
print(f"Accuracy: {np.mean(preds == y):.2%}")
print(f"Support vector count: {np.sum(np.abs(y * (X @ svm.w + svm.b)) < 1.05)}")`}
        />

        <h3>With scikit-learn</h3>
        <CodeBlock
          language="python"
          title="svm_sklearn.py"
          code={`from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.datasets import make_moons

# Non-linearly separable data
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# RBF kernel SVM — always scale features first!
svm_rbf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
)
svm_rbf.fit(X_train, y_train)
print(f"RBF SVM accuracy: {svm_rbf.score(X_test, y_test):.2%}")

# Tune C and gamma with grid search
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": [0.01, 0.1, 1, "scale"],
}
grid = GridSearchCV(svm_rbf, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.2%}")
print(f"Test accuracy: {grid.score(X_test, y_test):.2%}")

# For large datasets, use LinearSVC (much faster)
linear_svm = make_pipeline(
    StandardScaler(),
    LinearSVC(C=1.0, max_iter=10000)
)
linear_svm.fit(X_train, y_train)
print(f"Linear SVM accuracy: {linear_svm.score(X_test, y_test):.2%}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always scale features</strong>: SVMs are not scale-invariant. Features on different scales will dominate the distance calculations. Use <code>StandardScaler</code>.</li>
          <li><strong>RBF kernel is the default choice</strong>: It works well for most nonlinear problems and has only two hyperparameters (<InlineMath math="C" /> and <InlineMath math="\gamma" />). Start here before trying polynomial or custom kernels.</li>
          <li><strong>C and gamma interact</strong>: Large <InlineMath math="C" /> + large <InlineMath math="\gamma" /> = very complex boundary (overfitting). Small <InlineMath math="C" /> + small <InlineMath math="\gamma" /> = very smooth boundary (underfitting). Use grid search or Bayesian optimization.</li>
          <li><strong>SVMs don&apos;t scale to large datasets</strong>: Training is <InlineMath math="O(n^2)" /> to <InlineMath math="O(n^3)" /> for kernel SVMs. For 100K+ samples, use <code>LinearSVC</code>, SGDClassifier with hinge loss, or switch to gradient boosting.</li>
          <li><strong>Probabilities require extra work</strong>: SVM outputs distances, not probabilities. Set <code>probability=True</code> to enable Platt scaling, but it&apos;s slow and adds a 5-fold CV internally.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Forgetting to scale</strong>: The number one SVM mistake. An unscaled SVM performs dramatically worse.</li>
          <li><strong>Using kernel SVM on huge datasets</strong>: RBF SVM on 1M samples will run for days. Use LinearSVC or approximate kernels (<code>sklearn.kernel_approximation.Nystroem</code>).</li>
          <li><strong>Not tuning gamma</strong>: The default <code>gamma=&quot;scale&quot;</code> is reasonable, but the optimal value can vary by orders of magnitude. Always cross-validate.</li>
          <li><strong>Expecting SVMs to handle missing values</strong>: They can&apos;t. Impute first.</li>
          <li><strong>Assuming more support vectors = better model</strong>: If most training points are support vectors, the model is underfitting (C is too small) or the problem is fundamentally hard.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain the kernel trick. Why can SVMs work in infinite-dimensional spaces efficiently?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>The problem</strong>: Some data isn&apos;t linearly separable in the original feature space. We could map it to a higher-dimensional space <InlineMath math="\phi(\mathbf{x})" /> where it becomes separable, but computing <InlineMath math="\phi" /> explicitly for high dimensions is prohibitively expensive.</li>
          <li><strong>The insight</strong>: The SVM dual formulation only needs <em>dot products</em> between data points, never the individual coordinates. We can replace <InlineMath math="\phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)" /> with a kernel function <InlineMath math="K(\mathbf{x}_i, \mathbf{x}_j)" /> that computes the same result directly in the original space.</li>
          <li><strong>Example — RBF kernel</strong>: <InlineMath math="K(\mathbf{x}, \mathbf{z}) = e^{-\gamma\|\mathbf{x}-\mathbf{z}\|^2}" /> corresponds to an <em>infinite-dimensional</em> feature mapping (a Taylor expansion that never terminates). Yet computing the kernel takes <InlineMath math="O(d)" /> time — same as a regular dot product.</li>
          <li><strong>Requirements</strong>: A function <InlineMath math="K" /> is a valid kernel if and only if the kernel matrix <InlineMath math="K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)" /> is positive semi-definite for any set of points (Mercer&apos;s theorem).</li>
          <li><strong>Computational cost</strong>: We only need to compute the <InlineMath math="n \times n" /> kernel matrix, not the (possibly infinite) feature vectors. This is what makes the trick practical.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Vapnik (1995) &quot;The Nature of Statistical Learning Theory&quot;</strong> — The original SVM paper, connecting margin to generalization.</li>
          <li><strong>ESL Chapter 12</strong> — Support Vector Machines and Kernels.</li>
          <li><strong>Scholkopf &amp; Smola (2002) &quot;Learning with Kernels&quot;</strong> — The comprehensive reference on kernel methods.</li>
          <li><strong>scikit-learn SVM guide</strong> — Practical tips on scaling, kernel selection, and computational considerations.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
