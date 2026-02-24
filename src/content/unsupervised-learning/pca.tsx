"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function PCA() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Principal Component Analysis (PCA) answers the question: <strong>if I could only keep a few dimensions of my data, which directions capture the most information?</strong> It finds new axes (called principal components) that are ordered by how much variance they explain. The first component points in the direction of maximum spread in the data, the second is perpendicular to the first and captures the next most spread, and so on.
        </p>
        <p>
          Think of it as rotating your data so that the axes align with the natural &quot;shape&quot; of the point cloud. If you have 100 features but most of the variation happens in 5 directions, PCA lets you keep just those 5 components and throw away the other 95 with minimal information loss. This is <strong>dimensionality reduction</strong> — and it helps with visualization, noise removal, speeding up models, and fighting the curse of dimensionality.
        </p>
        <p>
          Under the hood, PCA computes the <strong>eigenvectors</strong> of the covariance matrix (or equivalently, the <strong>singular vectors</strong> from SVD). The eigenvalues tell you how much variance each component explains. In practice, everyone uses SVD because it&apos;s numerically stable and doesn&apos;t require explicitly forming the covariance matrix.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Setup</h3>
        <p>Given centered data <InlineMath math="X \in \mathbb{R}^{n \times d}" /> (zero mean), the covariance matrix is:</p>
        <BlockMath math="\Sigma = \frac{1}{n-1}X^TX \in \mathbb{R}^{d \times d}" />

        <h3>Eigendecomposition Approach</h3>
        <p>Find eigenvectors of the covariance matrix:</p>
        <BlockMath math="\Sigma \mathbf{v}_j = \lambda_j \mathbf{v}_j" />
        <p>
          The eigenvector <InlineMath math="\mathbf{v}_1" /> with the largest eigenvalue <InlineMath math="\lambda_1" /> is the first principal component. The <InlineMath math="j" />-th component captures variance <InlineMath math="\lambda_j" />.
        </p>

        <h3>SVD Approach (Preferred)</h3>
        <BlockMath math="X = U \Sigma_s V^T" />
        <p>
          where <InlineMath math="V" /> contains the principal component directions (right singular vectors), <InlineMath math="\Sigma_s" /> contains singular values, and <InlineMath math="U\Sigma_s" /> gives the projected data. The singular values <InlineMath math="\sigma_j" /> relate to eigenvalues by <InlineMath math="\lambda_j = \frac{\sigma_j^2}{n-1}" />.
        </p>

        <h3>Projection</h3>
        <p>Project <InlineMath math="X" /> to <InlineMath math="k" /> dimensions:</p>
        <BlockMath math="Z = X V_k \in \mathbb{R}^{n \times k}" />
        <p>where <InlineMath math="V_k" /> is the matrix of the first <InlineMath math="k" /> principal components.</p>

        <h3>Variance Explained</h3>
        <BlockMath math="\text{Proportion of variance by component } j = \frac{\lambda_j}{\sum_{i=1}^{d}\lambda_i}" />
        <BlockMath math="\text{Cumulative variance by first } k \text{ components} = \frac{\sum_{j=1}^{k}\lambda_j}{\sum_{i=1}^{d}\lambda_i}" />

        <h3>Reconstruction Error</h3>
        <BlockMath math="\text{Reconstruction} = Z V_k^T, \quad \text{Error} = \|X - ZV_k^T\|_F^2 = \sum_{j=k+1}^{d}\lambda_j" />
        <p>PCA minimizes this reconstruction error among all linear projections — it is the <strong>optimal low-rank approximation</strong> (Eckart-Young theorem).</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>From Scratch</h3>
        <CodeBlock
          language="python"
          title="pca_scratch.py"
          code={`import numpy as np

class PCAScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        # Center the data
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        # SVD (more stable than eigendecomposition of covariance)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.components = Vt[:self.n_components]   # principal directions
        self.singular_values = S[:self.n_components]

        # Variance explained
        total_var = np.sum(S ** 2) / (len(X) - 1)
        self.explained_variance = S[:self.n_components]**2 / (len(X) - 1)
        self.explained_variance_ratio = self.explained_variance / total_var
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Z):
        return Z @ self.components + self.mean

# Example: reduce 50D data to 2D
np.random.seed(42)
# Data lives mostly in a 3D subspace of 50D
true_data = np.random.randn(500, 3) @ np.random.randn(3, 50)
noise = np.random.randn(500, 50) * 0.1
X = true_data + noise

pca = PCAScratch(n_components=5)
pca.fit(X)

print("Variance explained by first 5 components:")
for i, ratio in enumerate(pca.explained_variance_ratio):
    print(f"  PC{i+1}: {ratio:.4f} ({100*ratio:.1f}%)")
print(f"  Total: {pca.explained_variance_ratio.sum():.4f}")

X_reduced = pca.transform(X)
X_reconstructed = pca.inverse_transform(X_reduced[:, :3])
recon_error = np.mean((X - pca.inverse_transform(pca.transform(X)))**2)
print(f"\\nReconstruction MSE (5 components): {recon_error:.6f}")`}
        />

        <h3>With scikit-learn</h3>
        <CodeBlock
          language="python"
          title="pca_sklearn.py"
          code={`from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Always standardize before PCA!
pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95))
# n_components=0.95 automatically picks k to retain 95% variance

X_reduced = pipe.fit_transform(X)
pca = pipe.named_steps["pca"]

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions:  {X_reduced.shape[1]}")
print(f"Variance retained:   {pca.explained_variance_ratio_.sum():.4f}")

# Scree plot data
print("\\nVariance explained per component:")
cumulative = np.cumsum(pca.explained_variance_ratio_)
for i in range(min(10, len(cumulative))):
    print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} "
          f"(cumulative: {cumulative[i]:.4f})")

# PCA as preprocessing for a classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml

# Example: MNIST digit classification
# X_mnist, y_mnist = fetch_openml("mnist_784", version=1, return_X_y=True)
# pipe = make_pipeline(
#     StandardScaler(),
#     PCA(n_components=50),  # 784 -> 50 dimensions
#     LogisticRegression(max_iter=1000)
# )
# scores = cross_val_score(pipe, X_mnist[:5000], y_mnist[:5000], cv=5)
# print(f"Accuracy with 50 PCs: {scores.mean():.4f}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always standardize before PCA</strong>: PCA maximizes variance. If features are on different scales, the high-magnitude feature dominates. Use <code>StandardScaler</code> (zero mean, unit variance) unless features are already on the same scale.</li>
          <li><strong>Use n_components=0.95 for automatic selection</strong>: Retain 95% (or 99%) of variance and let sklearn choose the right number of components.</li>
          <li><strong>PCA for visualization</strong>: Project to 2 or 3 dimensions and plot. If clusters are visible in PCA space, your data has real structure. If not, you may need nonlinear methods (t-SNE, UMAP).</li>
          <li><strong>PCA for denoising</strong>: Project to <InlineMath math="k" /> components and reconstruct. The discarded components are mostly noise, so the reconstruction is a denoised version of the original data.</li>
          <li><strong>Beware of interpretability</strong>: Principal components are linear combinations of all original features. PC1 might be &quot;0.3 * age + 0.5 * income - 0.2 * debt + ...&quot; — hard to name.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Applying PCA without centering</strong>: PCA assumes zero-mean data. Without centering, the first component points toward the data mean, which is meaningless. Sklearn&apos;s PCA centers automatically, but double-check if implementing from scratch.</li>
          <li><strong>Using PCA on categorical or binary features</strong>: PCA assumes continuous data with linear relationships. For mixed data, consider FAMD or MCA (Multiple Correspondence Analysis).</li>
          <li><strong>Keeping too few components</strong>: If you reduce from 100 to 2 dimensions and only retain 20% of variance, you&apos;ve thrown away 80% of the signal. Check the cumulative variance plot.</li>
          <li><strong>Assuming PCA finds clusters</strong>: PCA finds directions of maximum variance, not clusters. A dataset can have clear clusters that aren&apos;t visible along the top principal components.</li>
          <li><strong>Fitting PCA on test data</strong>: Fit PCA on training data only, then use the same transformation on test data. Otherwise, you leak information from the test set.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain the relationship between PCA and SVD. Why do we use SVD in practice instead of eigendecomposition?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Connection</strong>: If <InlineMath math="X = U\Sigma V^T" /> (SVD of centered data), then the covariance matrix is <InlineMath math="\frac{1}{n-1}X^TX = V\frac{\Sigma^2}{n-1}V^T" />. So the right singular vectors <InlineMath math="V" /> are the eigenvectors of the covariance matrix, and the eigenvalues are <InlineMath math="\sigma_j^2/(n-1)" />.</li>
          <li><strong>Numerical stability</strong>: Forming <InlineMath math="X^TX" /> explicitly squares the condition number. If <InlineMath math="X" /> has condition number <InlineMath math="\kappa" />, <InlineMath math="X^TX" /> has condition number <InlineMath math="\kappa^2" />. SVD avoids this entirely.</li>
          <li><strong>Memory efficiency</strong>: If <InlineMath math="n \ll d" /> (more features than samples), forming the <InlineMath math="d \times d" /> covariance matrix is expensive. Truncated SVD computes only the top <InlineMath math="k" /> components directly.</li>
          <li><strong>Sparse data</strong>: <code>TruncatedSVD</code> (sklearn) works on sparse matrices directly, while PCA requires dense matrices. This is critical for text data (TF-IDF matrices).</li>
          <li><strong>Practical note</strong>: Sklearn&apos;s <code>PCA</code> uses a randomized SVD algorithm (Halko et al., 2011) when <InlineMath math="k \ll \min(n, d)" />, making it <InlineMath math="O(n d k)" /> instead of <InlineMath math="O(n d \min(n,d))" />.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>ESL Chapter 14.5</strong> — Principal Components, Curves, and Surfaces.</li>
          <li><strong>Shlens (2014) &quot;A Tutorial on Principal Component Analysis&quot;</strong> — Clear, accessible derivation from multiple perspectives.</li>
          <li><strong>Halko et al. (2011) &quot;Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions&quot;</strong> — The randomized SVD that sklearn uses.</li>
          <li><strong>scikit-learn Decomposition guide</strong> — PCA, Kernel PCA, Incremental PCA, Sparse PCA, NMF, and more.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
