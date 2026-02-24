"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";
import VectorTransform from "@/components/viz/VectorTransform";

export default function LinearAlgebra() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Linear algebra is the math of <strong>arrows and grids</strong>. A <strong>vector</strong> is
          just a list of numbers — but geometrically, it&apos;s an arrow pointing somewhere in space. A{" "}
          <strong>matrix</strong> is a machine that <em>transforms</em> those arrows: it can rotate them,
          stretch them, squish them, flip them, or project them onto lower dimensions.
        </p>
        <p>
          Why does this matter for ML? Every dataset is a matrix — rows are data points, columns are features.
          Every neural network layer is a matrix multiplication followed by a nonlinearity. When you do PCA,
          you&apos;re finding the eigenvalues of a covariance matrix. When you train a model, you&apos;re
          navigating a high-dimensional space using gradients (vectors).
        </p>
        <p>
          <strong>The key insight:</strong> matrices don&apos;t just store numbers — they represent
          <em> transformations</em>. Understanding what a matrix <em>does</em> to space is more important
          than memorizing formulas.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Vectors</h3>
        <p>
          A vector <InlineMath math="\mathbf{v} \in \mathbb{R}^n" /> is an ordered list of <InlineMath math="n" /> real numbers:
        </p>
        <BlockMath math="\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}" />
        <p>
          The <strong>dot product</strong> of two vectors measures how &quot;aligned&quot; they are:
        </p>
        <BlockMath math="\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta" />

        <h3>Matrix Multiplication as Transformation</h3>
        <p>
          A matrix <InlineMath math="A \in \mathbb{R}^{m \times n}" /> transforms vectors from <InlineMath math="\mathbb{R}^n" /> to <InlineMath math="\mathbb{R}^m" />:
        </p>
        <BlockMath math="\mathbf{y} = A\mathbf{x} \quad \text{where } A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}" />
        <p>
          The columns of <InlineMath math="A" /> tell you where the standard basis vectors land after transformation.
        </p>

        <h3>Eigenvalues and Eigenvectors</h3>
        <p>
          An eigenvector <InlineMath math="\mathbf{v}" /> of a matrix <InlineMath math="A" /> is a special direction that the matrix only <em>stretches</em> (doesn&apos;t rotate):
        </p>
        <BlockMath math="A\mathbf{v} = \lambda \mathbf{v}" />
        <p>
          The scalar <InlineMath math="\lambda" /> (eigenvalue) tells you how much it stretches. In PCA, the eigenvectors of the covariance matrix point in the directions of maximum variance.
        </p>

        <h3>Singular Value Decomposition (SVD)</h3>
        <p>
          Any matrix can be decomposed as:
        </p>
        <BlockMath math="A = U \Sigma V^T" />
        <p>
          where <InlineMath math="U" /> and <InlineMath math="V" /> are orthogonal matrices (rotations) and <InlineMath math="\Sigma" /> is diagonal (stretching). SVD is the Swiss Army knife of linear algebra — it powers PCA, pseudoinverses, low-rank approximations, and more.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>From Scratch — Dot Product & Matrix Multiply</h3>
        <CodeBlock
          language="python"
          title="linear_algebra_from_scratch.py"
          code={`import numpy as np

# Vectors
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Dot product (from scratch)
dot_manual = sum(a * b for a, b in zip(u, v))
dot_numpy = np.dot(u, v)
print(f"Dot product: {dot_manual} (manual) = {dot_numpy} (numpy)")
# Output: Dot product: 32 (manual) = 32 (numpy)

# Matrix multiplication (from scratch)
def matmul(A, B):
    """Multiply two matrices."""
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2, "Dimension mismatch"
    C = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
print("Manual matmul:", matmul(A, B))
# Output: [[19, 22], [43, 50]]

# NumPy version (use this in practice!)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("NumPy matmul:", A @ B)
# Output: [[19 22] [43 50]]`}
        />

        <h3>Eigendecomposition & SVD with NumPy</h3>
        <CodeBlock
          language="python"
          title="eigen_svd.py"
          code={`import numpy as np

# Create a symmetric matrix (covariance-like)
A = np.array([[4, 2], [2, 3]])

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(A)
print(f"Eigenvalues: {eigenvalues}")   # [1.586, 5.414]
print(f"Eigenvectors:\\n{eigenvectors}")

# Verify: A @ v = lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

# SVD
U, sigma, Vt = np.linalg.svd(A)
print(f"\\nSingular values: {sigma}")
# Reconstruct: A = U @ diag(sigma) @ Vt
A_reconstructed = U @ np.diag(sigma) @ Vt
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")

# Low-rank approximation (keep top k singular values)
k = 1
A_rank1 = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
print(f"Rank-1 approximation:\\n{A_rank1}")
print(f"Approximation error: {np.linalg.norm(A - A_rank1):.4f}")`}
        />
      </TopicSection>

      <TopicSection type="see-it">
        <p className="mb-4">
          Drag the sliders to transform 2D space with a matrix. Watch how the basis vectors{" "}
          <span className="text-pink-400 font-bold">e₁</span> and{" "}
          <span className="text-green-400 font-bold">e₂</span> move, and how the grid deforms.
          The <span className="text-indigo-400 font-bold">shaded parallelogram</span> shows the determinant — the factor by which the matrix scales area.
        </p>
        <VectorTransform />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>PCA</strong>: Eigendecomposition of the covariance matrix gives you the principal components (directions of max variance).</li>
          <li><strong>Neural networks</strong>: Every fully-connected layer computes <InlineMath math="y = Wx + b" /> — a linear transformation plus bias.</li>
          <li><strong>Embeddings</strong>: Word2Vec, BERT embeddings — all live in high-dimensional vector spaces where dot products measure similarity.</li>
          <li><strong>Recommendation systems</strong>: Matrix factorization decomposes the user-item interaction matrix into low-rank factors.</li>
          <li><strong>When NOT to memorize formulas</strong>: Focus on understanding what operations <em>do</em> geometrically. You almost never implement matmul from scratch — use NumPy/PyTorch.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Confusing dot product with element-wise multiply</strong>: <code>np.dot(a, b)</code> sums the products, <code>a * b</code> returns a vector of the same size.</li>
          <li><strong>Matrix dimension mismatches</strong>: <InlineMath math="(m \times n) \cdot (n \times p) = (m \times p)" />. The inner dimensions must match.</li>
          <li><strong>Assuming all matrices are invertible</strong>: A matrix is only invertible if its determinant is non-zero (full rank). In practice, many matrices are close to singular.</li>
          <li><strong>Forgetting that matrix multiply is NOT commutative</strong>: <InlineMath math="AB \neq BA" /> in general.</li>
          <li><strong>Using eigendecomposition on non-symmetric matrices</strong>: Use <code>np.linalg.eig</code> (not <code>eigh</code>) for non-symmetric. Better yet, use SVD which always works.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You have a 10,000 x 500 feature matrix. How would you reduce it to 50 dimensions while preserving maximum information? Walk through the math.</p>
        <p><strong>Solution:</strong></p>
        <ol>
          <li>Center the data (subtract column means).</li>
          <li>Compute the covariance matrix <InlineMath math="C = \frac{1}{n-1}X^TX" /> (500x500).</li>
          <li>Find the top 50 eigenvectors of <InlineMath math="C" /> — these are the principal components.</li>
          <li>Project: <InlineMath math="X_{reduced} = X \cdot V_{50}" /> where <InlineMath math="V_{50}" /> contains the top 50 eigenvectors.</li>
          <li>In practice, use SVD: <InlineMath math="X = U\Sigma V^T" />, then <InlineMath math="X_{50} = U_{50}\Sigma_{50}" />. This avoids computing the covariance matrix explicitly (numerically more stable).</li>
        </ol>
        <CodeBlock
          language="python"
          code={`from sklearn.decomposition import PCA

pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)  # (10000, 50)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")`}
        />
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>3Blue1Brown &quot;Essence of Linear Algebra&quot;</strong> — The best visual series on linear algebra ever made</li>
          <li><strong>Gilbert Strang, MIT 18.06</strong> — The gold standard linear algebra course (free on MIT OCW)</li>
          <li><strong>Matrix Cookbook</strong> — Quick reference for matrix identities and derivatives</li>
          <li><strong>NumPy Linear Algebra docs</strong> — <code>np.linalg</code> has everything you need in practice</li>
        </ul>
      </TopicSection>
    </div>
  );
}
