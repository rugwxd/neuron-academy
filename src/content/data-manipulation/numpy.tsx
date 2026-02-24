"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function NumPy() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          NumPy is the foundation of nearly every numerical computing library in Python. At its core, it provides the <strong>ndarray</strong> — a
          multidimensional, homogeneously-typed array that lives in a contiguous block of memory. This is fundamentally different from a Python list,
          which is an array of pointers to scattered objects on the heap. That contiguous layout is why NumPy can be 10-100x faster than pure Python
          loops: it enables CPU cache locality, SIMD vectorization, and calls down into optimized C/Fortran routines (BLAS, LAPACK) under the hood.
        </p>
        <p>
          <strong>Broadcasting</strong> is NumPy&apos;s mechanism for performing arithmetic on arrays of different shapes without copying data. When you
          add a scalar to a matrix, or add a row vector to every row of a matrix, NumPy &quot;broadcasts&quot; the smaller array across the larger one
          by virtually repeating it — no actual memory duplication occurs. Understanding broadcasting rules is essential because it eliminates explicit
          loops and makes your code both faster and more readable.
        </p>
        <p>
          <strong>Vectorization</strong> means replacing explicit Python for-loops with array-level operations. Instead of looping over every element to
          compute a dot product, you write <code>np.dot(a, b)</code> and let NumPy dispatch to a compiled routine. This shift in thinking — from
          &quot;iterate over elements&quot; to &quot;operate on whole arrays&quot; — is the single most important skill for writing performant data science
          code in Python.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Array Shapes and Dimensions</h3>
        <p>
          A NumPy array with shape <InlineMath math="(n_1, n_2, \ldots, n_k)" /> stores <InlineMath math="\prod_{i=1}^{k} n_i" /> elements
          in a contiguous block. The <strong>strides</strong> tuple tells NumPy how many bytes to jump to move along each axis.
        </p>

        <h3>Broadcasting Rules</h3>
        <p>
          When operating on two arrays, NumPy compares their shapes element-wise, starting from the <strong>trailing dimensions</strong>. Two
          dimensions are compatible when:
        </p>
        <ol>
          <li>They are equal, OR</li>
          <li>One of them is 1</li>
        </ol>
        <p>
          For example, adding arrays of shape <InlineMath math="(3, 4)" /> and <InlineMath math="(4,)" /> works because the trailing dimension
          matches. The <InlineMath math="(4,)" /> array is broadcast across the first axis. Formally:
        </p>
        <BlockMath math="A \in \mathbb{R}^{m \times n}, \; \mathbf{b} \in \mathbb{R}^{n} \implies (A + \mathbf{b})_{ij} = A_{ij} + b_j" />

        <h3>Vectorization: Computational Complexity</h3>
        <p>
          The algorithmic complexity of a vectorized operation and its loop equivalent are identical — both are <InlineMath math="O(n)" /> for
          element-wise ops, <InlineMath math="O(n^3)" /> for matrix multiply (naive), etc. The speedup comes from <strong>constant factor
          improvements</strong>: no Python interpreter overhead per element, cache-friendly memory access, and SIMD instructions that process
          4-8 floats per CPU cycle.
        </p>

        <h3>Matrix Operations</h3>
        <p>Matrix multiplication of <InlineMath math="A \in \mathbb{R}^{m \times p}" /> and <InlineMath math="B \in \mathbb{R}^{p \times n}" />:</p>
        <BlockMath math="C_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}" />
        <p>
          NumPy&apos;s <code>@</code> operator and <code>np.matmul</code> dispatch to BLAS <code>dgemm</code>, achieving near-theoretical
          peak FLOPS on modern CPUs.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Array Creation and Basics</h3>
        <CodeBlock
          language="python"
          title="numpy_basics.py"
          code={`import numpy as np

# --- Array Creation ---
a = np.array([1, 2, 3, 4, 5])               # From a list
b = np.zeros((3, 4))                          # 3x4 matrix of zeros
c = np.ones((2, 3, 4))                        # 3D tensor of ones
d = np.arange(0, 10, 0.5)                     # Like range(), but returns ndarray
e = np.linspace(0, 1, 50)                     # 50 evenly spaced points in [0, 1]
f = np.random.randn(1000, 5)                  # 1000x5 standard normal samples
eye = np.eye(4)                                # 4x4 identity matrix

# --- Shapes and Reshaping ---
x = np.arange(24)
print(x.shape)                                 # (24,)
x_3d = x.reshape(2, 3, 4)                     # Reshape to 3D (must preserve total elements)
print(x_3d.shape)                              # (2, 3, 4)
print(x_3d[1, 2, 3])                          # Access element: 23

# Flatten back
print(x_3d.ravel().shape)                     # (24,)  — returns a view if possible

# Transpose
A = np.random.randn(3, 5)
print(A.T.shape)                               # (5, 3)

# --- Data Types ---
arr_float = np.array([1, 2, 3], dtype=np.float32)
arr_int = np.array([1.7, 2.3], dtype=np.int64)   # Truncates: [1, 2]
print(arr_float.dtype, arr_int.dtype)              # float32 int64`}
        />

        <h3>Broadcasting in Action</h3>
        <CodeBlock
          language="python"
          title="broadcasting.py"
          code={`import numpy as np

# --- Scalar broadcast ---
A = np.array([[1, 2, 3],
              [4, 5, 6]])            # shape (2, 3)
print(A * 10)                         # Each element multiplied by 10

# --- Row vector broadcast ---
row = np.array([100, 200, 300])       # shape (3,)
print(A + row)
# [[101, 202, 303],
#  [104, 205, 306]]
# The row is added to EVERY row of A

# --- Column vector broadcast ---
col = np.array([[10],
                [20]])                # shape (2, 1)
print(A + col)
# [[11, 12, 13],
#  [24, 25, 26]]
# The column is added to EVERY column of A

# --- Outer product via broadcasting ---
x = np.array([1, 2, 3])              # shape (3,)
y = np.array([10, 20])               # shape (2,)
outer = x[:, np.newaxis] * y[np.newaxis, :]   # (3,1) * (1,2) -> (3,2)
print(outer)
# [[10, 20],
#  [20, 40],
#  [30, 60]]

# --- Real example: z-score normalization per column ---
data = np.random.randn(1000, 5) * [10, 1, 0.1, 100, 50] + [5, -3, 0, 200, 50]
means = data.mean(axis=0)            # shape (5,) — one mean per column
stds = data.std(axis=0)              # shape (5,)
normalized = (data - means) / stds   # Broadcasting: (1000,5) - (5,) / (5,)
print(f"After normalization: means ~ {normalized.mean(axis=0).round(2)}")
print(f"                     stds  ~ {normalized.std(axis=0).round(2)}")`}
        />

        <h3>Vectorization vs Loops</h3>
        <CodeBlock
          language="python"
          title="vectorization_benchmark.py"
          code={`import numpy as np
import time

n = 1_000_000
a = np.random.randn(n)
b = np.random.randn(n)

# --- SLOW: Python loop ---
start = time.perf_counter()
result_loop = 0.0
for i in range(n):
    result_loop += a[i] * b[i]
loop_time = time.perf_counter() - start

# --- FAST: Vectorized ---
start = time.perf_counter()
result_vec = np.dot(a, b)
vec_time = time.perf_counter() - start

print(f"Loop:       {loop_time:.4f}s  result={result_loop:.4f}")
print(f"Vectorized: {vec_time:.6f}s  result={result_vec:.4f}")
print(f"Speedup:    {loop_time / vec_time:.0f}x")
# Typical output: ~100-200x speedup

# --- Practical vectorized operations ---
# Euclidean distance matrix between 500 points in 10D
X = np.random.randn(500, 10)

# Broadcasting trick: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
sq_norms = np.sum(X ** 2, axis=1)                     # (500,)
dist_matrix = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T  # (500, 500)
dist_matrix = np.sqrt(np.maximum(dist_matrix, 0))     # Clamp negatives from float error
print(f"Distance matrix shape: {dist_matrix.shape}")
print(f"Mean pairwise distance: {dist_matrix.mean():.2f}")`}
        />

        <h3>Indexing, Slicing, and Boolean Masks</h3>
        <CodeBlock
          language="python"
          title="advanced_indexing.py"
          code={`import numpy as np

data = np.random.randn(100, 4)

# --- Boolean indexing (filtering) ---
# Select rows where column 0 > 1.0
mask = data[:, 0] > 1.0
filtered = data[mask]
print(f"Rows with col0 > 1.0: {filtered.shape[0]}")

# --- np.where: conditional selection ---
# Replace negative values with 0
clean = np.where(data > 0, data, 0)

# --- Fancy indexing ---
indices = np.array([0, 5, 10, 50])
subset = data[indices]                # Select specific rows

# --- Combining conditions ---
mask = (data[:, 0] > 0) & (data[:, 1] < 0)  # Note: & not 'and'
print(f"Rows matching compound condition: {mask.sum()}")

# --- argsort: indices that would sort an array ---
scores = np.random.randn(10)
sorted_indices = np.argsort(scores)[::-1]   # Descending order
print(f"Top 3 indices: {sorted_indices[:3]}")
print(f"Top 3 values: {scores[sorted_indices[:3]]}")

# --- np.unique: count unique values ---
labels = np.random.choice(['cat', 'dog', 'bird'], size=100)
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u}: {c}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always vectorize</strong>: If you write a for-loop over array elements in NumPy, you are almost certainly doing it wrong. Rewrite with broadcasting, <code>np.where</code>, or fancy indexing.</li>
          <li><strong>Use the right dtype</strong>: <code>float32</code> uses half the memory of <code>float64</code> and is sufficient for most ML. For integer IDs, use <code>int32</code> instead of <code>int64</code>.</li>
          <li><strong>Preallocate, don&apos;t append</strong>: Never build arrays by appending in a loop. Preallocate with <code>np.empty</code> or <code>np.zeros</code> and fill in values.</li>
          <li><strong>Views vs copies</strong>: Slicing returns a <em>view</em> (shared memory). Boolean/fancy indexing returns a <em>copy</em>. Use <code>.copy()</code> explicitly when you need a copy to avoid accidental mutations.</li>
          <li><strong>Memory layout matters</strong>: C-order (row-major, default) vs Fortran-order (column-major). Operations along the last axis are fastest in C-order. Use <code>np.ascontiguousarray</code> if you need guaranteed layout.</li>
          <li><strong>Avoid temporary arrays</strong>: <code>a * b + c</code> creates two temporaries. For large arrays, use <code>np.multiply(a, b, out=result); np.add(result, c, out=result)</code> to reduce memory allocations.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Confusing <code>*</code> with <code>@</code></strong>: <code>A * B</code> is element-wise multiplication. <code>A @ B</code> is matrix multiplication. This is the #1 source of shape errors for beginners.</li>
          <li><strong>Modifying a view unintentionally</strong>: <code>b = a[::2]</code> creates a view. Modifying <code>b</code> modifies <code>a</code> too. Use <code>b = a[::2].copy()</code> if you need independence.</li>
          <li><strong>Using <code>==</code> for float comparison</strong>: Floating point arithmetic is not exact. Use <code>np.allclose(a, b)</code> or <code>np.isclose(a, b)</code> instead of <code>a == b</code>.</li>
          <li><strong>Forgetting axis parameter</strong>: <code>np.sum(matrix)</code> sums ALL elements. <code>np.sum(matrix, axis=0)</code> sums down rows (per column). <code>axis=1</code> sums across columns (per row). Get this wrong and your shapes silently break.</li>
          <li><strong>Broadcasting shape mismatches</strong>: A shape <code>(3,)</code> vector broadcasts with <code>(5, 3)</code> but NOT with <code>(5, 4)</code>. The error message &quot;operands could not be broadcast together&quot; means you violated the trailing-dimension rule.</li>
          <li><strong>Integer overflow</strong>: <code>np.array([200], dtype=np.int8)</code> silently overflows to <code>-56</code>. Always check your dtypes when working with large numbers.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Given a matrix <InlineMath math="X" /> of shape <code>(n, d)</code> representing n data points in d dimensions, compute the pairwise Euclidean distance matrix without any Python loops.</p>
        <p><strong>Answer:</strong></p>
        <p>
          We use the identity <InlineMath math="\|x_i - x_j\|^2 = \|x_i\|^2 + \|x_j\|^2 - 2 x_i \cdot x_j" /> and broadcasting:
        </p>
        <CodeBlock
          language="python"
          title="pairwise_distance.py"
          code={`import numpy as np

def pairwise_distances(X):
    """Compute Euclidean distance matrix without loops.

    Args:
        X: array of shape (n, d)
    Returns:
        D: array of shape (n, n) where D[i,j] = ||X[i] - X[j]||
    """
    # Step 1: Compute squared norms ||x_i||^2 for each row
    sq_norms = np.sum(X ** 2, axis=1)           # shape (n,)

    # Step 2: Compute gram matrix X @ X.T
    gram = X @ X.T                               # shape (n, n)

    # Step 3: Use broadcasting for ||x_i - x_j||^2
    #   sq_norms[:, None] has shape (n, 1) — broadcasts across columns
    #   sq_norms[None, :] has shape (1, n) — broadcasts across rows
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * gram

    # Step 4: Clamp numerical errors and take sqrt
    return np.sqrt(np.maximum(sq_dists, 0))

# Verify
X = np.random.randn(5, 3)
D = pairwise_distances(X)
print(D.round(2))
# Check: D[i,i] should be 0, D[i,j] == D[j,i]
assert np.allclose(np.diag(D), 0)
assert np.allclose(D, D.T)`}
        />
        <p>
          <strong>Key insight</strong>: This avoids an <InlineMath math="O(n^2)" /> Python loop by leveraging broadcasting and
          a single <InlineMath math="O(n^2 d)" /> matrix multiplication. The approach scales to thousands of points efficiently
          and is exactly what <code>sklearn.metrics.pairwise_distances</code> does internally.
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>NumPy documentation — Broadcasting</strong> — The official guide with visual diagrams of how shapes align.</li>
          <li><strong>From Python to NumPy (Nicolas Rougier)</strong> — Free online book focused on vectorization techniques and eliminating loops.</li>
          <li><strong>NumPy Internals (strides, memory layout)</strong> — Understanding how strides enable views, transposition without copying, and advanced memory tricks.</li>
          <li><strong>BLAS/LAPACK</strong> — The Fortran libraries that power <code>np.linalg</code>. Understanding these helps explain why NumPy is fast and where the limits are.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
