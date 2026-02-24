"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function TSneUmap() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          PCA is a linear method — it can only find straight-line patterns. But real data often lies on curved, twisted surfaces (manifolds) in high-dimensional space. <strong>t-SNE</strong> and <strong>UMAP</strong> are <em>nonlinear</em> dimensionality reduction methods designed specifically for <strong>visualization</strong>. They excel at preserving the local neighborhood structure: points that are close in the original space stay close in the 2D plot.
        </p>
        <p>
          <strong>t-SNE</strong> (t-distributed Stochastic Neighbor Embedding) works by converting distances between points into probabilities. In high-dimensional space, nearby points get high probability; far points get low probability. It then finds a 2D layout where the same probability relationships hold as closely as possible. The &quot;t-distribution&quot; part is crucial — it uses heavy tails in the low-dimensional space, which prevents the &quot;crowding problem&quot; where all points collapse into a ball.
        </p>
        <p>
          <strong>UMAP</strong> (Uniform Manifold Approximation and Projection) is the modern successor. It&apos;s faster (minutes vs hours on large datasets), preserves more global structure (relationships between clusters, not just within), and has a rigorous mathematical foundation in topological data analysis. For most practical purposes, <strong>UMAP has replaced t-SNE</strong> as the go-to visualization tool.
        </p>
        <p>
          Critical caveat: both methods are for <strong>visualization only</strong>. The axes of a t-SNE or UMAP plot have no meaning. Distances between clusters can be misleading. Cluster sizes are unreliable. Never use these for quantitative analysis — use them to build intuition and generate hypotheses.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>t-SNE: High-Dimensional Affinities</h3>
        <p>Convert distances to conditional probabilities using a Gaussian kernel:</p>
        <BlockMath math="p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}" />
        <p>Symmetrize: <InlineMath math="p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}" /></p>
        <p>The bandwidth <InlineMath math="\sigma_i" /> is set so that the entropy of the distribution equals <InlineMath math="\log(\text{perplexity})" />. Perplexity is the effective number of neighbors (typically 5-50).</p>

        <h3>t-SNE: Low-Dimensional Affinities</h3>
        <p>In 2D, use a Student-t distribution with 1 degree of freedom (Cauchy distribution):</p>
        <BlockMath math="q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}" />
        <p>The heavy tails of the t-distribution allow nearby points to be modeled faithfully while giving far-apart points room to spread out.</p>

        <h3>t-SNE: Objective</h3>
        <p>Minimize the KL divergence between the two distributions:</p>
        <BlockMath math="C = KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}" />
        <p>Optimized via gradient descent on the 2D coordinates <InlineMath math="\mathbf{y}_i" />.</p>

        <h3>UMAP: Key Differences</h3>
        <p>UMAP models the data as a fuzzy topological structure (simplicial complex):</p>
        <BlockMath math="p_{ij} = p_{i|j} + p_{j|i} - p_{i|j} \cdot p_{j|i}" />
        <p>where <InlineMath math="p_{i|j} = \exp(-(d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i) / \sigma_i)" /> and <InlineMath math="\rho_i" /> is the distance to the nearest neighbor.</p>
        <p>Low-dimensional affinities:</p>
        <BlockMath math="q_{ij} = \left(1 + a\|\mathbf{y}_i - \mathbf{y}_j\|^{2b}\right)^{-1}" />
        <p>Objective: binary cross-entropy instead of KL divergence:</p>
        <BlockMath math="C = \sum_{i \neq j}\left[p_{ij}\log\frac{p_{ij}}{q_{ij}} + (1 - p_{ij})\log\frac{1-p_{ij}}{1-q_{ij}}\right]" />
        <p>The second term (repulsive force) is what helps UMAP preserve global structure better than t-SNE.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>t-SNE with scikit-learn</h3>
        <CodeBlock
          language="python"
          title="tsne_example.py"
          code={`from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load MNIST digits (subset for speed)
X_full, y_full = fetch_openml("mnist_784", version=1, return_X_y=True,
                              as_frame=False)
n = 5000
idx = np.random.RandomState(42).choice(len(X_full), n, replace=False)
X, y = X_full[idx], y_full[idx]

# Normalize pixel values
X = X / 255.0

# t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,        # effective number of neighbors (try 5-50)
    learning_rate="auto",  # auto is recommended
    n_iter=1000,
    init="pca",           # PCA initialization for stability
    random_state=42,
)
X_tsne = tsne.fit_transform(X)
print(f"t-SNE KL divergence: {tsne.kl_divergence_:.4f}")
print(f"Output shape: {X_tsne.shape}")

# Plot (conceptual — you'd use matplotlib)
# Each digit class should form a distinct cluster
for digit in range(10):
    mask = y == str(digit)
    print(f"Digit {digit}: center = ({X_tsne[mask, 0].mean():.1f}, "
          f"{X_tsne[mask, 1].mean():.1f})")`}
        />

        <h3>UMAP</h3>
        <CodeBlock
          language="python"
          title="umap_example.py"
          code={`import umap

# UMAP — faster and better global structure
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,        # similar to perplexity (local vs global)
    min_dist=0.1,          # how tightly points cluster (0.0 to 0.99)
    metric="euclidean",    # also supports cosine, manhattan, etc.
    random_state=42,
)
X_umap = reducer.fit_transform(X)
print(f"Output shape: {X_umap.shape}")

# UMAP can also do supervised dimensionality reduction
X_umap_supervised = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    random_state=42,
).fit_transform(X, y=y.astype(int))  # uses labels to guide embedding

# UMAP can transform new data (t-SNE cannot!)
X_new = X_full[5000:5100] / 255.0
X_new_embedded = reducer.transform(X_new)
print(f"New data embedded: {X_new_embedded.shape}")

# Speed comparison on 70K samples:
# t-SNE: ~10-30 minutes
# UMAP: ~1-3 minutes`}
        />

        <h3>Comparing Methods Side by Side</h3>
        <CodeBlock
          language="python"
          title="comparison.py"
          code={`from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import time

methods = {
    "PCA": PCA(n_components=2),
    "t-SNE": TSNE(n_components=2, random_state=42, init="pca",
                  learning_rate="auto"),
    "UMAP": umap.UMAP(n_components=2, random_state=42),
}

for name, method in methods.items():
    start = time.time()
    X_reduced = method.fit_transform(X)
    elapsed = time.time() - start
    print(f"{name:8s}: {elapsed:.2f}s, shape={X_reduced.shape}")

# Typical output:
# PCA     : 0.02s  — linear, preserves global structure, misses nonlinear
# t-SNE   : 45.3s  — nonlinear, great clusters, no global structure
# UMAP    : 8.1s   — nonlinear, good clusters + some global structure`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Use PCA first to reduce to 50 dimensions</strong>: Both t-SNE and UMAP benefit from PCA preprocessing on high-dimensional data. It reduces noise and speeds up computation dramatically.</li>
          <li><strong>UMAP is the default choice</strong>: It&apos;s faster, preserves more global structure, can transform new points, and supports supervised mode. Use t-SNE only if you need to compare with older literature.</li>
          <li><strong>Perplexity / n_neighbors controls local vs global</strong>: Low values (5-15) emphasize local structure (tight clusters). High values (50-200) show more global relationships. Always try multiple values.</li>
          <li><strong>min_dist in UMAP</strong>: Low values (0.0-0.1) create tighter, more separated clusters. High values (0.5-0.99) spread points out more evenly.</li>
          <li><strong>Run multiple random seeds</strong>: Both methods are stochastic. Different runs may produce different layouts. If clusters are consistent across runs, they&apos;re real.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Interpreting distances between clusters</strong>: In t-SNE, the distance between two clusters is meaningless. Two well-separated clusters in t-SNE might actually be close in the original space, or vice versa. UMAP is slightly better at preserving inter-cluster distances, but still unreliable.</li>
          <li><strong>Interpreting cluster sizes</strong>: t-SNE tends to make all clusters roughly the same size, regardless of their true density. Don&apos;t conclude that one group is larger than another based on the plot.</li>
          <li><strong>Using t-SNE/UMAP for clustering</strong>: These are visualization tools, not clustering algorithms. Run clustering (k-means, DBSCAN) in the original high-dimensional space, then <em>visualize</em> the results with t-SNE/UMAP.</li>
          <li><strong>Using a single perplexity/n_neighbors value</strong>: The results can change dramatically. Always explore a range and report which value you used.</li>
          <li><strong>Applying t-SNE to new data</strong>: Standard t-SNE has no <code>transform</code> method — it can&apos;t embed new points. UMAP can. If you need out-of-sample embeddings, use UMAP or parametric t-SNE.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You show a t-SNE plot to your manager and they say: &quot;Great, cluster A is twice as big as cluster B, and they&apos;re very far apart, so they&apos;re very different.&quot; What&apos;s wrong with this interpretation?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Cluster size is unreliable</strong>: t-SNE normalizes local densities, which means it tends to expand dense clusters and compress sparse ones. A cluster that looks &quot;twice as big&quot; in the plot may actually contain fewer points or occupy less volume in the original space.</li>
          <li><strong>Inter-cluster distances are meaningless</strong>: t-SNE optimizes local neighborhood preservation, not global distances. Two clusters that appear far apart might actually be close in the original space. The distance depends heavily on perplexity and random initialization.</li>
          <li><strong>What you <em>can</em> say</strong>: The data has two distinct groups (the fact that they form separate clusters is meaningful). Points within each cluster are genuinely similar to each other.</li>
          <li><strong>Better approach</strong>: Quantify the actual distance between clusters in the original feature space (e.g., Euclidean distance between centroids). Report the actual cluster sizes. Use t-SNE only as a qualitative visualization, not as evidence for quantitative claims.</li>
          <li><strong>UMAP is better but not perfect</strong>: UMAP preserves more global structure than t-SNE, so inter-cluster distances are more <em>relatively</em> meaningful (farther in UMAP usually means farther in reality), but still shouldn&apos;t be used for precise quantitative comparisons.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>van der Maaten &amp; Hinton (2008) &quot;Visualizing Data using t-SNE&quot;</strong> — The original t-SNE paper.</li>
          <li><strong>McInnes et al. (2018) &quot;UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction&quot;</strong> — The UMAP paper, grounded in topological data analysis.</li>
          <li><strong>Wattenberg et al. (2016) &quot;How to Use t-SNE Effectively&quot;</strong> — Interactive Distill article showing how hyperparameters affect results. Essential reading.</li>
          <li><strong>Kobak &amp; Berens (2019) &quot;The art of using t-SNE for single-cell transcriptomics&quot;</strong> — Practical guidelines from the bioinformatics community.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
