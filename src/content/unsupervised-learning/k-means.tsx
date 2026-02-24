"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function KMeansClustering() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          K-means is the simplest and most widely used clustering algorithm. Given <InlineMath math="k" /> (the number of clusters you want), it finds <InlineMath math="k" /> cluster centers and assigns each data point to the nearest center. The algorithm iterates between two steps: <strong>assign</strong> each point to its closest center, then <strong>update</strong> each center to be the mean of its assigned points. Repeat until nothing changes.
        </p>
        <p>
          The beauty of k-means is its simplicity — you can explain it in 30 seconds and implement it in 20 lines of code. It converges quickly (usually in 10-20 iterations) and scales to massive datasets. The catch: you have to choose <InlineMath math="k" /> in advance, it only finds spherical (globular) clusters, it&apos;s sensitive to initialization, and it&apos;s sensitive to outliers (because it uses means).
        </p>
        <p>
          The <strong>elbow method</strong> helps choose <InlineMath math="k" />: plot the total within-cluster distance (inertia) for different values of <InlineMath math="k" /> and look for the &quot;elbow&quot; where adding more clusters stops providing much improvement. The <strong>silhouette score</strong> provides a more rigorous alternative — it measures how similar each point is to its own cluster compared to the nearest other cluster.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Objective: Minimize Within-Cluster Sum of Squares (WCSS)</h3>
        <BlockMath math="J = \sum_{j=1}^{k}\sum_{\mathbf{x}_i \in C_j} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2" />
        <p>where <InlineMath math="C_j" /> is the set of points assigned to cluster <InlineMath math="j" /> and <InlineMath math="\boldsymbol{\mu}_j" /> is the centroid of cluster <InlineMath math="j" />.</p>

        <h3>Lloyd&apos;s Algorithm</h3>
        <p><strong>Step 1 (Assignment):</strong> Assign each point to its nearest centroid:</p>
        <BlockMath math="c_i = \arg\min_{j \in \{1,\dots,k\}} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2" />
        <p><strong>Step 2 (Update):</strong> Recompute centroids:</p>
        <BlockMath math="\boldsymbol{\mu}_j = \frac{1}{|C_j|}\sum_{\mathbf{x}_i \in C_j} \mathbf{x}_i" />
        <p>Repeat until convergence. Each step is guaranteed to decrease (or maintain) <InlineMath math="J" />, so the algorithm always converges — but possibly to a <strong>local minimum</strong>.</p>

        <h3>Silhouette Score</h3>
        <p>For each point <InlineMath math="i" />:</p>
        <BlockMath math="s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}" />
        <p>where <InlineMath math="a(i)" /> = mean distance to other points in the same cluster, <InlineMath math="b(i)" /> = mean distance to points in the nearest other cluster. Range: [-1, 1]. Higher is better.</p>

        <h3>k-Means++ Initialization</h3>
        <p>Instead of random initialization, choose initial centroids that are spread out:</p>
        <BlockMath math="P(\mathbf{x}_i \text{ chosen as next centroid}) \propto D(\mathbf{x}_i)^2" />
        <p>where <InlineMath math="D(\mathbf{x}_i)" /> is the distance to the nearest existing centroid. This gives an <InlineMath math="O(\log k)" />-competitive approximation to the optimal clustering.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>From Scratch</h3>
        <CodeBlock
          language="python"
          title="kmeans_scratch.py"
          code={`import numpy as np

class KMeansScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-6):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        n, d = X.shape

        # k-means++ initialization
        centroids = [X[np.random.randint(n)]]
        for _ in range(1, self.k):
            dists = np.min([np.sum((X - c)**2, axis=1) for c in centroids], axis=0)
            probs = dists / dists.sum()
            centroids.append(X[np.random.choice(n, p=probs)])
        self.centroids = np.array(centroids)

        for iteration in range(self.max_iters):
            # Assignment step
            dists = np.array([np.sum((X - c)**2, axis=1) for c in self.centroids])
            self.labels = np.argmin(dists, axis=0)

            # Update step
            new_centroids = np.array([
                X[self.labels == j].mean(axis=0)
                if np.sum(self.labels == j) > 0 else self.centroids[j]
                for j in range(self.k)
            ])

            # Check convergence
            shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            if shift < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break

        self.inertia_ = sum(
            np.sum((X[self.labels == j] - self.centroids[j])**2)
            for j in range(self.k)
        )
        return self

# Example
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + [3, 3],
    np.random.randn(100, 2) + [-3, -3],
    np.random.randn(100, 2) + [3, -3],
])

km = KMeansScratch(k=3)
km.fit(X)
print(f"Inertia: {km.inertia_:.2f}")
print(f"Centroids:\\n{km.centroids}")`}
        />

        <h3>With scikit-learn: Elbow Method and Silhouette Analysis</h3>
        <CodeBlock
          language="python"
          title="kmeans_sklearn.py"
          code={`from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Always scale before k-means!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Elbow method ---
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.4f}")

best_k = K_range[np.argmax(silhouettes)]
print(f"\\nBest k by silhouette: {best_k}")

# --- Final model ---
final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
final.fit(X_scaled)
print(f"\\nCluster sizes: {np.bincount(final.labels_)}")
print(f"Centroids (scaled):\\n{final.cluster_centers_}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always scale features</strong>: K-means uses Euclidean distance. If one feature ranges 0-1 and another 0-1000, the second dominates all distance calculations.</li>
          <li><strong>Run multiple initializations</strong>: K-means converges to local minima. <code>n_init=10</code> (default) runs 10 times and keeps the best result. For important applications, increase to 20-50.</li>
          <li><strong>Use Mini-Batch K-Means for large data</strong>: <code>MiniBatchKMeans</code> uses random subsets per iteration, scaling to millions of points with minimal quality loss.</li>
          <li><strong>K-means assumes spherical clusters</strong>: If your clusters are elongated, overlapping, or have different densities, consider DBSCAN, Gaussian Mixture Models, or spectral clustering.</li>
          <li><strong>Combine elbow + silhouette</strong>: The elbow can be ambiguous. Use silhouette score as a tiebreaker. Also consider domain knowledge — sometimes the right <InlineMath math="k" /> is determined by the business question, not the data.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Not scaling features</strong>: This is the most common error. K-means on unscaled data produces meaningless clusters dominated by high-magnitude features.</li>
          <li><strong>Choosing k by inertia alone</strong>: Inertia always decreases with more clusters. Don&apos;t just pick the &quot;elbow&quot; — validate with silhouette scores, domain knowledge, or downstream task performance.</li>
          <li><strong>Interpreting cluster assignments as ground truth</strong>: K-means will <em>always</em> find clusters, even in random noise. Validate that clusters are meaningful (check cluster stability, use gap statistic).</li>
          <li><strong>Using k-means on categorical data</strong>: Euclidean distance is meaningless for categories. Use k-modes or k-prototypes instead.</li>
          <li><strong>Assuming equal cluster sizes</strong>: K-means tends to create roughly equal-sized clusters. If true cluster sizes vary widely, consider Gaussian Mixture Models.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> K-means converged, but you suspect the result is suboptimal. How can you tell, and what can you do about it?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>How to detect</strong>: Run k-means multiple times with different random initializations. If you get very different inertia values or cluster assignments, the algorithm is hitting different local minima.</li>
          <li><strong>Solution 1 — Multiple restarts</strong>: Increase <code>n_init</code> (e.g., to 50). Sklearn picks the run with the lowest inertia.</li>
          <li><strong>Solution 2 — k-means++ initialization</strong>: This is sklearn&apos;s default (<code>init=&quot;k-means++&quot;</code>). It spreads initial centroids to avoid bad starting positions, which often reaches the global optimum.</li>
          <li><strong>Solution 3 — Check cluster quality</strong>: Compute silhouette scores for individual points. Points with negative silhouette scores are likely assigned to the wrong cluster.</li>
          <li><strong>Solution 4 — Try different algorithms</strong>: If k-means consistently produces poor results, the data may not have spherical clusters. Switch to Gaussian Mixture Models (soft assignments), DBSCAN (density-based), or spectral clustering (graph-based).</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Lloyd (1982) &quot;Least Squares Quantization in PCM&quot;</strong> — The original k-means paper (originally written in 1957!).</li>
          <li><strong>Arthur &amp; Vassilvitskii (2007) &quot;k-means++: The Advantages of Careful Seeding&quot;</strong> — Proves the O(log k) guarantee for smart initialization.</li>
          <li><strong>Tibshirani et al. (2001) &quot;Estimating the Number of Clusters via the Gap Statistic&quot;</strong> — A principled method for choosing k.</li>
          <li><strong>scikit-learn Clustering guide</strong> — Comparison of all clustering algorithms with visual examples.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
