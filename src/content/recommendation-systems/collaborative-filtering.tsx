"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function CollaborativeFiltering() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Collaborative filtering is the idea that <strong>people who agreed in the past will agree in the
          future</strong>. If you and I both loved the same five movies, and I loved a sixth movie you
          haven&apos;t seen, the system will recommend that sixth movie to you. No one needs to analyze the
          movie content — the patterns of user behavior are enough.
        </p>
        <p>
          There are two main flavors. <strong>User-based</strong> collaborative filtering finds users similar
          to you and recommends what they liked. <strong>Item-based</strong> collaborative filtering finds items
          similar to ones you already liked and recommends those. In practice, item-based tends to work better
          because item similarities are more stable than user similarities (a user&apos;s tastes change over
          time, but the similarity between two movies doesn&apos;t).
        </p>
        <p>
          The most powerful version is <strong>matrix factorization</strong>: decompose the giant user-item
          ratings matrix into two smaller matrices — one that gives each user a vector of latent preferences,
          and one that gives each item a vector of latent attributes. A user&apos;s predicted rating for an
          item is just the dot product of their two vectors. This is what powered Netflix&apos;s recommendation
          engine and won the famous Netflix Prize.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>User-Item Rating Matrix</h3>
        <p>
          We have <InlineMath math="m" /> users and <InlineMath math="n" /> items. The rating matrix
          <InlineMath math="R \in \mathbb{R}^{m \times n}" /> has entry <InlineMath math="r_{ui}" /> if
          user <InlineMath math="u" /> rated item <InlineMath math="i" />, and is missing otherwise. The goal
          is to fill in the missing entries.
        </p>

        <h3>Similarity Measures</h3>
        <p><strong>Cosine similarity</strong> between users <InlineMath math="u" /> and <InlineMath math="v" />:</p>
        <BlockMath math="\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} r_{ui} \cdot r_{vi}}{\sqrt{\sum_{i \in I_{uv}} r_{ui}^2} \cdot \sqrt{\sum_{i \in I_{uv}} r_{vi}^2}}" />
        <p>where <InlineMath math="I_{uv}" /> is the set of items rated by both users.</p>

        <p><strong>Pearson correlation</strong> (centered cosine):</p>
        <BlockMath math="\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}" />

        <h3>Prediction (User-Based)</h3>
        <p>Predict user <InlineMath math="u" />&apos;s rating for item <InlineMath math="i" /> using the <InlineMath math="k" /> most similar users who rated item <InlineMath math="i" />:</p>
        <BlockMath math="\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} \text{sim}(u, v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |\text{sim}(u, v)|}" />

        <h3>Matrix Factorization</h3>
        <p>Decompose the rating matrix into user and item latent factors:</p>
        <BlockMath math="R \approx P Q^T, \quad P \in \mathbb{R}^{m \times k}, \; Q \in \mathbb{R}^{n \times k}" />
        <p>Each user <InlineMath math="u" /> has a latent vector <InlineMath math="p_u \in \mathbb{R}^k" />, each item <InlineMath math="i" /> has <InlineMath math="q_i \in \mathbb{R}^k" />:</p>
        <BlockMath math="\hat{r}_{ui} = p_u^T q_i + b_u + b_i + \mu" />
        <p>where <InlineMath math="\mu" /> is the global mean, <InlineMath math="b_u" /> is user bias, and <InlineMath math="b_i" /> is item bias.</p>

        <h3>Optimization Objective</h3>
        <p>Minimize regularized squared error over observed ratings:</p>
        <BlockMath math="\min_{P, Q, b} \sum_{(u,i) \in \Omega} (r_{ui} - \hat{r}_{ui})^2 + \lambda(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2)" />
        <p>Solved via <strong>SGD</strong> or <strong>Alternating Least Squares (ALS)</strong>.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Memory-Based Collaborative Filtering</h3>
        <CodeBlock
          language="python"
          title="memory_based_cf.py"
          code={`import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item rating matrix (0 = not rated)
R = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 0, 1, 0],
    [1, 1, 0, 5, 0],
    [0, 0, 5, 4, 0],
    [0, 3, 4, 0, 5],
])

def user_based_cf(R, user, item, k=2):
    """Predict rating using k most similar users."""
    # Compute user similarity (Pearson = centered cosine)
    means = np.where(R > 0, R, np.nan)
    user_means = np.nanmean(means, axis=1)

    # Center the ratings
    R_centered = R.copy().astype(float)
    for u in range(R.shape[0]):
        mask = R[u] > 0
        R_centered[u, mask] -= user_means[u]
        R_centered[u, ~mask] = 0

    # Cosine similarity on centered ratings = Pearson
    sim = cosine_similarity(R_centered)
    np.fill_diagonal(sim, 0)  # exclude self

    # Find users who rated this item
    rated_mask = R[:, item] > 0
    rated_mask[user] = False  # exclude target user

    # Get k most similar users who rated the item
    sim_scores = sim[user] * rated_mask
    top_k = np.argsort(sim_scores)[-k:]
    top_k = top_k[sim_scores[top_k] > 0]  # keep only positive similarity

    if len(top_k) == 0:
        return user_means[user]

    # Weighted average of deviations
    weights = sim[user, top_k]
    deviations = R[top_k, item] - user_means[top_k]
    pred = user_means[user] + np.dot(weights, deviations) / (np.abs(weights).sum() + 1e-8)
    return pred

# Predict user 1's rating for item 2
pred = user_based_cf(R, user=1, item=2, k=2)
print(f"Predicted rating for User 1, Item 2: {pred:.2f}")`}
        />

        <h3>Matrix Factorization with SGD</h3>
        <CodeBlock
          language="python"
          title="matrix_factorization.py"
          code={`import numpy as np

class MatrixFactorization:
    def __init__(self, n_users, n_items, k=20, lr=0.005, reg=0.02):
        self.k = k
        self.lr = lr
        self.reg = reg

        # Initialize latent factors
        self.P = np.random.normal(0, 0.1, (n_users, k))  # user factors
        self.Q = np.random.normal(0, 0.1, (n_items, k))  # item factors
        self.b_u = np.zeros(n_users)                       # user bias
        self.b_i = np.zeros(n_items)                       # item bias
        self.mu = 0                                        # global mean

    def fit(self, ratings, epochs=50):
        """
        ratings: list of (user, item, rating) tuples
        """
        self.mu = np.mean([r for _, _, r in ratings])

        for epoch in range(epochs):
            np.random.shuffle(ratings)
            total_loss = 0

            for u, i, r in ratings:
                # Prediction
                pred = self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
                error = r - pred

                # SGD updates
                self.b_u[u] += self.lr * (error - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (error - self.reg * self.b_i[i])

                P_u_old = self.P[u].copy()
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * P_u_old - self.reg * self.Q[i])

                total_loss += error ** 2

            rmse = np.sqrt(total_loss / len(ratings))
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | RMSE: {rmse:.4f}")

    def predict(self, user, item):
        return self.mu + self.b_u[user] + self.b_i[item] + self.P[user] @ self.Q[item]

# Example usage
R = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 0, 1, 0],
    [1, 1, 0, 5, 0],
    [0, 0, 5, 4, 0],
    [0, 3, 4, 0, 5],
])

# Convert to (user, item, rating) tuples
ratings = [(u, i, R[u, i]) for u in range(R.shape[0])
           for i in range(R.shape[1]) if R[u, i] > 0]

mf = MatrixFactorization(n_users=5, n_items=5, k=3)
mf.fit(ratings, epochs=100)

# Predict missing entries
for u in range(5):
    for i in range(5):
        if R[u, i] == 0:
            print(f"User {u}, Item {i}: {mf.predict(u, i):.2f}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Cold start problem</strong>: New users/items have no interaction data. Solutions: use content-based features for new items, ask new users to rate a few items, or use hybrid methods that combine collaborative and content-based signals.</li>
          <li><strong>Implicit vs explicit feedback</strong>: Explicit ratings are rare. Most data is implicit (clicks, views, purchases). For implicit feedback, use <strong>ALS with weighted regularization</strong> (Hu et al., 2008) or <strong>BPR (Bayesian Personalized Ranking)</strong>.</li>
          <li><strong>Latent dimension <InlineMath math="k" /></strong>: Typical values are 50-200. Too small and the model can&apos;t capture complex preferences. Too large and it overfits (especially with sparse data).</li>
          <li><strong>ALS vs SGD</strong>: ALS is easily parallelizable (used in Spark MLlib) and works well for implicit feedback. SGD is more flexible and handles streaming data better.</li>
          <li><strong>At scale</strong>: Netflix, Spotify, and Amazon use matrix factorization as a baseline component in ensemble systems. The final system typically blends many models.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Evaluating on all entries</strong>: You can only evaluate on held-out <em>observed</em> ratings. Predicting that a user would rate all unseen items as 1 might minimize global error but gives terrible recommendations.</li>
          <li><strong>Treating missing entries as zeros</strong>: A missing rating does NOT mean the user dislikes the item — they just haven&apos;t seen it. Only optimize over observed ratings. This is crucial and a very common error.</li>
          <li><strong>Ignoring popularity bias</strong>: Without biases (<InlineMath math="b_u, b_i, \mu" />), the model tries to explain a generous user&apos;s high ratings purely through latent factors. Always include bias terms.</li>
          <li><strong>Using RMSE as the only metric</strong>: RMSE measures rating prediction accuracy. But for recommendations, ranking metrics (NDCG, MAP, Hit Rate) matter more — you care about the top-10, not predicting every rating perfectly.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> A new e-commerce site has 1M users, 100K products, and sparse purchase history. Design a recommendation system. How do you handle the cold start problem?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Base model: Matrix Factorization</strong>:
            <ul>
              <li>Treat purchases as implicit feedback (binary: interacted or not).</li>
              <li>Use <strong>ALS with implicit feedback</strong> (confidence-weighted). Confidence increases with interaction count.</li>
              <li>Train with <InlineMath math="k = 100" /> latent factors, regularization tuned via validation.</li>
            </ul>
          </li>
          <li><strong>Cold start for new items</strong>:
            <ul>
              <li>Use item content features (category, description embeddings, price) to estimate the item&apos;s latent vector.</li>
              <li>Train a mapping: <InlineMath math="q_i = f_\theta(\text{features}_i)" /> where <InlineMath math="f_\theta" /> is a small neural network.</li>
              <li>Once the item gets enough interactions, switch to learned latent factors.</li>
            </ul>
          </li>
          <li><strong>Cold start for new users</strong>:
            <ul>
              <li>Show popular items (popularity baseline) until the user has enough interactions.</li>
              <li>Use demographic features or browsing behavior for initial personalization.</li>
              <li>After 5-10 interactions, collaborative filtering becomes effective.</li>
            </ul>
          </li>
          <li><strong>Evaluation</strong>:
            <ul>
              <li>Use offline ranking metrics: NDCG@10, Hit Rate@10 on held-out interactions.</li>
              <li>A/B test online with click-through rate and conversion rate.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Koren et al. (2009) &quot;Matrix Factorization Techniques for Recommender Systems&quot;</strong> — The Netflix Prize paper. The best overview of matrix factorization for RecSys.</li>
          <li><strong>Hu et al. (2008) &quot;Collaborative Filtering for Implicit Feedback Datasets&quot;</strong> — ALS for implicit feedback, used in industry everywhere.</li>
          <li><strong>Rendle et al. (2009) &quot;BPR: Bayesian Personalized Ranking from Implicit Feedback&quot;</strong> — Pairwise ranking objective for recommendations.</li>
          <li><strong>Aggarwal &quot;Recommender Systems&quot; textbook</strong> — Comprehensive reference covering all CF approaches.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
