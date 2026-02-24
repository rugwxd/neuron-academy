"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function DeepRecsys() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Matrix factorization works well, but it has a fundamental limitation: it models user-item interactions
          as a <strong>linear dot product</strong> of latent vectors. This means it can only capture linear
          relationships between user preferences and item attributes. In reality, preferences are highly nonlinear
          — a user who loves both action movies and romantic comedies might hate action-romance hybrids. A dot
          product cannot express this kind of complex interaction.
        </p>
        <p>
          <strong>Deep learning for recommendation systems</strong> replaces the dot product with neural networks
          that can learn arbitrary nonlinear interaction patterns. The simplest approach, <strong>Neural Collaborative
          Filtering (NCF)</strong>, feeds user and item embeddings through an MLP instead of taking their dot product.
          This lets the model learn complex, non-additive relationships between user preferences and item attributes
          that matrix factorization fundamentally cannot capture.
        </p>
        <p>
          At scale, modern recommendation systems use a <strong>two-tower architecture</strong>: one neural network
          (tower) encodes the user, and another encodes the item. Each tower produces an embedding vector, and
          candidates are retrieved by approximate nearest neighbor search on these embeddings. This separates the
          problem into two stages: <strong>candidate generation</strong> (retrieve hundreds of candidates from millions
          using fast vector search) and <strong>ranking</strong> (score the candidates with a more expensive model
          that considers rich features and interactions). This is how YouTube, Netflix, and virtually every
          large-scale recommendation system works.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Neural Collaborative Filtering</h3>
        <p>
          Given user embedding <InlineMath math="e_u \in \mathbb{R}^d" /> and item
          embedding <InlineMath math="e_i \in \mathbb{R}^d" />, NCF replaces the dot product
          with an MLP:
        </p>
        <BlockMath math="\hat{y}_{ui} = \sigma\!\left(W_L^T \cdot \text{ReLU}\!\left(\ldots \text{ReLU}(W_1 \begin{bmatrix} e_u \\ e_i \end{bmatrix} + b_1) \ldots\right) + b_L\right)" />
        <p>
          where <InlineMath math="\sigma" /> is the sigmoid function for implicit feedback
          (click/no-click). The concatenation <InlineMath math="[e_u; e_i]" /> allows the MLP to learn
          arbitrary interactions, unlike the dot product which is limited to
          <InlineMath math="\sum_k e_{u,k} \cdot e_{i,k}" />.
        </p>

        <h3>Dot Product vs MLP Interaction</h3>
        <p>
          The dot product computes a fixed bilinear form:
        </p>
        <BlockMath math="\hat{y}_{ui}^{\text{dot}} = e_u^T e_i = \sum_{k=1}^{d} e_{u,k} \cdot e_{i,k}" />
        <p>
          This is a linear function of the element-wise product. An MLP on the concatenation is strictly
          more expressive — it can approximate any continuous function of the two embedding vectors (universal
          approximation theorem). In practice, a <strong>hybrid</strong> approach (GMF + MLP) often works best:
        </p>
        <BlockMath math="\hat{y}_{ui} = \sigma\!\left(W^T \begin{bmatrix} e_u^{\text{GMF}} \odot e_i^{\text{GMF}} \\ \text{MLP}(e_u^{\text{MLP}}, e_i^{\text{MLP}}) \end{bmatrix}\right)" />
        <p>
          where <InlineMath math="\odot" /> denotes element-wise product and GMF stands for Generalized
          Matrix Factorization.
        </p>

        <h3>Two-Tower Retrieval</h3>
        <p>
          The user tower <InlineMath math="f_\theta" /> and item tower <InlineMath math="g_\phi" /> produce
          embeddings independently:
        </p>
        <BlockMath math="u = f_\theta(\text{user features}), \quad v = g_\phi(\text{item features})" />
        <BlockMath math="\text{score}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}" />
        <p>
          At serving time, all item embeddings are precomputed and indexed in an ANN structure (FAISS, ScaNN).
          Given a user query, retrieve the top-<InlineMath math="k" /> items:
        </p>
        <BlockMath math="\text{candidates} = \text{ANN}(u, \{v_1, \ldots, v_N\}, k)" />
        <p>
          This is sublinear in the number of items — typically <InlineMath math="O(\log N)" /> with
          hierarchical navigable small world (HNSW) graphs.
        </p>

        <h3>Multi-Objective Ranking</h3>
        <p>
          Real recommendation systems optimize for multiple objectives simultaneously (clicks, watch time,
          likes, shares). The combined loss is a weighted sum:
        </p>
        <BlockMath math="\mathcal{L} = \sum_{j=1}^{M} w_j \cdot \mathcal{L}_j(\hat{y}_j, y_j)" />
        <p>
          where each <InlineMath math="\mathcal{L}_j" /> is a task-specific loss (binary cross-entropy for
          click, regression loss for watch time). The final ranking score combines predictions:
        </p>
        <BlockMath math="\text{rank\_score} = w_{\text{click}} \cdot P(\text{click}) + w_{\text{time}} \cdot E[\text{watch time}] + w_{\text{like}} \cdot P(\text{like})" />
      </TopicSection>

      <TopicSection type="code">
        <h3>Neural Collaborative Filtering from Scratch</h3>
        <CodeBlock
          language="python"
          title="ncf_scratch.py"
          code={`import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class InteractionDataset(Dataset):
    """Dataset of (user_id, item_id, label) tuples."""
    def __init__(self, users, items, labels):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


class NeuralCollaborativeFiltering(nn.Module):
    """
    NCF: combines GMF (dot product) and MLP pathways.
    Reference: He et al., "Neural Collaborative Filtering" (WWW 2017)
    """
    def __init__(self, n_users, n_items, embed_dim=32, mlp_dims=[64, 32, 16]):
        super().__init__()
        # GMF pathway embeddings
        self.gmf_user_embed = nn.Embedding(n_users, embed_dim)
        self.gmf_item_embed = nn.Embedding(n_items, embed_dim)

        # MLP pathway embeddings (separate from GMF)
        self.mlp_user_embed = nn.Embedding(n_users, embed_dim)
        self.mlp_item_embed = nn.Embedding(n_items, embed_dim)

        # MLP layers
        mlp_layers = []
        input_dim = embed_dim * 2  # concatenation of user + item
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(input_dim, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction layer: GMF output + MLP output -> score
        self.output = nn.Linear(embed_dim + mlp_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids, item_ids):
        # GMF pathway: element-wise product
        gmf_user = self.gmf_user_embed(user_ids)
        gmf_item = self.gmf_item_embed(item_ids)
        gmf_out = gmf_user * gmf_item  # element-wise product

        # MLP pathway: concatenate and pass through layers
        mlp_user = self.mlp_user_embed(user_ids)
        mlp_item = self.mlp_item_embed(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp(mlp_input)

        # Combine and predict
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        score = self.sigmoid(self.output(combined)).squeeze(-1)
        return score


def train_ncf(model, train_loader, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for users, items, labels in train_loader:
            preds = model(users, items)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")


# ---- Generate synthetic data ----
np.random.seed(42)
n_users, n_items = 1000, 500
n_interactions = 20000

users = np.random.randint(0, n_users, n_interactions)
items = np.random.randint(0, n_items, n_interactions)
labels = np.random.randint(0, 2, n_interactions).astype(float)

dataset = InteractionDataset(users, items, labels)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

model = NeuralCollaborativeFiltering(n_users, n_items, embed_dim=32)
train_ncf(model, loader, epochs=20)

# Predict: top items for user 0
model.eval()
with torch.no_grad():
    user_tensor = torch.LongTensor([0] * n_items)
    item_tensor = torch.arange(n_items)
    scores = model(user_tensor, item_tensor)
    top10 = torch.topk(scores, 10)
    print(f"\\nTop 10 items for user 0: {top10.indices.tolist()}")
    print(f"Scores: {[f'{s:.3f}' for s in top10.values.tolist()]}")`}
        />

        <h3>Two-Tower Model with Separate User/Item Encoders</h3>
        <CodeBlock
          language="python"
          title="two_tower.py"
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UserTower(nn.Module):
    """Encodes user features into a dense embedding."""
    def __init__(self, n_users, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, user_ids):
        x = self.user_embed(user_ids)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)  # L2 normalize for cosine similarity


class ItemTower(nn.Module):
    """Encodes item features into a dense embedding."""
    def __init__(self, n_items, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.item_embed = nn.Embedding(n_items, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, item_ids):
        x = self.item_embed(item_ids)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Two-tower retrieval model.
    User and item towers are independent -- item embeddings can be
    precomputed and indexed for fast ANN retrieval at serving time.
    """
    def __init__(self, n_users, n_items, embed_dim=64):
        super().__init__()
        self.user_tower = UserTower(n_users, embed_dim)
        self.item_tower = ItemTower(n_items, embed_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, user_ids, item_ids):
        user_emb = self.user_tower(user_ids)
        item_emb = self.item_tower(item_ids)
        # Cosine similarity scaled by learned temperature
        logits = torch.sum(user_emb * item_emb, dim=-1) / self.temperature
        return logits

    def get_user_embeddings(self, user_ids):
        return self.user_tower(user_ids)

    def get_item_embeddings(self, item_ids):
        return self.item_tower(item_ids)


def train_two_tower(model, train_data, epochs=20, lr=0.001):
    """Train with in-batch negative sampling."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_users, batch_items in train_data:
            batch_size = batch_users.size(0)

            # Compute all user and item embeddings in the batch
            user_emb = model.user_tower(batch_users)   # (B, d)
            item_emb = model.item_tower(batch_items)   # (B, d)

            # In-batch negatives: all user-item pairs in the batch
            # Similarity matrix: (B, B) -- diagonal entries are positives
            sim_matrix = (user_emb @ item_emb.T) / model.temperature

            # Labels: diagonal entries (index i matches with item i)
            labels = torch.arange(batch_size, device=sim_matrix.device)
            loss = F.cross_entropy(sim_matrix, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss / len(train_data):.4f}")


# ---- Build ANN index for serving ----
def build_ann_index(model, n_items):
    """Precompute item embeddings and build a FAISS index."""
    import faiss

    model.eval()
    with torch.no_grad():
        item_ids = torch.arange(n_items)
        item_embs = model.get_item_embeddings(item_ids).numpy()

    dim = item_embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cosine after L2 norm)
    index.add(item_embs)
    return index, item_embs


# Usage: retrieve top-k candidates for a user
# user_emb = model.get_user_embeddings(torch.LongTensor([user_id])).numpy()
# scores, indices = index.search(user_emb, k=100)
# candidate_item_ids = indices[0]`}
        />

        <h3>Multi-Objective Ranking with Weighted Loss</h3>
        <CodeBlock
          language="python"
          title="multi_objective_ranker.py"
          code={`import torch
import torch.nn as nn
import numpy as np

class MultiObjectiveRanker(nn.Module):
    """
    Ranking model that predicts multiple objectives:
    - P(click), P(like), E[watch_time]
    Shared bottom layers + task-specific heads.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        # Shared bottom layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Task-specific heads
        self.click_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.like_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.watch_time_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),  # watch time is non-negative
        )

    def forward(self, x):
        shared_repr = self.shared(x)
        return {
            "click": self.click_head(shared_repr).squeeze(-1),
            "like": self.like_head(shared_repr).squeeze(-1),
            "watch_time": self.watch_time_head(shared_repr).squeeze(-1),
        }


def multi_objective_loss(preds, targets, weights=None):
    """
    Compute weighted multi-objective loss.

    Args:
        preds: dict of predictions {task_name: tensor}
        targets: dict of ground truth {task_name: tensor}
        weights: dict of task weights (default: equal)
    """
    if weights is None:
        weights = {"click": 1.0, "like": 1.0, "watch_time": 0.1}

    bce = nn.BCELoss()
    mse = nn.MSELoss()

    losses = {
        "click": bce(preds["click"], targets["click"]),
        "like": bce(preds["like"], targets["like"]),
        "watch_time": mse(preds["watch_time"], targets["watch_time"]),
    }

    total = sum(weights[task] * losses[task] for task in losses)
    return total, losses


def compute_ranking_score(preds, weights=None):
    """
    Combine multi-objective predictions into a single ranking score.
    This is the formula used at serving time to sort candidates.
    """
    if weights is None:
        weights = {"click": 1.0, "like": 2.0, "watch_time": 0.5}

    score = (
        weights["click"] * preds["click"]
        + weights["like"] * preds["like"]
        + weights["watch_time"] * preds["watch_time"]
    )
    return score


# ---- Training loop ----
model = MultiObjectiveRanker(input_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Synthetic training data
n_samples = 5000
X = torch.randn(n_samples, 128)
targets = {
    "click": torch.randint(0, 2, (n_samples,)).float(),
    "like": torch.randint(0, 2, (n_samples,)).float(),
    "watch_time": torch.rand(n_samples) * 300,  # 0-300 seconds
}

for epoch in range(20):
    model.train()
    preds = model(X)
    loss, task_losses = multi_objective_loss(preds, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d} | Total: {loss:.4f} | "
              f"Click: {task_losses['click']:.4f} | "
              f"Like: {task_losses['like']:.4f} | "
              f"WatchTime: {task_losses['watch_time']:.4f}")

# ---- Ranking at serving time ----
model.eval()
with torch.no_grad():
    candidate_features = torch.randn(100, 128)  # 100 candidates
    preds = model(candidate_features)
    scores = compute_ranking_score(preds)
    top10 = torch.topk(scores, 10)
    print(f"\\nTop 10 candidate indices: {top10.indices.tolist()}")
    print(f"Ranking scores: {[f'{s:.3f}' for s in top10.values.tolist()]}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Candidate generation + ranking is the standard architecture</strong>: The candidate generation stage (two-tower with ANN) retrieves hundreds of candidates from millions of items in milliseconds. The ranking stage applies a richer, more expensive model to score just those candidates. Never try to rank all items with a single model.</li>
          <li><strong>Embedding table sizes dominate model size</strong>: With millions of users and items, embedding tables can be gigabytes. Use techniques like hashing tricks, mixed-dimension embeddings (popular items get larger embeddings), or compositional embeddings to manage memory.</li>
          <li><strong>Serving at scale</strong>: Precompute all item embeddings and store in a vector index (FAISS, ScaNN, Milvus). User embeddings are computed on-the-fly from recent activity. The ANN lookup is <InlineMath math="O(\log N)" />, enabling sub-10ms retrieval over billions of items.</li>
          <li><strong>Cold start with content features</strong>: New items have no interaction data. Feed item content features (title, description, category, image embeddings) into the item tower so it can produce a reasonable embedding without any behavioral signal. This is where deep learning has a massive advantage over pure matrix factorization.</li>
          <li><strong>In-batch negatives are essential</strong>: For implicit feedback, you need negative examples. In-batch negative sampling (treating other items in the mini-batch as negatives) is efficient and effective. Be careful of popularity bias — popular items appear as negatives more often, pushing their embeddings away from all users.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Not separating retrieval from ranking</strong>: Trying to build a single model that scores all items is computationally infeasible at scale. You need a fast, approximate retrieval stage followed by a precise ranking stage. This two-stage design is not optional for any system with more than ~10K items.</li>
          <li><strong>Ignoring position bias</strong>: Items shown in the first position get clicked more regardless of relevance. If you train on click data without correcting for position, the model learns to recommend items that are already popular. Use inverse propensity weighting or position features to debias.</li>
          <li><strong>Training on implicit feedback without negative sampling</strong>: With implicit feedback (clicks, views), you only observe positive interactions. Without carefully constructed negatives, the model has no signal for what users dislike. Use random negatives, in-batch negatives, or hard negative mining.</li>
          <li><strong>Embedding dimension too large</strong>: Larger embeddings are not always better. With sparse data, large embeddings overfit. Start with 32-64 dimensions and increase only if validation metrics improve. YouTube&apos;s production model used 256 dimensions for hundreds of millions of videos.</li>
          <li><strong>Not using features beyond IDs</strong>: Pure ID-based embeddings cannot generalize to new items. Always include content features (text, category, metadata) in the item tower. This is what makes deep learning better than matrix factorization for recommendation.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Design a recommendation system for a video platform with 100M users and 10M videos. Walk through the architecture from candidate generation to final ranking.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Candidate generation (two-tower model)</strong>:
            <ul>
              <li>User tower inputs: user ID embedding, recent watch history (average of last 50 video embeddings), demographics, time features.</li>
              <li>Item tower inputs: video ID embedding, title/description embeddings (from a pretrained language model), category, duration, upload date, creator features.</li>
              <li>Train with in-batch negative sampling on watch events. Loss: softmax cross-entropy over the batch.</li>
              <li>Precompute all 10M video embeddings, index with FAISS HNSW. At serving time, compute user embedding and retrieve top 500 candidates in ~5ms.</li>
            </ul>
          </li>
          <li>
            <strong>Ranking model (multi-objective)</strong>:
            <ul>
              <li>Input: user features + candidate features + cross features (user-video interaction history, context).</li>
              <li>Architecture: shared bottom layers with task-specific heads for P(click), P(complete), E[watch time], P(like), P(share).</li>
              <li>Final ranking score: weighted combination tuned to business objectives (e.g., maximize watch time while maintaining engagement diversity).</li>
            </ul>
          </li>
          <li>
            <strong>Re-ranking and business rules</strong>:
            <ul>
              <li>Apply diversity constraints: don&apos;t show 10 videos from the same creator.</li>
              <li>Freshness boost: upweight recent uploads to encourage content creation.</li>
              <li>Safety filters: remove policy-violating content.</li>
            </ul>
          </li>
          <li>
            <strong>Cold start strategy</strong>:
            <ul>
              <li>New videos: content features in item tower provide a reasonable initial embedding. Explore/exploit with Thompson sampling for exposure.</li>
              <li>New users: start with popular/trending videos. After 5-10 watches, the user tower produces meaningful embeddings.</li>
            </ul>
          </li>
          <li>
            <strong>Evaluation</strong>:
            <ul>
              <li>Offline: NDCG@10, Recall@500 (retrieval), AUC for each objective head.</li>
              <li>Online A/B test: daily active users, total watch time, user retention at day 7 and day 30.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>He et al. (2017) &quot;Neural Collaborative Filtering&quot;</strong> &mdash; The foundational paper replacing dot products with neural networks for collaborative filtering.</li>
          <li><strong>Covington et al. (2016) &quot;Deep Neural Networks for YouTube Recommendations&quot;</strong> &mdash; The seminal paper on two-tower architecture and candidate generation at YouTube scale.</li>
          <li><strong>Naumov et al. (2019) &quot;Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM)&quot;</strong> &mdash; Meta&apos;s production recommendation architecture combining embeddings with MLPs.</li>
          <li><strong>Zhang et al. (2019) &quot;Deep Learning based Recommender Systems: A Survey and New Perspectives&quot;</strong> &mdash; Comprehensive survey of deep learning approaches in recommendation systems.</li>
          <li><strong>Yi et al. (2019) &quot;Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations&quot;</strong> &mdash; Google&apos;s approach to correcting popularity bias in two-tower models with in-batch negatives.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
