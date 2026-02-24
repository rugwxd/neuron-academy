"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function EncodingStrategies() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Machine learning models operate on numbers, but real-world data is full of categories: country,
          product type, color, user segment, day of week. <strong>Encoding</strong> is the process of
          converting these categorical variables into numerical representations that models can learn from.
          The choice of encoding strategy has a surprisingly large impact on model performance — often
          more than the choice of algorithm itself.
        </p>
        <p>
          <strong>One-hot encoding</strong> creates a binary column for each category. It&apos;s simple and
          safe but explodes dimensionality for high-cardinality features (a &quot;city&quot; column with
          50,000 unique values creates 50,000 columns). <strong>Target encoding</strong> replaces each
          category with the mean of the target variable for that category — powerful but risky because
          it can leak the target into the features. <strong>Frequency encoding</strong> replaces each
          category with its count or proportion — a simple, leak-free heuristic.
        </p>
        <p>
          For high-cardinality features (user IDs, product IDs, zip codes), <strong>learned embeddings</strong>
          are the modern solution. Instead of a 50,000-dimensional one-hot vector, each category maps to a
          dense vector of, say, 32 dimensions. These embeddings are learned jointly with the model, capturing
          semantic similarities: cities with similar demographics end up near each other in embedding space.
          This is the same idea as word embeddings (Word2Vec) applied to tabular data.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>One-Hot Encoding</h3>
        <p>
          For a categorical feature with <InlineMath math="K" /> unique values, one-hot encoding maps each
          value to a vector in <InlineMath math="\{0, 1\}^K" />:
        </p>
        <BlockMath math="\text{one\_hot}(\text{category}_k) = \mathbf{e}_k = (0, \ldots, \underbrace{1}_{k\text{-th}}, \ldots, 0)" />
        <p>
          Dimensionality: <InlineMath math="K" /> new columns per feature. For tree-based models, this
          is wasteful — the model must learn <InlineMath math="K" /> separate splits instead of one
          ordered split.
        </p>

        <h3>Target Encoding (Mean Encoding)</h3>
        <p>
          Replace category <InlineMath math="c" /> with the average target value for that category, with
          smoothing to handle rare categories:
        </p>
        <BlockMath math="\text{encode}(c) = \frac{n_c \cdot \bar{y}_c + m \cdot \bar{y}_{\text{global}}}{n_c + m}" />
        <p>
          where <InlineMath math="n_c" /> is the count of category <InlineMath math="c" />,
          <InlineMath math="\bar{y}_c" /> is its mean target, <InlineMath math="\bar{y}_{\text{global}}" /> is
          the global mean, and <InlineMath math="m" /> is a smoothing parameter (typically 10-100). When
          <InlineMath math="n_c" /> is small, the encoding is pulled toward the global mean, reducing
          overfitting to rare categories.
        </p>

        <h3>Frequency Encoding</h3>
        <p>Replace each category with its frequency or proportion:</p>
        <BlockMath math="\text{freq\_encode}(c) = \frac{|\{x_i : x_i = c\}|}{n}" />
        <p>
          Simple, no target leakage, and often informative (popular products, frequent cities tend to
          behave differently from rare ones).
        </p>

        <h3>Learned Embeddings</h3>
        <p>
          Map each category <InlineMath math="c \in \{1, \ldots, K\}" /> to a dense
          vector <InlineMath math="\mathbf{e}_c \in \mathbb{R}^d" /> where <InlineMath math="d \ll K" />:
        </p>
        <BlockMath math="\mathbf{e}_c = E[c, :] \quad \text{where } E \in \mathbb{R}^{K \times d}" />
        <p>
          The embedding matrix <InlineMath math="E" /> is learned via backpropagation. The lookup is simply
          an indexing operation — no matrix multiplication needed (unlike multiplying by a one-hot vector).
        </p>
        <p>Rule of thumb for embedding dimension:</p>
        <BlockMath math="d = \min\left(\left\lfloor \frac{K}{2} \right\rfloor, 50\right) \quad \text{or} \quad d = \min\left(\left\lceil K^{0.25} \right\rceil, 600\right)" />
      </TopicSection>

      <TopicSection type="code">
        <h3>All Encoding Strategies Compared</h3>
        <CodeBlock
          language="python"
          title="encoding_strategies.py"
          code={`import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold

# Sample data
df = pd.DataFrame({
    "city": ["NYC", "LA", "NYC", "Chicago", "LA", "NYC",
             "Chicago", "LA", "NYC", "Boston"],
    "color": ["red", "blue", "red", "green", "blue",
              "red", "green", "blue", "red", "blue"],
    "target": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
})

# --- 1. One-Hot Encoding ---
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
one_hot = ohe.fit_transform(df[["city"]])
print("One-hot shape:", one_hot.shape)  # (10, 4) — one column per city
print("Columns:", ohe.get_feature_names_out())

# --- 2. Label Encoding (ordinal — only for tree models!) ---
le = LabelEncoder()
df["city_label"] = le.fit_transform(df["city"])
print("\\nLabel encoded:", df["city_label"].values)
# [2, 1, 2, 0, 1, 2, 0, 1, 2, 3] — ordinal, implies false order

# --- 3. Frequency Encoding ---
freq = df["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq)
print("\\nFrequency encoded:", df["city_freq"].values)
# NYC: 0.4, LA: 0.3, Chicago: 0.2, Boston: 0.1

# --- 4. Target Encoding (with smoothing + CV to prevent leakage) ---
def target_encode_cv(df, col, target, n_folds=5, smoothing=10):
    """Target encoding with cross-validation to prevent overfitting."""
    global_mean = df[target].mean()
    encoded = pd.Series(index=df.index, dtype=float)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df):
        # Compute means on training fold only
        means = df.iloc[train_idx].groupby(col)[target].agg(["mean", "count"])
        smoothed = (
            means["count"] * means["mean"] + smoothing * global_mean
        ) / (means["count"] + smoothing)

        # Apply to validation fold
        encoded.iloc[val_idx] = df.iloc[val_idx][col].map(smoothed)

    # Fill missing (unseen categories) with global mean
    encoded = encoded.fillna(global_mean)
    return encoded

df["city_target_enc"] = target_encode_cv(df, "city", "target")
print("\\nTarget encoded:", df["city_target_enc"].values.round(3))

# --- 5. Binary / Hash Encoding (for very high cardinality) ---
import category_encoders as ce

binary_enc = ce.BinaryEncoder(cols=["city"])
binary_df = binary_enc.fit_transform(df[["city"]])
print("\\nBinary encoding shape:", binary_df.shape)
# log2(K) columns instead of K — much more compact`}
        />

        <h3>Learned Embeddings in PyTorch</h3>
        <CodeBlock
          language="python"
          title="learned_embeddings.py"
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularModel(nn.Module):
    """
    Neural network for tabular data with learned embeddings.
    This is the architecture behind entity embeddings (Cheng et al.),
    used at Google (Wide & Deep), Instacart, and many Kaggle winners.
    """
    def __init__(self, cat_dims, embedding_dims, n_continuous, n_classes):
        """
        Args:
            cat_dims: list of cardinalities [100, 50, 7, ...]
            embedding_dims: list of embedding sizes [16, 8, 3, ...]
            n_continuous: number of continuous features
            n_classes: output classes
        """
        super().__init__()

        # One embedding layer per categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats, emb_dim)
            for n_cats, emb_dim in zip(cat_dims, embedding_dims)
        ])

        total_emb_dim = sum(embedding_dims)
        input_dim = total_emb_dim + n_continuous

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x_cat, x_cont):
        """
        x_cat: (batch, n_cat_features) — integer category indices
        x_cont: (batch, n_continuous) — continuous features
        """
        # Look up embeddings for each categorical feature
        emb_outputs = [
            emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)
        ]
        x_emb = torch.cat(emb_outputs, dim=1)  # (batch, total_emb_dim)

        # Concatenate embeddings with continuous features
        x = torch.cat([x_emb, x_cont], dim=1)
        return self.mlp(x)

# Example usage
# 3 categorical features: city (1000 unique), category (50), day_of_week (7)
# 5 continuous features: price, quantity, etc.
model = TabularModel(
    cat_dims=[1000, 50, 7],
    embedding_dims=[32, 12, 3],    # Rule of thumb: min(K/2, 50)
    n_continuous=5,
    n_classes=2,
)

# Dummy batch
x_cat = torch.randint(0, 50, (32, 3))    # 32 samples, 3 cat features
x_cont = torch.randn(32, 5)               # 32 samples, 5 cont features
logits = model(x_cat, x_cont)
print(f"Output shape: {logits.shape}")     # (32, 2)

# After training, inspect embeddings:
# city_emb = model.embeddings[0].weight.detach()
# Similar cities (by demographics/behavior) will cluster together`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>For tree-based models (XGBoost, LightGBM)</strong>: Use label encoding or target encoding. One-hot encoding is wasteful because trees can natively split on integer-encoded categories. LightGBM has built-in categorical support that handles this automatically.</li>
          <li><strong>For linear models and neural networks</strong>: Use one-hot for low cardinality (&lt;20 categories) and embeddings for high cardinality. Never use label encoding — it implies a false ordinal relationship.</li>
          <li><strong>Target encoding is the strongest single technique</strong>: It consistently wins Kaggle competitions. But always use cross-validation or leave-one-out encoding to prevent leakage. The <code>category_encoders</code> library has a robust implementation.</li>
          <li><strong>Embedding dimension rule of thumb</strong>: <InlineMath math="d = \min(K/2, 50)" /> works well. For user/item IDs with millions of values, use 64-256 dimensions. Too large = overfitting; too small = underfitting.</li>
          <li><strong>Pre-trained embeddings transfer</strong>: Train embeddings on a large task (e.g., predicting user clicks), then reuse them in downstream models. This is exactly how Word2Vec works, but for tabular entities.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Label encoding for non-tree models</strong>: Assigning city=0, city=1, city=2 tells a linear model that city 2 is &quot;twice as much&quot; as city 1. This is nonsensical for nominal categories. Only use label encoding with tree-based models.</li>
          <li><strong>Target encoding without cross-validation</strong>: Computing target means on the full dataset leaks the label into the features. The model will overfit, especially for rare categories. Always use k-fold target encoding where the encoding for each fold is computed on the other folds.</li>
          <li><strong>One-hot encoding high-cardinality features</strong>: A zip code feature with 40,000 unique values creates 40,000 sparse columns. This slows training, wastes memory, and doesn&apos;t capture similarity between nearby zip codes. Use embeddings or target encoding instead.</li>
          <li><strong>Ignoring unknown categories at inference time</strong>: Your training data has 100 cities. At serving time, a new city appears. One-hot: error. Target encoding: use the global mean. Embeddings: use a special &quot;unknown&quot; index. Always handle this case.</li>
          <li><strong>Not encoding cyclical features properly</strong>: Day of week (1-7) and hour of day (0-23) are cyclical — hour 23 is close to hour 0. Use sine/cosine encoding: <InlineMath math="x_{\sin} = \sin(2\pi \cdot h/24)" />, <InlineMath math="x_{\cos} = \cos(2\pi \cdot h/24)" />.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You have a categorical feature &quot;merchant_id&quot; with 500,000 unique values. Your model is XGBoost. How would you encode this feature? What changes if the model is a neural network?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>For XGBoost</strong>:
            <ul>
              <li><strong>Target encoding</strong> is the best approach. Replace each merchant_id with the smoothed mean of the target for that merchant, computed using k-fold cross-validation to prevent leakage. This compresses 500K categories into a single informative column.</li>
              <li>Supplement with <strong>frequency encoding</strong> (how many transactions per merchant) and <strong>aggregation features</strong> (mean transaction amount, fraud rate per merchant).</li>
              <li>Do <strong>not</strong> use one-hot encoding — 500K columns is infeasible. Do <strong>not</strong> use plain label encoding — XGBoost would need to learn 500K split points on an arbitrary ordering.</li>
            </ul>
          </li>
          <li><strong>For a neural network</strong>:
            <ul>
              <li>Use a <strong>learned embedding layer</strong>: <code>nn.Embedding(500_000, 64)</code>. The 64-dimensional embedding captures relationships between merchants (similar merchants cluster in embedding space).</li>
              <li>This is the approach used in Google&apos;s Wide &amp; Deep and Alibaba&apos;s Deep Interest Network.</li>
              <li>Advantages over target encoding: the embedding is jointly optimized with the model (not a fixed preprocessing step), and it captures multi-dimensional relationships (not just the target mean).</li>
              <li>Handle new merchants with an &quot;unknown&quot; embedding index (index 0), or use a hash embedding (<code>merchant_hash = hash(id) % 100_000</code>) to bound the vocabulary.</li>
            </ul>
          </li>
          <li><strong>Hybrid approach</strong>: In practice, combine both. Use target-encoded merchant features as input to XGBoost for the first model. Then train a neural network with embeddings as a second model and ensemble them.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Guo &amp; Berkhahn (2016) &quot;Entity Embeddings of Categorical Variables&quot;</strong> — The Kaggle-winning paper that popularized learned embeddings for tabular data.</li>
          <li><strong>Cheng et al. (2016) &quot;Wide &amp; Deep Learning for Recommender Systems&quot;</strong> — Google&apos;s architecture combining memorization (wide) with generalization (deep embeddings).</li>
          <li><strong>Micci-Barreca (2001) &quot;A Preprocessing Scheme for High-Cardinality Categorical Attributes&quot;</strong> — The original target encoding paper with smoothing.</li>
          <li><strong>category_encoders Python library</strong> — 20+ encoding strategies with scikit-learn-compatible API.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
