"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function LearningToRank() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Learning to Rank (LTR) is the application of machine learning to the ranking problem. Instead of
          hand-crafting a scoring formula like BM25, we <strong>learn</strong> a function that takes a
          query-document pair and predicts how relevant the document is. The model learns from human
          relevance judgments — data where editors have labeled documents as &quot;highly relevant,&quot;
          &quot;somewhat relevant,&quot; or &quot;not relevant&quot; for given queries.
        </p>
        <p>
          There are three families of approaches, and the distinction matters because it determines what
          the model is optimizing. <strong>Pointwise</strong> methods treat each query-document pair
          independently and predict a relevance score — essentially a regression or classification problem.
          <strong> Pairwise</strong> methods look at pairs of documents for the same query and learn which
          one should rank higher — this is closer to the actual ranking objective.
          <strong> Listwise</strong> methods consider the entire list of documents for a query at once and
          directly optimize a ranking metric like NDCG.
        </p>
        <p>
          In practice, <strong>LambdaMART</strong> (a pairwise/listwise hybrid using gradient-boosted trees)
          has been the workhorse of production search and recommendation for over a decade. It powers
          ranking at Microsoft Bing, Yahoo, Airbnb, LinkedIn, and countless other companies. More recently,
          Transformer-based cross-encoders and neural rankers have gained ground, but LambdaMART remains
          extremely competitive, especially when you have strong hand-crafted features.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Pointwise Approach</h3>
        <p>
          Treat ranking as regression or classification. Given a feature
          vector <InlineMath math="\mathbf{x}_{q,d}" /> for query-document pair <InlineMath math="(q, d)" />,
          predict the relevance label <InlineMath math="y \in \{0, 1, 2, 3, 4\}" />:
        </p>
        <BlockMath math="\hat{y} = f(\mathbf{x}_{q,d})" />
        <p>Loss: MSE for regression, cross-entropy for classification. Problem: it doesn&apos;t model the
          relative order between documents — a model that predicts 4.1 vs 4.0 for two documents is penalized
          the same as one that predicts 4.0 vs 1.0, even though only the relative order matters for ranking.</p>

        <h3>Pairwise Approach (RankNet)</h3>
        <p>
          For a query <InlineMath math="q" />, given two documents <InlineMath math="d_i" /> and <InlineMath math="d_j" /> where
          <InlineMath math="d_i" /> is more relevant, learn a scoring function <InlineMath math="s = f(\mathbf{x})" /> such
          that <InlineMath math="s_i > s_j" />. The probability that <InlineMath math="d_i" /> should be ranked
          above <InlineMath math="d_j" />:
        </p>
        <BlockMath math="P_{ij} = \sigma(s_i - s_j) = \frac{1}{1 + e^{-(s_i - s_j)}}" />
        <p>Loss (binary cross-entropy on pairs):</p>
        <BlockMath math="L = -\bar{P}_{ij} \log P_{ij} - (1 - \bar{P}_{ij}) \log(1 - P_{ij})" />
        <p>
          where <InlineMath math="\bar{P}_{ij} = 1" /> if <InlineMath math="d_i \succ d_j" /> (i is more relevant).
        </p>

        <h3>LambdaMART (Pairwise + Listwise)</h3>
        <p>
          The key idea: weight each pairwise gradient by how much <strong>swapping</strong> the two documents
          would change the evaluation metric (NDCG). This is the <InlineMath math="\lambda" />-gradient:
        </p>
        <BlockMath math="\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} \cdot |\Delta \text{NDCG}_{ij}|" />
        <p>
          where <InlineMath math="|\Delta \text{NDCG}_{ij}|" /> is the change in NDCG if you swap the positions
          of documents <InlineMath math="i" /> and <InlineMath math="j" />. This means: &quot;if swapping these
          two documents would greatly change NDCG, assign a large gradient; if it barely matters, assign a small
          one.&quot; The gradients are then used to train gradient-boosted trees (MART).
        </p>

        <h3>NDCG (Normalized Discounted Cumulative Gain)</h3>
        <p>The standard evaluation metric for ranking. For a ranked list of <InlineMath math="k" /> documents:</p>
        <BlockMath math="\text{DCG}@k = \sum_{i=1}^{k} \frac{2^{r_i} - 1}{\log_2(i + 1)}" />
        <BlockMath math="\text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}" />
        <p>
          where <InlineMath math="r_i" /> is the relevance grade at position <InlineMath math="i" />,
          and <InlineMath math="\text{IDCG}" /> is the DCG of the ideal (perfect) ranking. NDCG ranges
          from 0 to 1, with 1 meaning the ranking is perfect.
        </p>

        <h3>Listwise Approach (ListNet)</h3>
        <p>
          Define a probability distribution over all permutations of documents using Plackett-Luce:
        </p>
        <BlockMath math="P(\pi | s) = \prod_{i=1}^{n} \frac{e^{s_{\pi(i)}}}{\sum_{j=i}^{n} e^{s_{\pi(j)}}}" />
        <p>Minimize the KL divergence between the predicted permutation distribution and the ground truth distribution.
          In practice, the &quot;top-one&quot; approximation is used, only considering the probability of each
          document being ranked first.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>LambdaMART with LightGBM</h3>
        <CodeBlock
          language="python"
          title="lambdamart_lightgbm.py"
          code={`import lightgbm as lgb
import numpy as np

# --- Prepare ranking data ---
# Features: [bm25_score, title_match, click_rate, doc_freshness, ...]
# Each row is a (query, document) pair
# group_sizes tells the model which rows belong to the same query

np.random.seed(42)
n_queries = 500
docs_per_query = 20
n_features = 10

# Simulate features and relevance labels (0-4 scale)
X_train = np.random.randn(n_queries * docs_per_query, n_features)
y_train = np.random.randint(0, 5, n_queries * docs_per_query)
group_train = [docs_per_query] * n_queries  # 20 docs per query

X_val = np.random.randn(100 * docs_per_query, n_features)
y_val = np.random.randint(0, 5, 100 * docs_per_query)
group_val = [docs_per_query] * 100

# Create LightGBM datasets with group info
train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
val_data = lgb.Dataset(X_val, label=y_val, group=group_val)

# Train LambdaMART
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5, 10],           # Evaluate NDCG@5 and NDCG@10
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 10,
    "max_depth": 7,
    "lambdarank_truncation_level": 10,  # Focus on top 10 positions
    "verbose": -1,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=300,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
)

# Predict relevance scores for new documents
scores = model.predict(X_val[:20])  # Scores for first query
ranked_indices = np.argsort(-scores)  # Sort descending
print(f"Predicted ranking: {ranked_indices}")
print(f"True labels:       {y_val[:20][ranked_indices]}")`}
        />

        <h3>NDCG Evaluation from Scratch</h3>
        <CodeBlock
          language="python"
          title="ndcg_evaluation.py"
          code={`import numpy as np

def dcg_at_k(relevances, k):
    """Compute DCG@k for a single ranked list."""
    relevances = np.array(relevances[:k])
    positions = np.arange(1, len(relevances) + 1)
    return np.sum((2**relevances - 1) / np.log2(positions + 1))

def ndcg_at_k(relevances, k):
    """Compute NDCG@k for a single ranked list."""
    dcg = dcg_at_k(relevances, k)
    # Ideal DCG: sort relevances descending
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0

# Example: search results with relevance grades (0-4)
# Position 1 has relevance 3, position 2 has relevance 0, etc.
predicted_ranking = [3, 0, 2, 1, 4, 0, 1, 2]

print(f"NDCG@3: {ndcg_at_k(predicted_ranking, 3):.4f}")
print(f"NDCG@5: {ndcg_at_k(predicted_ranking, 5):.4f}")
print(f"NDCG@10: {ndcg_at_k(predicted_ranking, 10):.4f}")

# Perfect ranking would be [4, 3, 2, 2, 1, 1, 0, 0]
perfect = sorted(predicted_ranking, reverse=True)
print(f"\\nIdeal NDCG@5: {ndcg_at_k(perfect, 5):.4f}")  # 1.0

# --- Compare different rankings ---
ranking_a = [4, 3, 2, 1, 0]   # Good: relevant docs at top
ranking_b = [0, 1, 2, 3, 4]   # Bad: relevant docs at bottom
print(f"\\nRanking A NDCG@5: {ndcg_at_k(ranking_a, 5):.4f}")  # ~1.0
print(f"Ranking B NDCG@5: {ndcg_at_k(ranking_b, 5):.4f}")    # Much lower`}
        />

        <h3>Cross-Encoder Re-Ranker</h3>
        <CodeBlock
          language="python"
          title="cross_encoder_reranker.py"
          code={`from sentence_transformers import CrossEncoder

# Cross-encoder: jointly encodes query + document
# Much more accurate than bi-encoder, but O(n) forward passes
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "What is gradient descent?"
documents = [
    "Gradient descent is an optimization algorithm for finding minima.",
    "The stock market experienced a steep descent in March.",
    "SGD, Adam, and RMSprop are variants of gradient descent.",
    "Mountains with a steep descent require careful hiking.",
]

# Score each (query, document) pair
pairs = [(query, doc) for doc in documents]
scores = model.predict(pairs)

# Rank by score
ranked = sorted(zip(scores, documents), reverse=True)
for score, doc in ranked:
    print(f"  {score:.4f}: {doc}")

# Output:
#   0.9987: Gradient descent is an optimization algorithm...
#   0.9921: SGD, Adam, and RMSprop are variants...
#   0.0023: The stock market experienced a steep descent...
#   0.0011: Mountains with a steep descent...`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>LambdaMART is still king for tabular features</strong>: When you have strong engineered features (BM25 score, click-through rate, freshness, authority), LambdaMART is extremely hard to beat. It&apos;s what LinkedIn, Airbnb, and Microsoft use.</li>
          <li><strong>Features matter more than the model</strong>: The ranking at most companies improves more from adding a new feature (e.g., user-item affinity, recency signal) than from changing the algorithm.</li>
          <li><strong>Two-pass architecture</strong>: Use a fast retrieval model (BM25 or bi-encoder) to get top-1000, then a slow re-ranker (LambdaMART or cross-encoder) for the final top-20. This is the standard at scale.</li>
          <li><strong>Position bias in click data</strong>: Users click higher-ranked results more, regardless of relevance. If you train on click data, you need to correct for position bias (inverse propensity weighting or position features).</li>
          <li><strong>Evaluate with NDCG, not accuracy</strong>: NDCG is the standard metric because it rewards putting highly relevant documents at the top. Also report MRR (Mean Reciprocal Rank) for navigational queries where there&apos;s one right answer.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Training pointwise when you care about ranking</strong>: Pointwise models minimize prediction error, not ranking quality. A model that assigns scores 4.1, 4.0, 3.9 (correct order) is penalized the same as 4.1, 4.0, 0.1 (also correct order) even though the second is much better at separation.</li>
          <li><strong>Leaking future information</strong>: If you use click-through rate as a feature and train on the same time period, you&apos;re leaking the label into the features. Always use temporal splits.</li>
          <li><strong>Ignoring position bias in training data</strong>: Documents shown at position 1 get clicked more, so naive click models learn &quot;whatever was at position 1 is good.&quot; Use propensity-weighted loss or randomized experiments to collect unbiased data.</li>
          <li><strong>Optimizing the wrong metric</strong>: NDCG@5 for web search (users rarely scroll), NDCG@20 for e-commerce (users browse more), MRR for Q&amp;A (one right answer). Using the wrong cutoff leads to poor user experience.</li>
          <li><strong>Not grouping by query during training</strong>: Pairwise and listwise methods require documents to be grouped by query. Shuffling across queries produces meaningless gradients.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain the difference between pointwise, pairwise, and listwise Learning to Rank. When would you choose each? What is LambdaMART and why is it so widely used?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Pointwise</strong> (e.g., linear regression, single DNN):
            <ul>
              <li>Predicts an absolute relevance score for each (query, doc) pair independently.</li>
              <li>Simple to implement but ignores relative ordering between documents.</li>
              <li>Use when: you have limited data or need a quick baseline.</li>
            </ul>
          </li>
          <li><strong>Pairwise</strong> (e.g., RankNet, RankSVM):
            <ul>
              <li>Learns from pairs: given <InlineMath math="(d_i, d_j)" />, predicts which is more relevant.</li>
              <li>Better than pointwise because it directly models relative order.</li>
              <li>Limitation: treats all pairs equally — swapping docs at positions 1 and 2 is as important as swapping positions 99 and 100.</li>
            </ul>
          </li>
          <li><strong>Listwise</strong> (e.g., ListNet, LambdaRank):
            <ul>
              <li>Considers the entire ranked list and directly optimizes an IR metric (NDCG).</li>
              <li>Best alignment with the evaluation metric, but harder to optimize.</li>
            </ul>
          </li>
          <li><strong>LambdaMART</strong> is a hybrid:
            <ul>
              <li>Uses pairwise comparisons but weights each pair by <InlineMath math="|\Delta\text{NDCG}|" /> — the change in NDCG if you swap the two documents.</li>
              <li>This means pairs near the top of the list (where NDCG is most sensitive) get larger gradients.</li>
              <li>Built on gradient-boosted trees (MART = Multiple Additive Regression Trees), which handle heterogeneous features, missing values, and are fast to train/evaluate.</li>
              <li>It&apos;s widely used because it combines the practical strengths of GBDTs (feature flexibility, no feature scaling, interpretable feature importance) with a loss function that directly optimizes the metric we care about (NDCG).</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Burges (2010) &quot;From RankNet to LambdaRank to LambdaMART: An Overview&quot;</strong> — The definitive overview of the Lambda family from Microsoft Research.</li>
          <li><strong>Liu (2011) &quot;Learning to Rank for Information Retrieval&quot;</strong> — Comprehensive textbook covering all three approaches.</li>
          <li><strong>LightGBM LambdaRank documentation</strong> — Practical guide to training LambdaMART with LightGBM.</li>
          <li><strong>Nogueira &amp; Cho (2020) &quot;Passage Re-ranking with BERT&quot;</strong> — How Transformers are used as re-rankers.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
