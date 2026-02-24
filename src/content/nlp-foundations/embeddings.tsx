"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Embeddings() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Words are symbols — the string &quot;king&quot; has no inherent mathematical meaning. For a neural network to process
          language, we need to represent each word as a dense vector of real numbers. These vectors are called
          <strong> word embeddings</strong>, and the key insight is that words with similar meanings should end up with similar vectors.
        </p>
        <p>
          The breakthrough idea behind <strong>Word2Vec</strong> (2013) was that you can learn these vectors purely from context.
          A word is defined by the company it keeps: &quot;king&quot; appears in similar contexts to &quot;queen&quot;, so their vectors
          should be close. Even more remarkably, the vectors capture <strong>analogies</strong>:
          <InlineMath math="\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}" />.
          The embedding space encodes semantic relationships as geometric directions.
        </p>
        <p>
          <strong>GloVe</strong> (2014) took a different approach: instead of predicting context words one at a time, it factorizes
          the entire word co-occurrence matrix. This global perspective produces vectors where the dot product between two word
          vectors equals the log of their co-occurrence probability. Both methods produce 100-300 dimensional vectors that capture
          rich semantic and syntactic information.
        </p>
        <p>
          Modern transformer models use <strong>contextual embeddings</strong> where the same word gets different vectors depending on
          context (&quot;bank&quot; in &quot;river bank&quot; vs. &quot;bank account&quot;). But static embeddings like Word2Vec and GloVe
          remain fundamental — they are the conceptual foundation for the embedding layers in every neural language model.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Word2Vec: Skip-gram</h3>
        <p>
          Given a center word <InlineMath math="w_c" /> and a context word <InlineMath math="w_o" />, the skip-gram model maximizes:
        </p>
        <BlockMath math="P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_c})}" />
        <p>
          where <InlineMath math="\mathbf{v}_{w_c} \in \mathbb{R}^d" /> is the center word embedding and <InlineMath math="\mathbf{u}_{w_o} \in \mathbb{R}^d" /> is the context word embedding. The objective over the corpus is:
        </p>
        <BlockMath math="J = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-m \le j \le m, j \ne 0} \log P(w_{t+j} \mid w_t)" />
        <p>
          where <InlineMath math="T" /> is the corpus length and <InlineMath math="m" /> is the context window size.
        </p>

        <h3>Negative Sampling</h3>
        <p>
          Computing the full softmax over vocabulary <InlineMath math="|V|" /> is expensive. Negative sampling approximates it by
          contrasting the true context pair against <InlineMath math="k" /> randomly sampled negative pairs:
        </p>
        <BlockMath math="J_{\text{NEG}} = \log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n} \left[ \log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_{w_c}) \right]" />
        <p>
          where <InlineMath math="\sigma" /> is the sigmoid function and <InlineMath math="P_n(w) \propto f(w)^{3/4}" /> is the noise distribution (the 3/4 exponent up-weights rare words relative to their frequency).
        </p>

        <h3>Word2Vec: CBOW</h3>
        <p>
          Continuous Bag of Words predicts the center word from the context. Given context words <InlineMath math="\{w_{t-m}, \ldots, w_{t+m}\}" />:
        </p>
        <BlockMath math="P(w_t \mid \text{context}) = \frac{\exp(\mathbf{u}_{w_t}^\top \bar{\mathbf{v}})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \bar{\mathbf{v}})}, \quad \bar{\mathbf{v}} = \frac{1}{2m} \sum_{j=-m, j \ne 0}^{m} \mathbf{v}_{w_{t+j}}" />

        <h3>GloVe</h3>
        <p>
          Let <InlineMath math="X_{ij}" /> be the count of word <InlineMath math="j" /> appearing in the context of word <InlineMath math="i" />.
          GloVe minimizes:
        </p>
        <BlockMath math="J = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( \mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2" />
        <p>
          where <InlineMath math="f(x) = \min\left((x/x_{\max})^\alpha, 1\right)" /> is a weighting function (<InlineMath math="\alpha = 0.75" /> typically) that prevents very frequent pairs from dominating. The key insight: the dot product of word vectors should approximate the log co-occurrence count.
        </p>

        <h3>Cosine Similarity</h3>
        <p>
          Word similarity is measured by cosine similarity between embedding vectors:
        </p>
        <BlockMath math="\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \cos(\theta)" />
        <p>
          This ranges from -1 (opposite) to 1 (identical direction), ignoring vector magnitude.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Word2Vec with Gensim</h3>
        <CodeBlock
          language="python"
          title="word2vec_gensim.py"
          code={`import gensim.downloader as api
from gensim.models import Word2Vec

# Load pre-trained Word2Vec (trained on Google News, 3M words, 300d)
wv = api.load("word2vec-google-news-300")

# Find similar words
print("Most similar to 'king':")
for word, score in wv.most_similar("king", topn=5):
    print(f"  {word}: {score:.4f}")
# queen: 0.6510, prince: 0.6160, monarch: 0.5899, ...

# Famous analogy: king - man + woman = ?
result = wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
print("\\nking - man + woman =")
for word, score in result:
    print(f"  {word}: {score:.4f}")
# queen: 0.7118

# Compute similarity between two words
print(f"\\nsim(king, queen) = {wv.similarity('king', 'queen'):.4f}")
print(f"sim(king, banana) = {wv.similarity('king', 'banana'):.4f}")

# Find the word that doesn't belong
odd = wv.doesnt_match(["breakfast", "lunch", "dinner", "python"])
print(f"\\nOdd one out: {odd}")  # python`}
        />

        <h3>GloVe Embeddings with Nearest Neighbor Search</h3>
        <CodeBlock
          language="python"
          title="glove_explore.py"
          code={`import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_glove(path="glove.6B.100d.txt"):
    """Load GloVe vectors into a dict."""
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    return embeddings

glove = load_glove()
print(f"Loaded {len(glove)} word vectors of dim {len(next(iter(glove.values())))}")

# Build matrix for fast search
words = list(glove.keys())
matrix = np.stack([glove[w] for w in words])  # (vocab_size, 100)

def nearest_neighbors(word, k=10):
    """Find k nearest neighbors by cosine similarity."""
    vec = glove[word].reshape(1, -1)
    sims = cosine_similarity(vec, matrix)[0]
    top_k = np.argsort(sims)[-k-1:-1][::-1]  # exclude self
    return [(words[i], sims[i]) for i in top_k]

def analogy(a, b, c, k=5):
    """Solve a:b :: c:? using vector arithmetic."""
    vec = glove[b] - glove[a] + glove[c]
    vec = vec.reshape(1, -1)
    sims = cosine_similarity(vec, matrix)[0]
    top_k = np.argsort(sims)[-k-1:][::-1]
    exclude = {a, b, c}
    results = [(words[i], sims[i]) for i in top_k if words[i] not in exclude]
    return results[:k]

print("Nearest to 'python':", nearest_neighbors("python", k=5))
print("\\nman:woman :: king:?", analogy("man", "woman", "king"))`}
        />

        <h3>Training Word2Vec from Scratch in PyTorch</h3>
        <CodeBlock
          language="python"
          title="word2vec_scratch.py"
          code={`import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np

class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Two embedding matrices: center and context
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

        # Initialize with small random values
        nn.init.uniform_(self.center_embeddings.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.uniform_(self.context_embeddings.weight, -0.5/embed_dim, 0.5/embed_dim)

    def forward(self, center_ids, context_ids, neg_ids):
        """
        center_ids:  (batch,)
        context_ids: (batch,)
        neg_ids:     (batch, num_neg)
        """
        center = self.center_embeddings(center_ids)     # (batch, d)
        context = self.context_embeddings(context_ids)   # (batch, d)
        negatives = self.context_embeddings(neg_ids)     # (batch, num_neg, d)

        # Positive: log sigmoid(u_o . v_c)
        pos_score = torch.sum(center * context, dim=1)   # (batch,)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)

        # Negative: sum log sigmoid(-u_neg . v_c)
        neg_score = torch.bmm(negatives, center.unsqueeze(2)).squeeze(2)  # (batch, num_neg)
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-10).sum(dim=1)

        return (pos_loss + neg_loss).mean()

# Hyperparameters
vocab_size, embed_dim, num_neg = 10000, 100, 5
model = SkipGramNegSampling(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (pseudocode — real data pipeline omitted for clarity)
# for epoch in range(num_epochs):
#     for center, context, negatives in dataloader:
#         loss = model(center, context, negatives)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# After training, the center embeddings are your word vectors
word_vectors = model.center_embeddings.weight.detach().numpy()
print(f"Learned embeddings shape: {word_vectors.shape}")  # (10000, 100)`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Pre-trained embeddings are a starting point</strong>: In classical NLP pipelines, initializing with GloVe/Word2Vec and fine-tuning on your task consistently outperformed random initialization. Modern transformers learn their own embeddings, but the principle remains.</li>
          <li><strong>Embedding dimension</strong>: 100-300 dimensions is standard for static embeddings. Transformer embedding dimensions are larger (768 for BERT-base, 4096+ for large LLMs) because they must capture context-dependent meaning.</li>
          <li><strong>Out-of-vocabulary handling</strong>: Static embeddings have no vector for unseen words. Common solutions: average character n-gram embeddings (FastText), use the zero vector, or fall back to subword tokenization.</li>
          <li><strong>FastText extends Word2Vec</strong>: By representing each word as a bag of character n-grams, FastText can produce embeddings for any word, even misspelled ones. &quot;unhappily&quot; shares n-grams with &quot;happy&quot;, &quot;unhappy&quot;, etc.</li>
          <li><strong>Bias in embeddings</strong>: Word embeddings absorb societal biases from training data. The analogy &quot;man:computer_programmer :: woman:homemaker&quot; appeared in real Word2Vec models. Debiasing techniques exist but remain imperfect.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Using static embeddings for polysemous words</strong>: Word2Vec gives &quot;bank&quot; one vector whether it means a financial institution or a river bank. If word sense matters, use contextual embeddings (BERT, etc.).</li>
          <li><strong>Confusing embedding similarity with relatedness</strong>: &quot;hot&quot; and &quot;cold&quot; have high cosine similarity because they appear in similar contexts (both describe temperature). High similarity does not mean synonymy.</li>
          <li><strong>Not normalizing before cosine similarity</strong>: If you are using dot product as a similarity measure, remember it conflates direction and magnitude. Always normalize to unit vectors first, or use cosine similarity explicitly.</li>
          <li><strong>Training on tiny corpora</strong>: Word2Vec needs hundreds of millions of words to produce high-quality embeddings. On small datasets, use pre-trained vectors instead of training from scratch.</li>
          <li><strong>Ignoring the two embedding matrices</strong>: Word2Vec produces both center (W) and context (W&apos;) embeddings. Common practice is to use just W or average W and W&apos;. Using only the context matrix is a mistake.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain how Word2Vec&apos;s skip-gram with negative sampling works. Why does the king-queen analogy emerge? How does GloVe differ?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Skip-gram with negative sampling</strong>:
            <ul>
              <li>For each word in the corpus, we create (center, context) pairs within a window of size <InlineMath math="m" />.</li>
              <li>The model has two embedding matrices: center embeddings <InlineMath math="V" /> and context embeddings <InlineMath math="U" />.</li>
              <li>For a positive pair <InlineMath math="(c, o)" />, we want <InlineMath math="\mathbf{u}_o^\top \mathbf{v}_c" /> to be large. For <InlineMath math="k" /> randomly sampled negatives, we want their dot products to be small.</li>
              <li>The loss is: <InlineMath math="-\log \sigma(\mathbf{u}_o^\top \mathbf{v}_c) - \sum_{i=1}^k \log \sigma(-\mathbf{u}_{n_i}^\top \mathbf{v}_c)" /></li>
            </ul>
          </li>
          <li>
            <strong>Why analogies emerge</strong>:
            <ul>
              <li>The training objective implicitly factorizes the pointwise mutual information (PMI) matrix of word co-occurrences.</li>
              <li>If &quot;king&quot; and &quot;queen&quot; share all contexts except those differentiating gender, then <InlineMath math="\vec{\text{king}} - \vec{\text{queen}}" /> isolates the gender direction.</li>
              <li>This same direction is shared by man/woman, uncle/aunt, etc. — so vector arithmetic captures relational structure.</li>
            </ul>
          </li>
          <li>
            <strong>GloVe vs. Word2Vec</strong>:
            <ul>
              <li>GloVe explicitly builds a co-occurrence matrix <InlineMath math="X" /> from the entire corpus, then learns vectors such that <InlineMath math="\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j \approx \log X_{ij}" />.</li>
              <li>Word2Vec is <strong>predictive</strong> (local context windows, online SGD). GloVe is <strong>count-based</strong> (global statistics, matrix factorization).</li>
              <li>Levy and Goldberg (2014) showed that Word2Vec skip-gram with negative sampling implicitly factorizes the PMI matrix — so the two approaches are theoretically connected.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Mikolov et al. (2013) &quot;Efficient Estimation of Word Representations in Vector Space&quot;</strong> — The original Word2Vec paper.</li>
          <li><strong>Pennington et al. (2014) &quot;GloVe: Global Vectors for Word Representation&quot;</strong> — The GloVe paper from Stanford NLP.</li>
          <li><strong>Levy &amp; Goldberg (2014) &quot;Neural Word Embedding as Implicit Matrix Factorization&quot;</strong> — Shows the theoretical connection between Word2Vec and co-occurrence matrix factorization.</li>
          <li><strong>Bojanowski et al. (2017) &quot;Enriching Word Vectors with Subword Information&quot;</strong> — The FastText paper, extending Word2Vec with character n-grams.</li>
          <li><strong>Stanford CS224N Lecture 1 &amp; 2</strong> — Thorough walkthrough of word vectors, GloVe derivation, and evaluation methods.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
