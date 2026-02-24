"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function InformationRetrieval() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Information retrieval (IR) is the science of finding relevant documents from a large collection
          given a user&apos;s query. Every time you type something into a search engine, an IR system
          is deciding which of billions of documents are worth showing you — and in what order.
        </p>
        <p>
          The fundamental data structure behind search is the <strong>inverted index</strong>. Instead of
          storing &quot;document → words it contains,&quot; we flip it: &quot;word → list of documents
          containing that word.&quot; This is why it&apos;s called &quot;inverted&quot; — it inverts the
          mapping. When you search for &quot;gradient descent,&quot; the system looks up the posting lists
          for &quot;gradient&quot; and &quot;descent,&quot; intersects them, and instantly finds all matching
          documents without scanning every page.
        </p>
        <p>
          But finding matching documents isn&apos;t enough — we need to <strong>rank</strong> them by
          relevance. This is where scoring functions come in. <strong>TF-IDF</strong> was the first great
          idea: words that appear frequently in a document (high term frequency) but rarely across all
          documents (high inverse document frequency) are likely important for that document.
          <strong> BM25</strong> improves on TF-IDF with diminishing returns on term frequency and document
          length normalization, and remains the default scoring function in Elasticsearch and Lucene today.
        </p>
        <p>
          Modern search systems combine these classical techniques with dense retrieval (embedding-based
          similarity), building a pipeline: first retrieve a candidate set cheaply using the inverted index,
          then re-rank with a more expensive model. Understanding the classical foundations is essential
          because they form the first stage of every production search system.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Inverted Index</h3>
        <p>
          Given a corpus of <InlineMath math="N" /> documents, each document <InlineMath math="d" /> is a
          bag of terms. The inverted index maps each term <InlineMath math="t" /> to a <strong>posting
          list</strong>: the sorted set of document IDs containing <InlineMath math="t" />, often with
          positions and term frequencies:
        </p>
        <BlockMath math="\text{index}(t) = \{(d_1, \text{tf}_{t,d_1}), (d_2, \text{tf}_{t,d_2}), \ldots\}" />

        <h3>TF-IDF</h3>
        <p>
          <strong>Term Frequency</strong>: how often term <InlineMath math="t" /> appears in
          document <InlineMath math="d" />:
        </p>
        <BlockMath math="\text{tf}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}" />

        <p>
          <strong>Inverse Document Frequency</strong>: downweights terms that appear in many documents:
        </p>
        <BlockMath math="\text{idf}(t) = \log \frac{N}{|\{d \in D : t \in d\}|}" />

        <p>The TF-IDF score for term <InlineMath math="t" /> in document <InlineMath math="d" />:</p>
        <BlockMath math="\text{tf-idf}(t, d) = \text{tf}(t, d) \cdot \text{idf}(t)" />

        <p>The relevance score of document <InlineMath math="d" /> for query <InlineMath math="q" />:</p>
        <BlockMath math="\text{score}(q, d) = \sum_{t \in q} \text{tf-idf}(t, d)" />

        <h3>BM25 (Best Match 25)</h3>
        <p>
          BM25 improves TF-IDF with <strong>saturation</strong> (diminishing returns on term frequency)
          and <strong>document length normalization</strong>:
        </p>
        <BlockMath math="\text{BM25}(q, d) = \sum_{t \in q} \text{idf}(t) \cdot \frac{f_{t,d} \cdot (k_1 + 1)}{f_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}" />
        <p>Where:</p>
        <ul>
          <li><InlineMath math="f_{t,d}" /> = raw term frequency of <InlineMath math="t" /> in <InlineMath math="d" /></li>
          <li><InlineMath math="k_1 \approx 1.2" /> controls term frequency saturation (higher = slower saturation)</li>
          <li><InlineMath math="b \approx 0.75" /> controls document length normalization (0 = no normalization, 1 = full)</li>
          <li><InlineMath math="|d|" /> = document length, <InlineMath math="\text{avgdl}" /> = average document length</li>
        </ul>
        <p>
          The key insight: in TF-IDF, if a word appears 100 times vs. 50 times, the score doubles. In BM25,
          there are diminishing returns — after a certain point, more occurrences barely increase the score.
          This matches intuition: a document mentioning &quot;machine learning&quot; 100 times isn&apos;t
          twice as relevant as one mentioning it 50 times.
        </p>

        <h3>IDF Variant in BM25</h3>
        <p>The Robertson-Sparck Jones IDF used in most BM25 implementations:</p>
        <BlockMath math="\text{idf}(t) = \log \frac{N - n_t + 0.5}{n_t + 0.5}" />
        <p>where <InlineMath math="n_t" /> is the number of documents containing term <InlineMath math="t" />.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Building an Inverted Index from Scratch</h3>
        <CodeBlock
          language="python"
          title="inverted_index.py"
          code={`import re
from collections import defaultdict
import math

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)  # term -> [(doc_id, tf), ...]
        self.doc_lengths = {}           # doc_id -> num_tokens
        self.doc_count = 0
        self.avg_doc_length = 0

    def tokenize(self, text):
        """Simple whitespace + lowercase tokenizer."""
        return re.findall(r'\\w+', text.lower())

    def add_document(self, doc_id, text):
        tokens = self.tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_count += 1
        self.avg_doc_length = (
            sum(self.doc_lengths.values()) / self.doc_count
        )

        # Count term frequencies in this document
        tf_map = defaultdict(int)
        for token in tokens:
            tf_map[token] += 1

        # Add to posting lists
        for term, freq in tf_map.items():
            self.index[term].append((doc_id, freq))

    def search_boolean(self, query):
        """Boolean AND search: return docs containing ALL query terms."""
        tokens = self.tokenize(query)
        if not tokens:
            return []

        # Get posting lists for each term
        posting_sets = []
        for token in tokens:
            doc_ids = {doc_id for doc_id, _ in self.index.get(token, [])}
            posting_sets.append(doc_ids)

        # Intersect all posting lists
        result = posting_sets[0]
        for s in posting_sets[1:]:
            result &= s
        return sorted(result)

# Build index
idx = InvertedIndex()
docs = {
    0: "the quick brown fox jumps over the lazy dog",
    1: "machine learning with gradient descent optimization",
    2: "the fox and the hound are friends",
    3: "deep learning gradient methods for optimization",
}
for doc_id, text in docs.items():
    idx.add_document(doc_id, text)

# Boolean search
print(idx.search_boolean("gradient optimization"))  # [1, 3]
print(idx.search_boolean("fox"))                     # [0, 2]`}
        />

        <h3>BM25 Scoring</h3>
        <CodeBlock
          language="python"
          title="bm25_scoring.py"
          code={`class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.index = defaultdict(list)
        self.doc_lengths = {}
        self.doc_count = 0
        self.avg_dl = 0

    def tokenize(self, text):
        return re.findall(r'\\w+', text.lower())

    def fit(self, documents):
        """Build index from dict of {doc_id: text}."""
        for doc_id, text in documents.items():
            tokens = self.tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            self.doc_count += 1

            tf_map = defaultdict(int)
            for t in tokens:
                tf_map[t] += 1
            for term, freq in tf_map.items():
                self.index[term].append((doc_id, freq))

        self.avg_dl = sum(self.doc_lengths.values()) / self.doc_count

    def idf(self, term):
        """Robertson-Sparck Jones IDF."""
        n_t = len(self.index.get(term, []))
        return math.log(
            (self.doc_count - n_t + 0.5) / (n_t + 0.5) + 1
        )

    def score(self, query):
        """Return sorted list of (doc_id, score) for the query."""
        tokens = self.tokenize(query)
        scores = defaultdict(float)

        for term in tokens:
            idf_val = self.idf(term)
            for doc_id, tf in self.index.get(term, []):
                dl = self.doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / self.avg_dl
                )
                scores[doc_id] += idf_val * numerator / denominator

        return sorted(scores.items(), key=lambda x: -x[1])

# Example
docs = {
    0: "machine learning is a subset of artificial intelligence",
    1: "deep learning uses neural networks for representation learning",
    2: "gradient descent is an optimization algorithm used in machine learning",
    3: "reinforcement learning trains agents through trial and error",
    4: "machine learning and deep learning are transforming industries",
}

bm25 = BM25(k1=1.2, b=0.75)
bm25.fit(docs)

results = bm25.score("machine learning optimization")
for doc_id, s in results:
    print(f"  Doc {doc_id} (score={s:.3f}): {docs[doc_id][:60]}...")
# Doc 2 scores highest — contains all three query terms`}
        />

        <h3>Using rank-bm25 and Elasticsearch</h3>
        <CodeBlock
          language="python"
          title="bm25_libraries.py"
          code={`# --- Option 1: rank_bm25 library (great for prototyping) ---
from rank_bm25 import BM25Okapi

corpus = [
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks for representation learning",
    "gradient descent is an optimization algorithm",
    "reinforcement learning trains agents through trial and error",
]
tokenized = [doc.lower().split() for doc in corpus]

bm25 = BM25Okapi(tokenized)
query = "machine learning optimization"
scores = bm25.get_scores(query.lower().split())
top_idx = scores.argsort()[::-1]

for i in top_idx[:3]:
    print(f"  Score {scores[i]:.3f}: {corpus[i]}")

# --- Option 2: Elasticsearch (production) ---
# from elasticsearch import Elasticsearch
# es = Elasticsearch("http://localhost:9200")
#
# # Index documents
# for i, doc in enumerate(corpus):
#     es.index(index="articles", id=i, body={"text": doc})
#
# # Search with BM25 (default scoring)
# results = es.search(
#     index="articles",
#     body={"query": {"match": {"text": "machine learning"}}}
# )
# for hit in results["hits"]["hits"]:
#     print(f"  Score {hit['_score']:.3f}: {hit['_source']['text']}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>BM25 is the baseline to beat</strong>: Before trying dense retrieval (embeddings), always benchmark against BM25. It&apos;s surprisingly hard to beat for keyword-heavy queries.</li>
          <li><strong>Two-stage retrieval</strong>: Production systems use BM25 for fast candidate retrieval (top 1000 from millions), then re-rank with a cross-encoder or learned ranker. This is the standard architecture at Google, Bing, and most search companies.</li>
          <li><strong>Analyzer matters more than the algorithm</strong>: In Elasticsearch/Solr, choosing the right tokenizer, stemmer, and stop-word list often has a bigger impact than tuning BM25 parameters.</li>
          <li><strong>Hybrid search</strong>: Combine BM25 (good for exact keyword match) with dense retrieval (good for semantic similarity). Reciprocal Rank Fusion (RRF) is a simple, effective way to merge results from both.</li>
          <li><strong>Tune <InlineMath math="k_1" /> and <InlineMath math="b" /> on your data</strong>: The defaults (1.2, 0.75) work well in general, but for short documents (tweets) use lower <InlineMath math="b" />, and for verbose queries use higher <InlineMath math="k_1" />.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Ignoring tokenization</strong>: &quot;running,&quot; &quot;runs,&quot; and &quot;ran&quot; won&apos;t match without stemming or lemmatization. This is the #1 reason simple IR systems underperform.</li>
          <li><strong>Not normalizing for document length</strong>: Without the <InlineMath math="b" /> parameter in BM25 (or equivalent), longer documents score higher simply because they contain more words, not because they&apos;re more relevant.</li>
          <li><strong>Confusing TF-IDF vectors with BM25</strong>: TF-IDF vectors (used in sklearn&apos;s <code>TfidfVectorizer</code>) represent documents in vector space for cosine similarity. BM25 is a direct scoring function. They&apos;re related but not the same.</li>
          <li><strong>Applying BM25 to semantic queries</strong>: BM25 is a lexical model — it cannot understand that &quot;car&quot; and &quot;automobile&quot; are synonymous. For semantic search, you need embeddings (dense retrieval).</li>
          <li><strong>Building your own index in production</strong>: Use Elasticsearch, OpenSearch, or Meilisearch. They handle index sharding, updates, caching, and concurrent queries at scale.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You are building a search system for an e-commerce site with 50 million products. A user searches for &quot;red running shoes under $100.&quot; Walk through how you would retrieve and rank results.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Query understanding</strong>: Parse the query to extract structured intent. &quot;red&quot; → color filter, &quot;running shoes&quot; → category, &quot;under $100&quot; → price filter. Use a query parser or NLU model.</li>
          <li><strong>Candidate retrieval (BM25)</strong>: Use the inverted index to find all products matching &quot;running shoes&quot; in the title and description fields. Apply the color and price filters as structured constraints (Elasticsearch filtered query). This narrows 50M products to perhaps 10,000 candidates in ~10ms.</li>
          <li><strong>Re-ranking</strong>: Take the top 1,000 BM25 results and re-rank with a learned model (LambdaMART or a cross-encoder) that considers:
            <ul>
              <li>Textual relevance (BM25 score as a feature)</li>
              <li>Behavioral signals (click-through rate, add-to-cart rate, purchase rate)</li>
              <li>Product quality signals (reviews, rating, return rate)</li>
              <li>Personalization (user&apos;s brand preferences, size, past purchases)</li>
            </ul>
          </li>
          <li><strong>Business rules</strong>: Boost promoted products, ensure diversity (don&apos;t show 10 items from the same seller), filter out out-of-stock items.</li>
          <li><strong>Return top 20</strong> with snippets highlighting matched terms. Total latency budget: &lt;200ms at p99.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Manning, Raghavan, Schutze &quot;Introduction to Information Retrieval&quot;</strong> — The definitive textbook (available free online at nlp.stanford.edu/IR-book).</li>
          <li><strong>Robertson &amp; Zaragoza (2009) &quot;The Probabilistic Relevance Framework: BM25 and Beyond&quot;</strong> — The theory behind BM25.</li>
          <li><strong>Elasticsearch: The Definitive Guide</strong> — Practical guide to building production search systems.</li>
          <li><strong>Karpukhin et al. (2020) &quot;Dense Passage Retrieval&quot;</strong> — DPR paper: how dense retrieval competes with BM25 for open-domain QA.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
