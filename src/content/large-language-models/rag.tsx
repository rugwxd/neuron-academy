"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function RAG() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Large language models have a fundamental limitation: their knowledge is frozen at training time. Ask a model about
          events after its training cutoff, about your company&apos;s internal documents, or about rapidly changing facts, and it
          will either hallucinate an answer or admit it doesn&apos;t know. <strong>Retrieval-Augmented Generation (RAG)</strong> solves
          this by giving the model access to an external knowledge base at inference time.
        </p>
        <p>
          The core idea is simple: before generating an answer, <strong>retrieve</strong> relevant documents from a knowledge base
          and include them in the prompt. The model then <strong>generates</strong> its answer grounded in the retrieved context.
          Instead of relying solely on memorized knowledge, the model can cite and reason over current, domain-specific information.
        </p>
        <p>
          A RAG pipeline has three stages: (1) <strong>Indexing</strong> — chunk documents and convert them to vector embeddings stored
          in a vector database, (2) <strong>Retrieval</strong> — given a user query, find the most relevant chunks using semantic
          similarity search, and (3) <strong>Generation</strong> — feed the query and retrieved context to the LLM to produce a
          grounded answer.
        </p>
        <p>
          RAG has become the standard architecture for building LLM applications over private data. It&apos;s used in customer support
          bots, enterprise search, code assistants, and any application where the model needs access to information not in its training
          data. Compared to fine-tuning, RAG is cheaper, doesn&apos;t require retraining, and the knowledge base can be updated instantly.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>RAG Formulation</h3>
        <p>
          Given a query <InlineMath math="q" />, a retriever <InlineMath math="R" />, and a generator <InlineMath math="G" />, the RAG
          output marginalizes over retrieved documents:
        </p>
        <BlockMath math="P(y \mid q) = \sum_{d \in \text{top-}k} P_R(d \mid q) \cdot P_G(y \mid q, d)" />
        <p>
          In practice, the simplified version concatenates the top-k documents into the context and generates:
        </p>
        <BlockMath math="y = G(q, [d_1, d_2, \ldots, d_k])" />

        <h3>Dense Retrieval</h3>
        <p>
          An embedding model <InlineMath math="E" /> maps both queries and documents to a shared vector space.
          Relevance is measured by cosine similarity or dot product:
        </p>
        <BlockMath math="\text{score}(q, d) = \frac{E(q) \cdot E(d)}{\|E(q)\| \|E(d)\|}" />
        <p>
          The top-k documents are retrieved using approximate nearest neighbor (ANN) search:
        </p>
        <BlockMath math="\{d_1, \ldots, d_k\} = \text{top-}k_{d \in \mathcal{C}} \; \text{score}(q, d)" />
        <p>
          where <InlineMath math="\mathcal{C}" /> is the corpus of all indexed document chunks.
        </p>

        <h3>Chunking Strategy</h3>
        <p>
          Documents are split into chunks of size <InlineMath math="n" /> tokens with overlap <InlineMath math="o" />:
        </p>
        <BlockMath math="\text{chunks}(D) = \{D[i \cdot (n-o) : i \cdot (n-o) + n] \mid i = 0, 1, \ldots\}" />
        <p>
          The overlap ensures that information at chunk boundaries is not lost. Typical values: <InlineMath math="n = 512" /> tokens,
          <InlineMath math="o = 50" /> tokens.
        </p>

        <h3>Hybrid Search</h3>
        <p>
          Combining sparse (BM25) and dense retrieval with Reciprocal Rank Fusion:
        </p>
        <BlockMath math="\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}" />
        <p>
          where <InlineMath math="k" /> is a constant (typically 60) and <InlineMath math="\text{rank}_r(d)" /> is the position of
          document <InlineMath math="d" /> in ranker <InlineMath math="r" />&apos;s results. This captures both lexical (exact keyword)
          and semantic (meaning-based) relevance.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Complete RAG Pipeline with LangChain</h3>
        <CodeBlock
          language="python"
          title="rag_langchain.py"
          code={`from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# 1. LOAD DOCUMENTS
loader = PyPDFLoader("company_handbook.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# 2. CHUNK DOCUMENTS
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # ~500 characters per chunk
    chunk_overlap=50,     # 50 char overlap between chunks
    separators=["\\n\\n", "\\n", ". ", " ", ""],  # split hierarchy
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# 3. EMBED AND INDEX
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save for later use
vectorstore.save_local("company_index")

# 4. RETRIEVE AND GENERATE
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},  # retrieve top 4 chunks
)

# Query the RAG pipeline
query = "What is the company's remote work policy?"
docs = retriever.get_relevant_documents(query)
print(f"\\nRetrieved {len(docs)} documents:")
for i, doc in enumerate(docs):
    print(f"  [{i}] {doc.page_content[:100]}...")`}
        />

        <h3>RAG from Scratch with Embeddings and FAISS</h3>
        <CodeBlock
          language="python"
          title="rag_scratch.py"
          code={`import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch

class SimpleRAG:
    def __init__(self, embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.model = AutoModel.from_pretrained(embed_model)
        self.chunks = []
        self.index = None

    def embed(self, texts):
        """Compute embeddings using mean pooling."""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling over token embeddings
        mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        return embeddings.numpy().astype("float32")

    def index_documents(self, documents, chunk_size=500, overlap=50):
        """Chunk documents and build FAISS index."""
        # Simple chunking by characters
        for doc in documents:
            for i in range(0, len(doc), chunk_size - overlap):
                chunk = doc[i:i + chunk_size]
                if len(chunk) > 50:  # skip tiny chunks
                    self.chunks.append(chunk)

        print(f"Indexing {len(self.chunks)} chunks...")
        embeddings = self.embed(self.chunks)
        dim = embeddings.shape[1]

        # Build FAISS index with L2 distance
        self.index = faiss.IndexFlatIP(dim)  # inner product (cosine after normalization)
        faiss.normalize_L2(embeddings)        # normalize for cosine similarity
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors")

    def retrieve(self, query, k=4):
        """Retrieve top-k most relevant chunks."""
        query_vec = self.embed([query])
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({"chunk": self.chunks[idx], "score": float(score)})
        return results

    def generate_prompt(self, query, k=4):
        """Build the augmented prompt with retrieved context."""
        results = self.retrieve(query, k)
        context = "\\n\\n".join([f"[Source {i+1}]: {r['chunk']}"
                               for i, r in enumerate(results)])
        prompt = f"""Answer the question based on the provided context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""
        return prompt

# Usage
rag = SimpleRAG()
docs = ["Your document text here...", "Another document..."]
rag.index_documents(docs)
prompt = rag.generate_prompt("What is the refund policy?")
print(prompt)
# Feed this prompt to any LLM (OpenAI, local model, etc.)`}
        />

        <h3>Evaluation: Retrieval Quality Metrics</h3>
        <CodeBlock
          language="python"
          title="rag_evaluation.py"
          code={`import numpy as np

def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of top-k retrieved docs that are relevant."""
    top_k = retrieved_ids[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_ids))
    return relevant_in_top_k / k

def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of relevant docs that appear in top-k."""
    top_k = retrieved_ids[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_ids))
    return relevant_in_top_k / len(relevant_ids) if relevant_ids else 0

def mrr(retrieved_ids, relevant_ids):
    """Mean Reciprocal Rank: 1/rank of first relevant document."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved_ids, relevance_scores, k):
    """Normalized Discounted Cumulative Gain."""
    dcg = sum(
        relevance_scores.get(doc_id, 0) / np.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
    )
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(
        score / np.log2(i + 2)
        for i, score in enumerate(ideal_scores)
    )
    return dcg / idcg if idcg > 0 else 0

# Example evaluation
retrieved = ["doc_3", "doc_1", "doc_7", "doc_2", "doc_5"]
relevant = {"doc_1", "doc_2", "doc_4"}

print(f"Precision@3: {precision_at_k(retrieved, relevant, 3):.3f}")   # 0.333
print(f"Recall@3:    {recall_at_k(retrieved, relevant, 3):.3f}")      # 0.333
print(f"MRR:         {mrr(retrieved, relevant):.3f}")                  # 0.500`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Chunking is the #1 lever</strong>: Bad chunking ruins everything downstream. Chunk by semantic boundaries (paragraphs, sections, headers) rather than fixed character counts. Use RecursiveCharacterTextSplitter with appropriate separators for your document type.</li>
          <li><strong>Embedding model matters more than LLM</strong>: If retrieval returns irrelevant chunks, the best LLM in the world can&apos;t produce a good answer. Invest in embedding model quality — test models from the MTEB leaderboard for your domain.</li>
          <li><strong>Hybrid search beats dense-only</strong>: Dense retrieval misses exact keyword matches; BM25 misses semantic similarity. Combining both with reciprocal rank fusion consistently outperforms either alone.</li>
          <li><strong>Reranking improves precision</strong>: After retrieving top-50 with fast ANN search, apply a cross-encoder reranker to select the best top-5. Cross-encoders are more accurate but too slow for the initial retrieval pass.</li>
          <li><strong>Metadata filtering</strong>: Attach metadata (date, source, category) to chunks and filter before semantic search. Asking about Q4 earnings? Filter to Q4 documents before retrieving.</li>
          <li><strong>Always include source attribution</strong>: RAG&apos;s killer feature over pure generation is that you can show users exactly which documents the answer came from. Always surface the source chunks.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Chunks too large</strong>: Embedding models have limited context windows (typically 512 tokens). Chunks exceeding this get truncated, losing information. Even within the window, larger chunks dilute the relevance signal.</li>
          <li><strong>Chunks too small</strong>: Tiny chunks lack context. A sentence like &quot;The policy was updated in 2024&quot; is meaningless without knowing which policy. Use enough overlap to preserve context.</li>
          <li><strong>Not testing retrieval separately</strong>: Many teams evaluate only the final generated answer. If retrieval fails, generation can&apos;t compensate. Measure retrieval precision/recall independently before tuning the generation prompt.</li>
          <li><strong>Ignoring the lost-in-the-middle problem</strong>: LLMs pay most attention to the beginning and end of long contexts, neglecting information in the middle. Put the most relevant chunks first, or limit to fewer high-quality chunks.</li>
          <li><strong>Using the same embedding model for queries and documents</strong>: Some embedding models are asymmetric — they expect queries and passages to be encoded differently (e.g., adding &quot;query: &quot; or &quot;passage: &quot; prefixes). Check your model&apos;s documentation.</li>
          <li><strong>No fallback for retrieval failures</strong>: If the knowledge base doesn&apos;t contain the answer, RAG should say so rather than hallucinating. Add instructions in the prompt: &quot;If the context doesn&apos;t contain the answer, say you don&apos;t know.&quot;</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Design a RAG system for a company&apos;s internal knowledge base. Walk through the architecture, key design decisions, and how you would evaluate it.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Indexing pipeline</strong>:
            <ul>
              <li>Ingest documents from various sources (Confluence, Google Docs, Slack, PDFs) using document loaders.</li>
              <li>Parse and clean: extract text, handle tables/images, remove boilerplate headers/footers.</li>
              <li>Chunk with RecursiveCharacterTextSplitter (~500 tokens, 50 token overlap), respecting document structure (don&apos;t split mid-table or mid-code-block).</li>
              <li>Embed using a strong model (e.g., BGE-large, E5, or domain-fine-tuned model). Store in a vector DB (Pinecone, Weaviate, Qdrant) with metadata (source URL, date, author, department).</li>
            </ul>
          </li>
          <li>
            <strong>Retrieval pipeline</strong>:
            <ul>
              <li>Hybrid search: BM25 (keyword) + dense retrieval (semantic), fused with RRF.</li>
              <li>Metadata pre-filtering when the query implies a specific source or time range.</li>
              <li>Cross-encoder reranking: retrieve top-50, rerank to top-5.</li>
              <li>Query expansion: use the LLM to rephrase ambiguous queries before retrieval.</li>
            </ul>
          </li>
          <li>
            <strong>Generation</strong>:
            <ul>
              <li>Construct a prompt with the query and top-k chunks, with clear instructions to cite sources and acknowledge when information is missing.</li>
              <li>Use a capable LLM (GPT-4, Claude) with low temperature for factual accuracy.</li>
            </ul>
          </li>
          <li>
            <strong>Evaluation</strong>:
            <ul>
              <li><strong>Retrieval</strong>: Precision@k, Recall@k, MRR on a labeled query-document relevance set.</li>
              <li><strong>Generation</strong>: Faithfulness (does the answer match the retrieved context?), relevance (does it answer the question?), and groundedness (no hallucinated claims beyond the sources).</li>
              <li>Use LLM-as-judge evaluation (RAGAS framework) for scalable assessment.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Lewis et al. (2020) &quot;Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks&quot;</strong> — The original RAG paper from Facebook AI.</li>
          <li><strong>Gao et al. (2024) &quot;Retrieval-Augmented Generation for Large Language Models: A Survey&quot;</strong> — Comprehensive survey of RAG techniques and architectures.</li>
          <li><strong>LangChain and LlamaIndex documentation</strong> — The two most popular frameworks for building RAG applications.</li>
          <li><strong>MTEB Leaderboard</strong> — Massive Text Embedding Benchmark for comparing retrieval embedding models.</li>
          <li><strong>RAGAS framework</strong> — Automated evaluation framework for RAG pipelines measuring faithfulness, relevance, and context quality.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
