"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";
import AttentionViz from "@/components/viz/AttentionViz";

export default function SelfAttention() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Self-attention is the mechanism that lets a model look at <strong>all parts of an input simultaneously</strong> and
          decide which parts are most relevant to each other. It&apos;s the core innovation that makes Transformers work.
        </p>
        <p>
          Imagine reading the sentence: &quot;The cat sat on the mat because <strong>it</strong> was tired.&quot;
          What does &quot;it&quot; refer to? You naturally look back at all the previous words and figure out that
          &quot;it&quot; = &quot;the cat&quot;. That&apos;s attention — for each word, the model computes how much
          it should &quot;attend to&quot; every other word.
        </p>
        <p>
          The brilliant insight: for each token, create three vectors:
        </p>
        <ul>
          <li><strong>Query (Q)</strong>: &quot;What am I looking for?&quot;</li>
          <li><strong>Key (K)</strong>: &quot;What do I contain?&quot;</li>
          <li><strong>Value (V)</strong>: &quot;What information do I provide?&quot;</li>
        </ul>
        <p>
          The attention weight between two tokens is determined by how well the <strong>query</strong> of one
          matches the <strong>key</strong> of the other. Then the output is a weighted sum of the <strong>values</strong>.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Scaled Dot-Product Attention</h3>
        <p>Given input embeddings <InlineMath math="X \in \mathbb{R}^{n \times d}" /> (n tokens, d dimensions):</p>
        <BlockMath math="Q = XW_Q, \quad K = XW_K, \quad V = XW_V" />
        <p>where <InlineMath math="W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}" /> are learned projection matrices.</p>

        <p>The attention scores and output:</p>
        <BlockMath math="\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V" />

        <p>Step by step:</p>
        <ol>
          <li><InlineMath math="QK^T \in \mathbb{R}^{n \times n}" /> — dot product between all query-key pairs (how relevant is token j to token i?)</li>
          <li>Scale by <InlineMath math="\frac{1}{\sqrt{d_k}}" /> — prevents dot products from getting too large as dimensions grow (which would make softmax saturate)</li>
          <li>Softmax row-wise — converts scores to probabilities (each row sums to 1)</li>
          <li>Multiply by <InlineMath math="V" /> — weighted sum of value vectors using attention weights</li>
        </ol>

        <h3>Multi-Head Attention</h3>
        <p>Instead of one attention function, use <InlineMath math="h" /> parallel attention &quot;heads&quot;:</p>
        <BlockMath math="\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O" />
        <BlockMath math="\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)" />
        <p>
          Each head can learn to attend to different types of information (one head might focus on syntax,
          another on coreference, another on positional proximity).
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Self-Attention from Scratch</h3>
        <CodeBlock
          language="python"
          title="self_attention_scratch.py"
          code={`import torch
import torch.nn.functional as F

def self_attention(X, W_q, W_k, W_v):
    """
    Single-head self-attention from scratch.

    Args:
        X: input embeddings (batch, seq_len, d_model)
        W_q, W_k, W_v: projection matrices (d_model, d_k)
    """
    Q = X @ W_q  # (batch, seq_len, d_k)
    K = X @ W_k  # (batch, seq_len, d_k)
    V = X @ W_v  # (batch, seq_len, d_k)

    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # (batch, seq_len, seq_len)

    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)  # each row sums to 1

    # Weighted sum of values
    output = attn_weights @ V  # (batch, seq_len, d_k)

    return output, attn_weights

# Example
batch_size, seq_len, d_model, d_k = 1, 7, 64, 64
X = torch.randn(batch_size, seq_len, d_model)
W_q = torch.randn(d_model, d_k)
W_k = torch.randn(d_model, d_k)
W_v = torch.randn(d_model, d_k)

output, weights = self_attention(X, W_q, W_k, W_v)
print(f"Output shape: {output.shape}")    # (1, 7, 64)
print(f"Attention weights shape: {weights.shape}")  # (1, 7, 7)
print(f"Weights sum per row: {weights[0].sum(dim=-1)}")  # all 1.0`}
        />

        <h3>Multi-Head Attention in PyTorch</h3>
        <CodeBlock
          language="python"
          title="multihead_attention.py"
          code={`import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape

        # Project and reshape for multi-head
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len, d_k)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ V  # (batch, num_heads, seq_len, d_k)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.W_o(out)

# Usage
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10
output = mha(x)
print(f"Output shape: {output.shape}")  # (2, 10, 512)`}
        />
      </TopicSection>

      <TopicSection type="see-it">
        <p className="mb-4">
          This heatmap shows <strong>attention weights</strong> for the sentence &quot;The cat sat on the mat [EOS]&quot;.
          Each row shows how much a <strong>query</strong> token attends to each <strong>key</strong> token.
          Click on tokens to highlight their attention pattern. Adjust <strong>temperature</strong> to see
          how sharper/softer attention changes the distribution.
        </p>
        <AttentionViz />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Complexity</strong>: Self-attention is <InlineMath math="O(n^2 d)" /> in sequence length. This is why context windows are limited (4K, 8K, 128K tokens = very expensive).</li>
          <li><strong>Flash Attention</strong>: An IO-aware implementation that computes exact attention in <InlineMath math="O(n^2)" /> but with much better memory access patterns. Used in all modern LLMs.</li>
          <li><strong>Causal masking</strong>: In GPT-style (decoder) models, each token can only attend to previous tokens. Implemented by setting future positions to <InlineMath math="-\infty" /> before softmax.</li>
          <li><strong>KV cache</strong>: During inference, cache the K and V matrices for previously generated tokens to avoid recomputing them. This makes autoregressive generation efficient.</li>
          <li><strong>The scaling factor matters</strong>: Without <InlineMath math="\frac{1}{\sqrt{d_k}}" />, dot products grow with dimension, pushing softmax into saturated regions where gradients vanish.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Forgetting the scaling factor</strong>: <InlineMath math="QK^T" /> without <InlineMath math="\frac{1}{\sqrt{d_k}}" /> leads to extremely peaked softmax distributions and poor training.</li>
          <li><strong>Confusing self-attention with cross-attention</strong>: In self-attention, Q, K, V all come from the same sequence. In cross-attention (encoder-decoder), Q comes from the decoder and K, V from the encoder.</li>
          <li><strong>Not applying causal mask in decoder models</strong>: Without it, the model can &quot;cheat&quot; by looking at future tokens during training.</li>
          <li><strong>Thinking each head is independent</strong>: Heads share the same input and their outputs are concatenated and linearly projected. They learn to specialize through training.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Why do Transformers use multi-head attention instead of a single large attention head? What is the computational cost?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Why multiple heads</strong>:
            <ul>
              <li>Each head operates in a lower-dimensional subspace (<InlineMath math="d_k = d_{model}/h" />), allowing it to specialize in different types of relationships.</li>
              <li>One head might learn positional proximity, another syntactic dependencies, another semantic similarity.</li>
              <li>The concatenation + output projection allows the model to combine different types of attention patterns.</li>
            </ul>
          </li>
          <li><strong>Cost</strong>: Multi-head attention with <InlineMath math="h" /> heads has the <strong>same</strong> computational cost as single-head attention with full <InlineMath math="d_k = d_{model}" />:
            <ul>
              <li>Single head: <InlineMath math="O(n^2 \cdot d_{model})" /></li>
              <li>Multi-head: <InlineMath math="h \cdot O(n^2 \cdot d_{model}/h) = O(n^2 \cdot d_{model})" /></li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Vaswani et al. (2017) &quot;Attention Is All You Need&quot;</strong> — The original Transformer paper. Required reading.</li>
          <li><strong>Jay Alammar &quot;The Illustrated Transformer&quot;</strong> — The best visual walkthrough of the architecture.</li>
          <li><strong>Dao et al. (2022) &quot;FlashAttention&quot;</strong> — IO-aware exact attention that enables longer contexts.</li>
          <li><strong>Karpathy &quot;Let&apos;s build GPT from scratch&quot;</strong> — YouTube video coding a Transformer from zero.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
