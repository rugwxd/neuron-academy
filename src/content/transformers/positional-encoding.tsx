"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function PositionalEncoding() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Self-attention is <strong>permutation-equivariant</strong>: if you shuffle the input tokens,
          the outputs get shuffled in exactly the same way. The attention mechanism has no inherent notion
          of order. Without positional information, &quot;the dog bit the man&quot; and &quot;the man bit the
          dog&quot; would produce identical representations. Positional encoding injects sequence order
          into the model.
        </p>
        <p>
          The original Transformer uses <strong>sinusoidal positional encodings</strong>: fixed patterns
          of sines and cosines at different frequencies that are added to the token embeddings. Each
          position gets a unique &quot;fingerprint,&quot; and the model can learn to attend to relative
          positions because the difference between any two sinusoidal encodings is itself a sinusoid.
        </p>
        <p>
          <strong>Learned positional embeddings</strong> (used in GPT-2, BERT) simply treat positions
          as tokens with a learnable embedding table. These work well but limit the model to a fixed
          maximum sequence length seen during training.
        </p>
        <p>
          Modern approaches encode <strong>relative positions</strong> rather than absolute ones.
          <strong> RoPE (Rotary Position Embedding)</strong>, used in LLaMA, Mistral, and most modern
          LLMs, applies a rotation matrix to query and key vectors, making the dot product between them
          depend only on the relative distance. <strong>ALiBi (Attention with Linear Biases)</strong>
          takes an even simpler approach: it adds a fixed, non-learned linear bias to attention scores
          that penalizes distant token pairs. Both approaches can extrapolate to longer sequences than
          seen during training.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Sinusoidal Positional Encoding (Vaswani et al.)</h3>
        <p>
          For position <InlineMath math="pos" /> and dimension <InlineMath math="i" /> (where
          <InlineMath math="i \in [0, d_{model}/2)" />):
        </p>
        <BlockMath math="PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)" />
        <BlockMath math="PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)" />
        <p>
          This creates wavelengths from <InlineMath math="2\pi" /> to <InlineMath math="10000 \cdot 2\pi" />.
          A key property: for any fixed offset <InlineMath math="k" />,
          <InlineMath math="PE_{pos+k}" /> can be expressed as a linear function of <InlineMath math="PE_{pos}" />:
        </p>
        <BlockMath math="PE_{pos+k} = M_k \cdot PE_{pos}" />
        <p>
          where <InlineMath math="M_k" /> is a rotation matrix that depends only on <InlineMath math="k" />,
          not on <InlineMath math="pos" />. This enables the model to learn relative position patterns.
        </p>

        <h3>Rotary Position Embedding (RoPE)</h3>
        <p>
          RoPE applies position-dependent rotations directly to Q and K vectors. For a 2D subspace at
          position <InlineMath math="m" />:
        </p>
        <BlockMath math="R_{\Theta, m} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}" />
        <p>
          Applied to query and key vectors in pairs of dimensions:
        </p>
        <BlockMath math="q_m = R_{\Theta, m} \cdot W_q x_m, \quad k_n = R_{\Theta, n} \cdot W_k x_n" />
        <p>The dot product between rotated Q and K depends only on the relative position:</p>
        <BlockMath math="q_m^T k_n = (R_{\Theta, m} W_q x_m)^T (R_{\Theta, n} W_k x_n) = x_m^T W_q^T R_{\Theta, n-m} W_k x_n" />
        <p>
          This is because <InlineMath math="R_m^T R_n = R_{n-m}" /> (rotation by the <em>relative</em>
          distance). RoPE is elegant because it encodes relative positions without any additional
          parameters.
        </p>

        <h3>ALiBi (Attention with Linear Biases)</h3>
        <p>
          ALiBi adds a fixed penalty to the pre-softmax attention scores based on key-query distance:
        </p>
        <BlockMath math="\text{score}_{i,j} = q_i^T k_j - m \cdot |i - j|" />
        <p>
          where <InlineMath math="m" /> is a head-specific slope. For <InlineMath math="h" /> heads, the slopes are
          a geometric sequence:
        </p>
        <BlockMath math="m_k = \frac{1}{2^{8k/h}}, \quad k = 1, \ldots, h" />
        <p>
          For 8 heads: <InlineMath math="m \in \{1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256\}" />.
          Heads with steeper slopes focus more on local context; heads with gentle slopes attend more
          broadly. No learned parameters are added at all.
        </p>

        <h3>Length Extrapolation</h3>
        <p>
          The ability to handle sequences longer than those seen in training:
        </p>
        <ul>
          <li><strong>Sinusoidal</strong>: Theoretically supports any length, but in practice extrapolation is poor because the model has not learned patterns for unseen absolute positions.</li>
          <li><strong>Learned</strong>: Cannot extrapolate at all beyond the maximum position in the embedding table.</li>
          <li><strong>RoPE</strong>: Extrapolates moderately well. With techniques like NTK-aware scaling or YaRN, RoPE can be extended to much longer contexts (e.g., 4K training &rarr; 128K inference).</li>
          <li><strong>ALiBi</strong>: Extrapolates very well because the linear bias naturally extends to any distance. Trained on 1K tokens, works on 2K+ without any modification.</li>
        </ul>
      </TopicSection>

      <TopicSection type="code">
        <h3>Sinusoidal Positional Encoding</h3>
        <CodeBlock
          language="python"
          title="sinusoidal_pe.py"
          code={`import torch
import math
import matplotlib.pyplot as plt

def sinusoidal_positional_encoding(max_len, d_model):
    """
    Original Transformer positional encoding.
    Returns: (max_len, d_model) tensor.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
    # Compute the division term: 10000^(2i/d_model)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)  # even dims
    pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
    return pe

pe = sinusoidal_positional_encoding(max_len=128, d_model=64)
print(f"PE shape: {pe.shape}")  # (128, 64)

# Visualize: each row is a position, each column is a dimension
plt.figure(figsize=(12, 4))
plt.imshow(pe.numpy(), cmap="RdBu", aspect="auto")
plt.xlabel("Dimension")
plt.ylabel("Position")
plt.title("Sinusoidal Positional Encoding")
plt.colorbar()
plt.show()

# Verify: dot product similarity between positions
# Nearby positions should have higher similarity
sims = pe @ pe.T  # (128, 128)
print(f"Similarity pos 0 vs pos 1: {sims[0, 1]:.2f}")
print(f"Similarity pos 0 vs pos 50: {sims[0, 50]:.2f}")
print(f"Similarity pos 0 vs pos 127: {sims[0, 127]:.2f}")`}
        />

        <h3>Rotary Position Embedding (RoPE)</h3>
        <CodeBlock
          language="python"
          title="rope.py"
          code={`import torch

def precompute_freqs_cis(dim, max_len, theta=10000.0):
    """Precompute the complex exponentials for RoPE (LLaMA style)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)  # (max_len, dim/2)
    # Return as complex numbers: e^(i * m * theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis  # (max_len, dim/2) complex

def apply_rotary_emb(x, freqs_cis):
    """
    Apply RoPE to query or key tensor.
    x: (batch, seq_len, num_heads, head_dim)
    freqs_cis: (seq_len, head_dim/2) complex
    """
    # Reshape x as pairs of adjacent dims -> complex numbers
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )  # (batch, seq_len, num_heads, head_dim/2)

    # Broadcast freqs to match shape
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim/2)

    # Multiply (rotate in complex plane)
    x_rotated = x_complex * freqs

    # Convert back to real
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.type_as(x)

# Example usage
batch, seq_len, num_heads, head_dim = 2, 128, 8, 64
freqs_cis = precompute_freqs_cis(head_dim, seq_len)

q = torch.randn(batch, seq_len, num_heads, head_dim)
k = torch.randn(batch, seq_len, num_heads, head_dim)

q_rotated = apply_rotary_emb(q, freqs_cis)
k_rotated = apply_rotary_emb(k, freqs_cis)

print(f"Q shape after RoPE: {q_rotated.shape}")  # (2, 128, 8, 64)

# Verify relative position property: dot product depends on distance
# q[pos_a] . k[pos_b] should equal q[pos_a+d] . k[pos_b+d]
dot_0_5 = (q_rotated[0, 0, 0] * k_rotated[0, 5, 0]).sum()
dot_10_15 = (q_rotated[0, 10, 0] * k_rotated[0, 15, 0]).sum()
print(f"q[0].k[5] = {dot_0_5:.4f}")
print(f"q[10].k[15] = {dot_10_15:.4f}")  # similar (same relative dist=5)`}
        />

        <h3>ALiBi Implementation</h3>
        <CodeBlock
          language="python"
          title="alibi.py"
          code={`import torch
import math

def get_alibi_slopes(num_heads):
    """Compute ALiBi slopes: geometric sequence."""
    ratio = 2 ** (-8.0 / num_heads)
    slopes = torch.tensor([ratio ** i for i in range(1, num_heads + 1)])
    return slopes

def get_alibi_biases(seq_len, num_heads):
    """
    Compute the ALiBi bias matrix.
    Returns: (num_heads, seq_len, seq_len)
    """
    slopes = get_alibi_slopes(num_heads)  # (num_heads,)

    # Distance matrix: |i - j|
    positions = torch.arange(seq_len)
    distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()

    # Multiply each head's slope by the distance matrix
    biases = -slopes.unsqueeze(-1).unsqueeze(-1) * distances.unsqueeze(0)
    return biases  # (num_heads, seq_len, seq_len)

# Usage in attention
def alibi_attention(Q, K, V, num_heads):
    """Attention with ALiBi (no positional embeddings needed!)."""
    B, H, T, D = Q.shape
    scores = Q @ K.transpose(-2, -1) / math.sqrt(D)

    # Add ALiBi biases
    alibi = get_alibi_biases(T, num_heads).to(Q.device)  # (H, T, T)
    scores = scores + alibi.unsqueeze(0)  # broadcast over batch

    # Causal mask
    causal = torch.triu(torch.ones(T, T, device=Q.device), diagonal=1).bool()
    scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return attn @ V

# Example
B, H, T, D = 2, 8, 64, 64
Q = torch.randn(B, H, T, D)
K = torch.randn(B, H, T, D)
V = torch.randn(B, H, T, D)

out = alibi_attention(Q, K, V, num_heads=H)
print(f"Output shape: {out.shape}")  # (2, 8, 64, 64)

# Visualize ALiBi biases for first head
alibi = get_alibi_biases(32, 8)
print(f"ALiBi slopes: {get_alibi_slopes(8)}")
print(f"Head 0 bias at distance 10: {alibi[0, 0, 10]:.4f}")
print(f"Head 7 bias at distance 10: {alibi[7, 0, 10]:.4f}")  # much smaller`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>RoPE is the current standard</strong>: LLaMA 1/2/3, Mistral, Qwen, and most modern open-source LLMs use RoPE. It provides a good balance of relative position encoding and length extrapolation.</li>
          <li><strong>NTK-aware RoPE scaling for long contexts</strong>: To extend context length, scale the base frequency <InlineMath math="\theta" /> in RoPE. LLaMA-2 Long uses this to go from 4K to 32K tokens. YaRN further improves extrapolation by scaling different frequency bands differently.</li>
          <li><strong>ALiBi is simpler but less common</strong>: Used in BLOOM and MPT models. Its main advantage is zero additional parameters and excellent extrapolation, but RoPE is more widely adopted.</li>
          <li><strong>Learned positions are fine for fixed-length models</strong>: BERT (512 tokens) and GPT-2 (1024 tokens) use learned positional embeddings. They work well when you never need to exceed the training length.</li>
          <li><strong>Sinusoidal encodings are rarely used in practice</strong>: Despite being theoretically elegant, learned or rotary encodings consistently outperform fixed sinusoidal ones. The original Transformer paper itself noted that learned and sinusoidal performed similarly.</li>
          <li><strong>Position encoding is added only once</strong>: Positional information is injected at the input layer (added to embeddings). Through self-attention, this position information propagates to all subsequent layers.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Forgetting that attention is permutation-equivariant</strong>: Without positional encoding, a Transformer treats &quot;A B C&quot; identically to &quot;C B A&quot;. If your model produces weird results, check that positional encodings are being applied.</li>
          <li><strong>Trying to extend learned positional embeddings naively</strong>: You cannot just index beyond the embedding table size. For longer sequences, you need interpolation (position interpolation), RoPE with NTK scaling, or ALiBi.</li>
          <li><strong>Applying RoPE to the value vectors</strong>: RoPE should only be applied to Q and K, not V. The values should remain position-independent; only the attention <em>scores</em> should be position-aware.</li>
          <li><strong>Not scaling embeddings before adding positional encodings</strong>: Token embeddings are typically initialized with variance <InlineMath math="\sim 1/d" />, making their norm <InlineMath math="\sim 1" />. Sinusoidal PEs have norm <InlineMath math="\sim \sqrt{d/2}" />. Without scaling embeddings by <InlineMath math="\sqrt{d}" />, positional information dominates.</li>
          <li><strong>Confusing absolute and relative position methods</strong>: Sinusoidal and learned embeddings encode absolute positions. RoPE and ALiBi encode relative positions. For tasks requiring length generalization, relative methods are strongly preferred.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Why does a Transformer need positional encoding? Compare sinusoidal, learned, RoPE, and ALiBi approaches. Which would you choose for a model that needs to handle variable-length inputs at inference?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Why it&apos;s needed</strong>: Self-attention computes <InlineMath math="Q K^T" />, a set of dot products. Dot products are commutative: the order of tokens does not affect the attention scores. Without position information, &quot;dog bites man&quot; = &quot;man bites dog&quot; to the model.
          </li>
          <li>
            <strong>Sinusoidal (fixed)</strong>: Deterministic, no extra parameters, theoretically infinite length. But models trained with them struggle to extrapolate because they never saw certain absolute positions during training.
          </li>
          <li>
            <strong>Learned (absolute)</strong>: A lookup table of size <InlineMath math="\text{max\_len} \times d" />. Flexible and easy to implement but hard-capped at max_len. Used in BERT/GPT-2.
          </li>
          <li>
            <strong>RoPE (relative)</strong>: Rotates Q and K so that their dot product depends on relative distance <InlineMath math="m - n" />, not absolute positions. No extra parameters. With NTK-aware scaling or YaRN, extrapolates well from 4K to 128K+ tokens.
          </li>
          <li>
            <strong>ALiBi (relative)</strong>: Adds <InlineMath math="-m|i-j|" /> to attention logits. Zero parameters, excellent extrapolation. Simpler than RoPE but provides slightly less representational power.
          </li>
          <li>
            <strong>Recommendation for variable-length</strong>: <strong>RoPE</strong> with NTK-aware scaling. It&apos;s the industry standard (LLaMA, Mistral), has strong empirical results, and the scaling techniques for context extension are well-established. If simplicity is paramount and slight accuracy trade-offs are acceptable, ALiBi is a solid alternative.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Vaswani et al. (2017) &quot;Attention Is All You Need&quot;</strong> -- Section 3.5 introduces sinusoidal positional encodings.</li>
          <li><strong>Su et al. (2021) &quot;RoFormer: Enhanced Transformer with Rotary Position Embedding&quot;</strong> -- The RoPE paper. Elegant mathematical derivation.</li>
          <li><strong>Press et al. (2022) &quot;Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation&quot;</strong> -- The ALiBi paper. Clear presentation of the length extrapolation problem.</li>
          <li><strong>Peng et al. (2023) &quot;YaRN: Efficient Context Window Extension of Large Language Models&quot;</strong> -- State-of-the-art RoPE scaling for context extension.</li>
          <li><strong>Chen et al. (2023) &quot;Extending Context Window of Large Language Models via Positional Interpolation&quot;</strong> -- Simple method to extend context by interpolating RoPE frequencies.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
