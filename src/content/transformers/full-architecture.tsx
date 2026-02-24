"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function FullArchitecture() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          The Transformer, introduced in &quot;Attention Is All You Need&quot; (2017), replaced recurrent
          networks entirely with <strong>self-attention</strong>. The original architecture is an
          <strong> encoder-decoder</strong> model: the encoder processes the input in parallel, and the
          decoder generates the output one token at a time.
        </p>
        <p>
          The encoder consists of <InlineMath math="N" /> identical layers (typically 6). Each layer has
          two sub-layers: (1) multi-head self-attention, and (2) a position-wise feed-forward network
          (two linear layers with a ReLU/GELU in between). Each sub-layer is wrapped with a
          <strong> residual connection</strong> and <strong>layer normalization</strong>. The output of each
          sub-layer is <InlineMath math="\text{LayerNorm}(x + \text{SubLayer}(x))" />.
        </p>
        <p>
          The decoder also has <InlineMath math="N" /> identical layers, but with an important addition:
          between the self-attention and the feed-forward network, there is a <strong>cross-attention</strong> layer.
          In cross-attention, the queries come from the decoder, while the keys and values come from the
          encoder output. This is how the decoder &quot;reads&quot; the input. The decoder&apos;s self-attention
          also uses a <strong>causal mask</strong> to prevent attending to future positions.
        </p>
        <p>
          Three architectural variants dominate modern AI: <strong>Encoder-only</strong> (BERT, for
          classification and understanding), <strong>Decoder-only</strong> (GPT, for generation), and
          <strong> Encoder-decoder</strong> (T5, for translation and summarization). Decoder-only models
          have become the dominant paradigm for large language models because they simplify training to
          next-token prediction.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Encoder Layer</h3>
        <p>Each of the <InlineMath math="N" /> encoder layers computes:</p>
        <BlockMath math="z = \text{LayerNorm}(x + \text{MultiHeadAttention}(x, x, x))" />
        <BlockMath math="\text{out} = \text{LayerNorm}(z + \text{FFN}(z))" />

        <h3>Feed-Forward Network (FFN)</h3>
        <p>Applied identically to each position (hence &quot;position-wise&quot;):</p>
        <BlockMath math="\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2" />
        <p>
          Where <InlineMath math="W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}" /> and
          <InlineMath math="W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}" />. Typically
          <InlineMath math="d_{ff} = 4 \times d_{model}" />. This expansion-then-projection acts as a
          per-token nonlinear transformation, similar to a 1x1 convolution in CNNs.
        </p>

        <h3>Decoder Layer</h3>
        <p>Each decoder layer has three sub-layers:</p>
        <BlockMath math="z_1 = \text{LayerNorm}(x + \text{CausalSelfAttention}(x, x, x))" />
        <BlockMath math="z_2 = \text{LayerNorm}(z_1 + \text{CrossAttention}(z_1, \text{enc\_out}, \text{enc\_out}))" />
        <BlockMath math="\text{out} = \text{LayerNorm}(z_2 + \text{FFN}(z_2))" />
        <p>
          In cross-attention, queries <InlineMath math="Q" /> come from the decoder, while keys
          <InlineMath math="K" /> and values <InlineMath math="V" /> come from the encoder output.
        </p>

        <h3>Causal Mask</h3>
        <p>
          The decoder self-attention applies a mask to prevent position <InlineMath math="i" /> from
          attending to positions <InlineMath math="> i" />:
        </p>
        <BlockMath math="\text{mask}_{i,j} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}" />
        <p>
          This ensures the model can only use past tokens when predicting the next one, enabling
          autoregressive generation.
        </p>

        <h3>Pre-Norm vs Post-Norm</h3>
        <p>The original paper uses <strong>Post-Norm</strong>: <InlineMath math="\text{LN}(x + f(x))" />. Most modern implementations use <strong>Pre-Norm</strong>:</p>
        <BlockMath math="\text{out} = x + f(\text{LayerNorm}(x))" />
        <p>
          Pre-Norm is more stable to train (gradients flow through the residual path without going
          through LayerNorm) but may converge to slightly worse solutions. GPT-2 and most modern LLMs
          use Pre-Norm.
        </p>

        <h3>Parameter Count</h3>
        <p>
          For a Transformer with <InlineMath math="N" /> layers, <InlineMath math="d_{model}" /> hidden
          dim, <InlineMath math="d_{ff} = 4d_{model}" />, and vocab size <InlineMath math="V" />:
        </p>
        <BlockMath math="\text{Per layer} \approx 4d_{model}^2 \; (\text{attention}) + 8d_{model}^2 \; (\text{FFN}) = 12d_{model}^2" />
        <BlockMath math="\text{Total} \approx 12 N d_{model}^2 + V \cdot d_{model} \; (\text{embeddings})" />
        <p>
          For GPT-3 (175B): <InlineMath math="N=96" />, <InlineMath math="d_{model}=12288" />,
          <InlineMath math="V=50257" />, which gives approximately 175 billion parameters.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Full Transformer Encoder-Decoder from Scratch</h3>
        <CodeBlock
          language="python"
          title="transformer.py"
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T, D = query.shape
        Q = self.W_q(query).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Pre-norm variant
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.cross_attn(x2, enc_out, enc_out, src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)

        # Sinusoidal positional encoding (registered as buffer)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask=None):
        x = self.dropout(self.src_embed(src) * math.sqrt(self.d_model) + self.pe[:, :src.size(1)])
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        x = self.dropout(self.tgt_embed(tgt) * math.sqrt(self.d_model) + self.pe[:, :tgt.size(1)])
        for layer in self.decoder_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.final_norm(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, tgt_mask, src_mask)
        return self.output_proj(dec_out)

    @staticmethod
    def make_causal_mask(size):
        """Upper-triangular mask to prevent attending to future tokens."""
        mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, size, size)

# --- Usage ---
model = Transformer(src_vocab=8000, tgt_vocab=6000, d_model=512,
                     num_heads=8, num_layers=6, d_ff=2048)

src = torch.randint(1, 8000, (2, 30))  # batch=2, src_len=30
tgt = torch.randint(1, 6000, (2, 25))  # tgt_len=25
tgt_mask = Transformer.make_causal_mask(25)

logits = model(src, tgt, tgt_mask=tgt_mask)
print(f"Output: {logits.shape}")  # (2, 25, 6000)

params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params / 1e6:.1f}M")  # ~65M`}
        />

        <h3>GPT-style Decoder-Only Transformer</h3>
        <CodeBlock
          language="python"
          title="gpt_decoder_only.py"
          code={`import torch
import torch.nn as nn
import math

class GPTBlock(nn.Module):
    """A single GPT block: Pre-Norm, causal self-attention + FFN."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                           dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, is_causal=True)[0]
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)  # learned positions
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: share embedding and output weights
        self.head.weight = self.token_embed.weight
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.dropout(self.token_embed(idx) + self.pos_embed(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)

# GPT-2 Small equivalent
gpt = GPT(vocab_size=50257, d_model=768, num_heads=12, num_layers=12)
tokens = torch.randint(0, 50257, (1, 128))
logits = gpt(tokens)
print(f"Logits: {logits.shape}")  # (1, 128, 50257)

params = sum(p.numel() for p in gpt.parameters())
print(f"GPT-2 Small params: {params / 1e6:.0f}M")  # ~124M`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Decoder-only is the dominant paradigm</strong>: GPT-2/3/4, LLaMA, Mistral, Gemini -- nearly all large language models are decoder-only. The encoder-decoder architecture is used mainly for translation (mBART, NLLB) and specialized seq2seq tasks (T5, Flan-T5).</li>
          <li><strong>Pre-Norm is standard</strong>: GPT-2 switched to Pre-Norm (<InlineMath math="x + f(\text{LN}(x))" />) for training stability. Nearly all modern Transformers follow this pattern.</li>
          <li><strong>GELU replaces ReLU</strong>: In the FFN, GELU (Gaussian Error Linear Unit) is the standard activation in GPT-2+, BERT, and modern Transformers. Some use SwiGLU (LLaMA) which adds a gating mechanism.</li>
          <li><strong>Weight tying</strong>: Sharing the input embedding and output projection matrices reduces parameter count and often improves performance. Standard in GPT-2 and most LLMs.</li>
          <li><strong>KV-cache for efficient inference</strong>: During autoregressive generation, cache the K and V matrices from previous tokens. This reduces each generation step from <InlineMath math="O(n^2)" /> to <InlineMath math="O(n)" /> but requires significant memory.</li>
          <li><strong>Flash Attention is essential</strong>: The naive attention implementation is memory-bound. Flash Attention reorders the computation to minimize HBM accesses, providing 2-4x speedup with no approximation.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Forgetting to scale embeddings</strong>: The original Transformer multiplies embeddings by <InlineMath math="\sqrt{d_{model}}" /> before adding positional encodings. Without this, positional encodings dominate the signal because embeddings are initialized with small values.</li>
          <li><strong>Getting the causal mask wrong</strong>: The mask must set future positions to <InlineMath math="-\infty" /> <em>before</em> softmax, not to 0. Setting them to 0 gives uniform attention, not zero attention.</li>
          <li><strong>Confusing self-attention and cross-attention</strong>: In self-attention, Q/K/V all come from the same sequence. In cross-attention, Q comes from the decoder, K/V from the encoder. This is the only way the decoder &quot;sees&quot; the source.</li>
          <li><strong>Not using dropout in all the right places</strong>: The original Transformer applies dropout to (1) attention weights, (2) after each sub-layer before the residual addition, and (3) to the sum of embeddings + positional encodings.</li>
          <li><strong>Thinking encoder-only models generate text</strong>: BERT-style models produce contextual embeddings, not text. For generation, you need a decoder with causal masking. BERT&apos;s [MASK] token prediction is not autoregressive generation.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Compare the three Transformer variants (encoder-only, decoder-only, encoder-decoder). What is each best suited for, and why has decoder-only become dominant for LLMs?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Encoder-only (BERT)</strong>:
            <ul>
              <li>Bidirectional self-attention -- each token sees all other tokens.</li>
              <li>Pre-trained with masked language modeling (MLM): predict randomly masked tokens.</li>
              <li>Best for: classification, NER, sentence embeddings, extractive QA.</li>
              <li>Cannot generate text autoregressively.</li>
            </ul>
          </li>
          <li>
            <strong>Decoder-only (GPT)</strong>:
            <ul>
              <li>Causal (unidirectional) self-attention -- each token only sees previous tokens.</li>
              <li>Pre-trained with next-token prediction (causal LM objective).</li>
              <li>Best for: text generation, in-context learning, general-purpose LLMs.</li>
              <li>Can handle any task by framing it as text generation (&quot;prompt in, completion out&quot;).</li>
            </ul>
          </li>
          <li>
            <strong>Encoder-decoder (T5)</strong>:
            <ul>
              <li>Encoder uses bidirectional attention on the input; decoder generates output with cross-attention to the encoder.</li>
              <li>Best for: translation, summarization, and tasks with distinct input/output sequences.</li>
              <li>More parameter-efficient for seq2seq tasks but less flexible for open-ended generation.</li>
            </ul>
          </li>
          <li>
            <strong>Why decoder-only dominates</strong>:
            <ul>
              <li>Simplicity: one objective (next-token prediction), one architecture, scales cleanly.</li>
              <li>Universality: any task can be reformulated as conditional generation.</li>
              <li>In-context learning emerges at scale, eliminating the need for task-specific fine-tuning.</li>
              <li>No need for separate encoder/decoder -- simpler infrastructure for training and serving.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Vaswani et al. (2017) &quot;Attention Is All You Need&quot;</strong> -- The original Transformer paper. Read sections 3.1-3.3 carefully for the architecture details.</li>
          <li><strong>Radford et al. (2019) &quot;Language Models are Unsupervised Multitask Learners&quot;</strong> -- GPT-2 paper showing decoder-only Transformers as general-purpose learners.</li>
          <li><strong>Raffel et al. (2020) &quot;Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer&quot;</strong> -- T5 paper comparing encoder-decoder with other variants.</li>
          <li><strong>Harvard NLP &quot;The Annotated Transformer&quot;</strong> -- Line-by-line annotated PyTorch implementation of the original Transformer. The best code walkthrough.</li>
          <li><strong>Karpathy &quot;nanoGPT&quot;</strong> -- A minimal, clean GPT implementation in ~300 lines of PyTorch. Perfect for understanding the decoder-only architecture.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
