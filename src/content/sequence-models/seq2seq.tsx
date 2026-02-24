"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Seq2Seq() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Sequence-to-Sequence (Seq2Seq) models take a variable-length input sequence and produce a
          variable-length output sequence. This architecture enabled the first truly effective neural
          machine translation systems, and its attention mechanism directly inspired the Transformer.
        </p>
        <p>
          The original Seq2Seq model (Sutskever et al., 2014) has two parts: an <strong>encoder</strong> that
          reads the entire input and compresses it into a fixed-size &quot;context vector&quot; (the final hidden
          state), and a <strong>decoder</strong> that generates the output one token at a time, conditioned on
          this context vector. The fundamental problem: squeezing an entire sentence into a single vector
          creates an information bottleneck. Long sentences lose critical details.
        </p>
        <p>
          <strong>Attention</strong> (Bahdanau et al., 2015) solves this brilliantly. Instead of relying on
          one fixed vector, the decoder can &quot;look back&quot; at all encoder hidden states and focus on the
          most relevant ones at each decoding step. When translating &quot;le chat noir&quot; to &quot;the black cat,&quot;
          when generating &quot;black,&quot; the decoder attends most strongly to &quot;noir.&quot; This was the key
          insight that paved the road to Transformers.
        </p>
        <p>
          Luong et al. (2015) later proposed a simpler <strong>dot-product attention</strong> variant,
          and the distinction between <strong>additive attention</strong> (Bahdanau) and
          <strong> multiplicative attention</strong> (Luong) became a foundational concept. The
          Transformer&apos;s scaled dot-product attention is essentially Luong attention generalized
          to multiple heads with learned projections.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Encoder</h3>
        <p>
          A bidirectional LSTM encodes the source sequence <InlineMath math="(x_1, \ldots, x_S)" /> into
          hidden states:
        </p>
        <BlockMath math="\overrightarrow{h_i} = \text{LSTM}_{\text{fwd}}(x_i, \overrightarrow{h_{i-1}})" />
        <BlockMath math="\overleftarrow{h_i} = \text{LSTM}_{\text{bwd}}(x_i, \overleftarrow{h_{i+1}})" />
        <BlockMath math="h_i = [\overrightarrow{h_i}; \overleftarrow{h_i}] \in \mathbb{R}^{2d}" />
        <p>
          The encoder outputs a sequence of annotations <InlineMath math="(h_1, \ldots, h_S)" />,
          each summarizing the input around position <InlineMath math="i" />.
        </p>

        <h3>Bahdanau (Additive) Attention</h3>
        <p>
          At each decoder step <InlineMath math="t" />, compute attention over all encoder
          states <InlineMath math="h_1, \ldots, h_S" /> using the decoder&apos;s previous hidden
          state <InlineMath math="s_{t-1}" />:
        </p>
        <BlockMath math="e_{t,i} = v^T \tanh(W_1 s_{t-1} + W_2 h_i)" />
        <BlockMath math="\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{S} \exp(e_{t,j})}" />
        <BlockMath math="c_t = \sum_{i=1}^{S} \alpha_{t,i} \, h_i" />
        <p>
          The context vector <InlineMath math="c_t" /> is a weighted average of encoder states,
          where <InlineMath math="\alpha_{t,i}" /> is the attention weight telling us how much the
          decoder focuses on source position <InlineMath math="i" /> at step <InlineMath math="t" />.
        </p>

        <h3>Luong (Multiplicative/Dot-Product) Attention</h3>
        <p>A simpler alternative that uses the current decoder state <InlineMath math="s_t" />:</p>
        <BlockMath math="e_{t,i} = s_t^T W_a h_i \quad \text{(general)}" />
        <BlockMath math="e_{t,i} = s_t^T h_i \quad \text{(dot)}" />
        <p>
          Dot-product attention is cheaper to compute (no learned parameters in the score function)
          and works well when <InlineMath math="s_t" /> and <InlineMath math="h_i" /> have the same
          dimensionality.
        </p>

        <h3>Decoder</h3>
        <p>The decoder LSTM takes the previous token, previous hidden state, and context vector:</p>
        <BlockMath math="s_t = \text{LSTM}([y_{t-1}; c_t], \, s_{t-1})" />
        <BlockMath math="P(y_t | y_{<t}, x) = \text{softmax}(W_o [s_t; c_t] + b_o)" />
        <p>
          During training, we use <strong>teacher forcing</strong>: feed the ground-truth <InlineMath math="y_{t-1}" /> as
          input. During inference, we feed the model&apos;s own previous prediction, often with
          <strong> beam search</strong> to find high-probability sequences.
        </p>

        <h3>Beam Search</h3>
        <p>
          Greedy decoding picks the most probable token at each step, which can lead to suboptimal
          sequences. Beam search maintains the top <InlineMath math="B" /> (beam width) partial
          hypotheses at each step:
        </p>
        <BlockMath math="\text{score}(y_1, \ldots, y_t) = \sum_{i=1}^{t} \log P(y_i | y_{<i}, x)" />
        <p>
          Typical beam width is 4-10. Length normalization is applied to avoid favoring shorter sequences:
        </p>
        <BlockMath math="\text{score}_{\text{normalized}} = \frac{1}{t^\alpha} \sum_{i=1}^{t} \log P(y_i | y_{<i}, x)" />
      </TopicSection>

      <TopicSection type="code">
        <h3>Seq2Seq with Attention from Scratch</h3>
        <CodeBlock
          language="python"
          title="seq2seq_attention.py"
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        # Project bidirectional hidden to decoder size
        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src):
        embedded = self.embedding(src)                # (batch, src_len, embed)
        enc_out, (h_n, c_n) = self.rnn(embedded)      # enc_out: (batch, src_len, 2*hidden)

        # Combine forward/backward final states for decoder init
        # h_n: (num_layers*2, batch, hidden)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)     # (batch, 2*hidden)
        c = torch.cat([c_n[-2], c_n[-1]], dim=-1)
        h = torch.tanh(self.fc_h(h)).unsqueeze(0)     # (1, batch, hidden)
        c = torch.tanh(self.fc_c(c)).unsqueeze(0)
        return enc_out, (h, c)

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W1 = nn.Linear(dec_dim, dec_dim, bias=False)
        self.W2 = nn.Linear(enc_dim, dec_dim, bias=False)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        # decoder_state: (batch, dec_dim)
        # encoder_outputs: (batch, src_len, enc_dim)
        query = self.W1(decoder_state).unsqueeze(1)    # (batch, 1, dec_dim)
        keys = self.W2(encoder_outputs)                 # (batch, src_len, dec_dim)
        scores = self.v(torch.tanh(query + keys))       # (batch, src_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=-1) # (batch, src_len)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)  # (batch, 1, enc_dim)
        return context.squeeze(1), weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, enc_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(enc_dim, hidden_size)
        self.rnn = nn.LSTM(embed_dim + enc_dim, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size + enc_dim + embed_dim, vocab_size)

    def forward(self, tgt_token, hidden, cell, encoder_outputs):
        # tgt_token: (batch,) -- single token
        embedded = self.embedding(tgt_token.unsqueeze(1))    # (batch, 1, embed)
        context, attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)

        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        prediction = self.fc_out(torch.cat([
            output.squeeze(1), context, embedded.squeeze(1)
        ], dim=-1))
        return prediction, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)

        enc_out, (h, c) = self.encoder(src)

        # First decoder input is <SOS> token
        dec_input = tgt[:, 0]

        for t in range(1, tgt_len):
            pred, h, c, _ = self.decoder(dec_input, h, c, enc_out)
            outputs[:, t] = pred

            # Teacher forcing: use ground truth or model prediction
            if torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = tgt[:, t]
            else:
                dec_input = pred.argmax(dim=-1)

        return outputs

# Build model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = Encoder(vocab_size=8000, embed_dim=256, hidden_size=512)
dec = Decoder(vocab_size=6000, embed_dim=256, hidden_size=512, enc_dim=1024)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

src = torch.randint(1, 8000, (4, 30)).to(DEVICE)   # batch=4, src_len=30
tgt = torch.randint(1, 6000, (4, 25)).to(DEVICE)   # tgt_len=25
output = model(src, tgt)
print(f"Output shape: {output.shape}")  # (4, 25, 6000)
params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {params / 1e6:.1f}M")`}
        />

        <h3>Greedy Decoding vs Beam Search</h3>
        <CodeBlock
          language="python"
          title="beam_search.py"
          code={`import torch
import torch.nn.functional as F

def greedy_decode(model, src, sos_idx, eos_idx, max_len=50):
    """Greedy decoding: always pick the most probable next token."""
    model.eval()
    with torch.no_grad():
        enc_out, (h, c) = model.encoder(src)
        token = torch.tensor([sos_idx], device=src.device)
        result = [sos_idx]

        for _ in range(max_len):
            pred, h, c, _ = model.decoder(token, h, c, enc_out)
            token = pred.argmax(dim=-1)
            result.append(token.item())
            if token.item() == eos_idx:
                break
    return result

def beam_search_decode(model, src, sos_idx, eos_idx, beam_width=5, max_len=50):
    """Beam search: maintain top-k hypotheses at each step."""
    model.eval()
    with torch.no_grad():
        enc_out, (h, c) = model.encoder(src)

        # Each beam: (log_prob, tokens, hidden, cell)
        beams = [(0.0, [sos_idx], h, c)]

        for _ in range(max_len):
            candidates = []
            for score, tokens, h_beam, c_beam in beams:
                if tokens[-1] == eos_idx:
                    candidates.append((score, tokens, h_beam, c_beam))
                    continue

                token = torch.tensor([tokens[-1]], device=src.device)
                pred, h_new, c_new, _ = model.decoder(token, h_beam, c_beam, enc_out)
                log_probs = F.log_softmax(pred, dim=-1).squeeze(0)

                top_probs, top_idx = log_probs.topk(beam_width)
                for prob, idx in zip(top_probs, top_idx):
                    candidates.append((
                        score + prob.item(),
                        tokens + [idx.item()],
                        h_new, c_new,
                    ))

            # Keep top-k beams
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

            # Stop if all beams ended
            if all(b[1][-1] == eos_idx for b in beams):
                break

    # Return best beam (with length normalization)
    best = max(beams, key=lambda b: b[0] / len(b[1]) ** 0.6)
    return best[1]`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Seq2Seq + attention has been superseded by Transformers</strong>: The Transformer encoder-decoder (e.g., T5, mBART) outperforms LSTM-based Seq2Seq on essentially all tasks. However, understanding Seq2Seq is critical because the Transformer is its direct descendant.</li>
          <li><strong>Teacher forcing ratio matters</strong>: 100% teacher forcing causes <strong>exposure bias</strong> -- the model never learns to recover from its own mistakes. Scheduled sampling (gradually reducing teacher forcing) or curriculum learning helps.</li>
          <li><strong>Beam search is still used everywhere</strong>: Even Transformer models use beam search for translation. Typical beam widths are 4-5. Larger beams have diminishing returns and increase latency.</li>
          <li><strong>Attention visualization is invaluable</strong>: Plotting the attention weight matrix <InlineMath math="\alpha_{t,i}" /> shows which source words the decoder focuses on at each step. For translation, you get a roughly diagonal pattern showing word alignment.</li>
          <li><strong>Copy mechanisms for rare words</strong>: Pointer networks allow the decoder to copy tokens from the source, handling out-of-vocabulary words in summarization and code generation.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Forgetting to handle the information bottleneck without attention</strong>: A vanilla Seq2Seq compresses the entire source into one vector. Performance degrades sharply for sequences longer than ~20 tokens. Always use attention.</li>
          <li><strong>Not using teacher forcing correctly</strong>: During evaluation/inference, teacher forcing must be 0 (use model predictions only). Some implementations accidentally leak ground truth during evaluation.</li>
          <li><strong>Ignoring the start-of-sequence token</strong>: The decoder needs a special &lt;SOS&gt; token to begin generation. Forgetting this causes the first output token to be wrong.</li>
          <li><strong>Not applying length normalization in beam search</strong>: Without it, beam search strongly favors shorter sequences because longer sequences accumulate more negative log-probability terms.</li>
          <li><strong>Confusing Bahdanau and Luong attention timing</strong>: Bahdanau uses the <em>previous</em> decoder state <InlineMath math="s_{t-1}" /> to compute attention, then feeds the context into the RNN. Luong uses the <em>current</em> state <InlineMath math="s_t" /> (post-RNN) and concatenates the context for prediction. Mixing these up causes subtle bugs.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> In a Seq2Seq model with attention, explain what happens at each decoder time step. What is the &quot;information bottleneck&quot; problem and how does attention solve it?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Without attention (bottleneck)</strong>:
            <ul>
              <li>The encoder reads the entire source and produces a single fixed vector <InlineMath math="c = h_S" /> (the final hidden state).</li>
              <li>The decoder must generate the entire output conditioned only on this single vector.</li>
              <li>For a 50-word sentence, all information must be compressed into ~512 dimensions. This is a severe bottleneck that causes BLEU scores to drop significantly for long sentences.</li>
            </ul>
          </li>
          <li>
            <strong>At each decoder step <InlineMath math="t" /> with attention</strong>:
            <ul>
              <li>The decoder has its previous hidden state <InlineMath math="s_{t-1}" /> and the set of all encoder hidden states <InlineMath math="\{h_1, \ldots, h_S\}" />.</li>
              <li>It computes a score for each encoder state: <InlineMath math="e_{t,i} = \text{score}(s_{t-1}, h_i)" />.</li>
              <li>Scores are normalized via softmax into attention weights <InlineMath math="\alpha_{t,i}" />.</li>
              <li>A context vector <InlineMath math="c_t = \sum_i \alpha_{t,i} h_i" /> is computed as a weighted sum of encoder states.</li>
              <li>The decoder RNN takes <InlineMath math="[y_{t-1}; c_t]" /> as input and produces the next hidden state and prediction.</li>
            </ul>
          </li>
          <li>
            <strong>Why this works</strong>: Attention gives the decoder a <em>different</em> context vector at each step, dynamically focusing on the relevant parts of the source. No information bottleneck because the full encoder sequence is always accessible.
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Sutskever et al. (2014) &quot;Sequence to Sequence Learning with Neural Networks&quot;</strong> -- The original Seq2Seq paper that started neural machine translation.</li>
          <li><strong>Bahdanau et al. (2015) &quot;Neural Machine Translation by Jointly Learning to Align and Translate&quot;</strong> -- Introduces additive attention. One of the most influential NLP papers ever.</li>
          <li><strong>Luong et al. (2015) &quot;Effective Approaches to Attention-based NMT&quot;</strong> -- Introduces dot-product (multiplicative) attention and compares attention variants.</li>
          <li><strong>Vinyals et al. (2015) &quot;Pointer Networks&quot;</strong> -- Extends attention to allow copying from the input, enabling variable output vocabularies.</li>
          <li><strong>Jay Alammar &quot;Visualizing A Neural Machine Translation Model&quot;</strong> -- Excellent visual walkthrough of Seq2Seq with attention.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
