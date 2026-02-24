"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function RNNsAndLSTMs() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A Recurrent Neural Network (RNN) is a neural network designed for <strong>sequential data</strong>.
          Unlike a feedforward network that processes each input independently, an RNN maintains a
          <strong> hidden state</strong> that gets updated at every time step. Think of it as the network&apos;s
          &quot;memory&quot; of what it has seen so far.
        </p>
        <p>
          The problem with vanilla RNNs is the <strong>vanishing gradient problem</strong>. When you
          backpropagate through many time steps, gradients get multiplied by the same weight matrix
          repeatedly. If the largest eigenvalue of that matrix is less than 1, gradients shrink
          exponentially and the network cannot learn long-range dependencies. If it&apos;s greater than
          1, gradients explode.
        </p>
        <p>
          <strong>Long Short-Term Memory (LSTM)</strong> networks solve this elegantly with a system
          of <strong>gates</strong>. The key insight is a <strong>cell state</strong> -- a separate
          memory highway that information can flow through without being transformed at every step.
          Three gates control what information enters, exits, and stays in the cell state:
        </p>
        <ul>
          <li><strong>Forget gate</strong>: Decides what to remove from cell state (&quot;Forget that we were in a question&quot;)</li>
          <li><strong>Input gate</strong>: Decides what new information to store (&quot;Remember this new subject&quot;)</li>
          <li><strong>Output gate</strong>: Decides what to output based on cell state (&quot;Output the verb tense&quot;)</li>
        </ul>
        <p>
          The <strong>GRU (Gated Recurrent Unit)</strong> is a simplified version that merges the cell
          state and hidden state into one, using only two gates (reset and update). It&apos;s faster
          to train and often performs comparably to LSTM.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Vanilla RNN</h3>
        <p>At each time step <InlineMath math="t" />:</p>
        <BlockMath math="h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)" />
        <BlockMath math="y_t = W_{hy} h_t + b_y" />
        <p>
          Where <InlineMath math="h_t \in \mathbb{R}^{d_h}" /> is the hidden state,
          <InlineMath math="x_t \in \mathbb{R}^{d_x}" /> is the input, and
          <InlineMath math="y_t" /> is the output.
        </p>

        <h3>Why Gradients Vanish</h3>
        <p>
          Backpropagation through time (BPTT) computes the gradient through the chain rule. The
          gradient of the loss at time <InlineMath math="T" /> with respect to hidden state at time <InlineMath math="t" />:
        </p>
        <BlockMath math="\frac{\partial \mathcal{L}_T}{\partial h_t} = \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}" />
        <p>
          Each Jacobian <InlineMath math="\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(\tanh'(z_k)) \cdot W_{hh}" />.
          Since <InlineMath math="\tanh'(z) \leq 1" />, if <InlineMath math="\|W_{hh}\| < 1" />, the
          product of <InlineMath math="T - t" /> such terms vanishes exponentially.
        </p>

        <h3>LSTM Equations</h3>
        <p>All four operations at time step <InlineMath math="t" />, given input <InlineMath math="x_t" />, previous hidden state <InlineMath math="h_{t-1}" />, and previous cell state <InlineMath math="c_{t-1}" />:</p>

        <p><strong>Forget gate</strong> (what to erase from cell state):</p>
        <BlockMath math="f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)" />

        <p><strong>Input gate</strong> (what new info to write):</p>
        <BlockMath math="i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)" />

        <p><strong>Candidate cell state</strong> (the new info):</p>
        <BlockMath math="\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)" />

        <p><strong>Cell state update</strong> (the key equation):</p>
        <BlockMath math="c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t" />

        <p><strong>Output gate</strong> (what to expose as hidden state):</p>
        <BlockMath math="o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)" />
        <BlockMath math="h_t = o_t \odot \tanh(c_t)" />

        <p>
          The cell state update is additive (<InlineMath math="c_t = f_t \odot c_{t-1} + \ldots" />),
          which means gradients flow through the cell state with a multiplicative factor
          of <InlineMath math="f_t" /> per step, not a product of weight matrices. When
          <InlineMath math="f_t \approx 1" />, gradients pass through essentially unchanged -- solving
          the vanishing gradient problem.
        </p>

        <h3>GRU Equations (Simplified Alternative)</h3>
        <BlockMath math="z_t = \sigma(W_z [h_{t-1}, x_t])" />
        <BlockMath math="r_t = \sigma(W_r [h_{t-1}, x_t])" />
        <BlockMath math="\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t])" />
        <BlockMath math="h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t" />
        <p>
          The update gate <InlineMath math="z_t" /> serves as both forget and input gate.
          The reset gate <InlineMath math="r_t" /> controls how much history to incorporate into the candidate.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Vanilla RNN from Scratch</h3>
        <CodeBlock
          language="python"
          title="rnn_scratch.py"
          code={`import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        Returns: all hidden states (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device)
        outputs = []

        for t in range(seq_len):
            h = self.tanh(self.W_xh(x[:, t]) + self.W_hh(h))
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (batch, seq_len, hidden)

# Test
rnn = VanillaRNN(input_size=10, hidden_size=32)
x = torch.randn(4, 20, 10)  # batch=4, seq_len=20, features=10
out = rnn(x)
print(f"Output shape: {out.shape}")  # (4, 20, 32)`}
        />

        <h3>LSTM from Scratch</h3>
        <CodeBlock
          language="python"
          title="lstm_scratch.py"
          code={`import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # All four gates in one matrix multiply for efficiency
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """Single time step."""
        combined = torch.cat([h_prev, x_t], dim=-1)
        gate_values = self.gates(combined)  # (batch, 4 * hidden)

        # Split into four gates
        i, f, g, o = gate_values.chunk(4, dim=-1)

        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        g = torch.tanh(g)      # candidate cell state
        o = torch.sigmoid(o)   # output gate

        c = f * c_prev + i * g  # cell state update (ADDITIVE!)
        h = o * torch.tanh(c)   # hidden state output

        return h, c

class LSTMFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device)
        c = torch.zeros(batch, self.hidden_size, device=x.device)
        outputs = []

        for t in range(seq_len):
            h, c = self.cell(x[:, t], h, c)
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1), (h, c)

# Compare with PyTorch built-in
custom_lstm = LSTMFromScratch(input_size=10, hidden_size=64)
pytorch_lstm = nn.LSTM(input_size=10, hidden_size=64, batch_first=True)

x = torch.randn(4, 50, 10)
out_custom, (h_c, c_c) = custom_lstm(x)
out_pytorch, (h_p, c_p) = pytorch_lstm(x)

print(f"Custom  output: {out_custom.shape}")   # (4, 50, 64)
print(f"PyTorch output: {out_pytorch.shape}")   # (4, 50, 64)`}
        />

        <h3>Practical: Sentiment Classification with LSTM</h3>
        <CodeBlock
          language="python"
          title="sentiment_lstm.py"
          code={`import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_factor, num_classes),
        )

    def forward(self, input_ids):
        """input_ids: (batch, seq_len) integer token IDs."""
        embeds = self.embedding(input_ids)           # (batch, seq_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(embeds)     # h_n: (num_layers*dirs, batch, hidden)

        # Concatenate final forward and backward hidden states
        if self.lstm.bidirectional:
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final_hidden = h_n[-1]

        return self.classifier(final_hidden)  # (batch, num_classes)

# Example
model = SentimentLSTM(vocab_size=10000, embed_dim=128,
                       hidden_size=256, num_classes=2)
tokens = torch.randint(0, 10000, (8, 100))  # batch=8, seq_len=100
logits = model(tokens)
print(f"Logits shape: {logits.shape}")  # (8, 2)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>LSTMs are largely replaced by Transformers</strong>: For most NLP tasks, Transformers outperform LSTMs. However, LSTMs remain useful for streaming/online tasks where you process one token at a time with constant memory.</li>
          <li><strong>Always use bidirectional LSTMs for classification</strong>: When you have the full sequence available (not autoregressive generation), biLSTMs capture context from both directions and significantly improve accuracy.</li>
          <li><strong>Gradient clipping is essential</strong>: Even with LSTMs, use <code>torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)</code> to prevent exploding gradients.</li>
          <li><strong>Pack padded sequences</strong>: Use <code>nn.utils.rnn.pack_padded_sequence</code> and <code>pad_packed_sequence</code> to avoid computing on padding tokens. This speeds up training and avoids polluting hidden states.</li>
          <li><strong>Layer normalization helps</strong>: Apply LayerNorm inside LSTM cells for more stable training, especially with deeper stacked LSTMs.</li>
          <li><strong>GRU is often good enough</strong>: GRU has fewer parameters (3 gates vs 4), trains faster, and often matches LSTM performance. Try it first for new tasks.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Using the wrong hidden state for classification</strong>: For many-to-one tasks, use the <em>last</em> hidden state <InlineMath math="h_T" />, not the cell state <InlineMath math="c_T" />. The hidden state is the &quot;output-ready&quot; representation.</li>
          <li><strong>Forgetting that LSTM hidden state shape depends on num_layers and bidirectionality</strong>: PyTorch returns <code>h_n</code> with shape <code>(num_layers * num_directions, batch, hidden)</code>. For a 2-layer biLSTM, that&apos;s <code>(4, batch, hidden)</code>.</li>
          <li><strong>Initializing forget gate bias too low</strong>: The forget gate should default to &quot;remember everything.&quot; Set the forget gate bias to 1.0 or higher at initialization. PyTorch does this automatically, but custom implementations often miss it.</li>
          <li><strong>Not detaching hidden states between batches in language modeling</strong>: For truncated BPTT, you must <code>.detach()</code> the hidden state when passing it between batches, or the computational graph grows indefinitely.</li>
          <li><strong>Thinking RNNs can handle truly long sequences</strong>: Even LSTMs struggle beyond ~500-1000 tokens in practice. For long documents, use Transformers or hierarchical approaches.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Walk through the LSTM cell state update equation. Why does it solve the vanishing gradient problem? What role does each gate play?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>The cell state update</strong>: <InlineMath math="c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t" />
            <ul>
              <li>This is <em>additive</em>, not multiplicative through a weight matrix.</li>
              <li>Compare with vanilla RNN: <InlineMath math="h_t = \tanh(W_{hh} h_{t-1} + ...)" /> where the hidden state is repeatedly multiplied by <InlineMath math="W_{hh}" />.</li>
            </ul>
          </li>
          <li>
            <strong>Gradient flow through cell state</strong>:
            <ul>
              <li><InlineMath math="\frac{\partial c_t}{\partial c_{t-1}} = f_t" />, which is a diagonal matrix of values in <InlineMath math="[0, 1]" />.</li>
              <li>When <InlineMath math="f_t \approx 1" />, gradients flow unchanged across time. The network <em>learns</em> when to remember and when to forget.</li>
              <li>This is analogous to ResNet&apos;s skip connection: an additive shortcut for gradient flow.</li>
            </ul>
          </li>
          <li>
            <strong>Gate roles</strong>:
            <ul>
              <li><strong>Forget gate</strong> (<InlineMath math="f_t" />): Element-wise scaling of old cell state. Values near 0 erase; values near 1 preserve.</li>
              <li><strong>Input gate</strong> (<InlineMath math="i_t" />): Controls how much of the new candidate <InlineMath math="\tilde{c}_t" /> is written. Prevents irrelevant inputs from corrupting memory.</li>
              <li><strong>Output gate</strong> (<InlineMath math="o_t" />): Determines which parts of the cell state are exposed as the hidden state. The cell can store information internally without outputting it.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Hochreiter &amp; Schmidhuber (1997) &quot;Long Short-Term Memory&quot;</strong> -- The original LSTM paper. Dense but foundational.</li>
          <li><strong>Colah&apos;s Blog &quot;Understanding LSTM Networks&quot;</strong> -- The single best visual explanation of LSTM gates. Required reading.</li>
          <li><strong>Cho et al. (2014) &quot;Learning Phrase Representations using RNN Encoder-Decoder&quot;</strong> -- Introduces the GRU architecture.</li>
          <li><strong>Pascanu et al. (2013) &quot;On the difficulty of training recurrent neural networks&quot;</strong> -- Rigorous analysis of vanishing/exploding gradients in RNNs.</li>
          <li><strong>Karpathy (2015) &quot;The Unreasonable Effectiveness of Recurrent Neural Networks&quot;</strong> -- Classic blog post showing what character-level RNNs can learn.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
