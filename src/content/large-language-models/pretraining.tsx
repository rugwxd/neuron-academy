"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Pretraining() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Pre-training is the phase where a language model learns the structure of language from massive amounts of unlabeled text.
          No human labels are needed — the supervision comes from the text itself. This <strong>self-supervised</strong> learning is
          what makes LLMs possible: we can train on trillions of tokens scraped from the internet, books, and code without annotating
          a single example.
        </p>
        <p>
          There are two dominant paradigms. <strong>Causal language modeling</strong> (used by GPT, LLaMA, Claude) predicts the next token
          given all previous tokens. The model reads left-to-right, and at each position it answers: &quot;What word comes next?&quot;
          This is also called <strong>autoregressive</strong> modeling. <strong>Masked language modeling</strong> (used by BERT) randomly
          masks some tokens and asks the model to predict the missing ones from bidirectional context — it can look both left and right.
        </p>
        <p>
          The causal LM approach has won out for generative models because it naturally supports text generation: just keep predicting the
          next token. BERT-style models are better for understanding tasks (classification, extraction) because bidirectional context gives
          richer representations, but they cannot generate text autoregressively.
        </p>
        <p>
          Pre-training is astronomically expensive. Training a frontier LLM costs tens of millions of dollars in compute, processes
          trillions of tokens, and takes weeks to months on thousands of GPUs. The resulting model is a <strong>foundation model</strong> —
          a general-purpose base that can be fine-tuned or prompted for specific tasks.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Causal Language Modeling (Autoregressive)</h3>
        <p>
          Given a sequence of tokens <InlineMath math="x_1, x_2, \ldots, x_T" />, the causal LM objective maximizes:
        </p>
        <BlockMath math="\mathcal{L}_{\text{CLM}} = \sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})" />
        <p>
          The probability is computed via a transformer with causal masking. At each position <InlineMath math="t" />, the model
          produces a distribution over the vocabulary:
        </p>
        <BlockMath math="P_\theta(x_t \mid x_{<t}) = \text{softmax}(\mathbf{h}_t W_E^\top)_{x_t}" />
        <p>
          where <InlineMath math="\mathbf{h}_t \in \mathbb{R}^d" /> is the hidden state at position <InlineMath math="t" /> and
          <InlineMath math="W_E \in \mathbb{R}^{|V| \times d}" /> is the (shared) embedding matrix.
        </p>

        <h3>Masked Language Modeling (BERT)</h3>
        <p>
          Randomly mask 15% of tokens. For each masked position <InlineMath math="t \in \mathcal{M}" />:
        </p>
        <BlockMath math="\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log P_\theta(x_t \mid x_{\backslash \mathcal{M}})" />
        <p>
          The masking strategy (from the original BERT paper): of the selected 15% tokens, 80% are replaced with [MASK], 10% with a
          random token, and 10% remain unchanged. This prevents the model from learning that [MASK] is special.
        </p>

        <h3>Next Sentence Prediction (BERT)</h3>
        <p>
          BERT was also trained to predict whether sentence B follows sentence A:
        </p>
        <BlockMath math="\mathcal{L}_{\text{NSP}} = -\left[y \log P(\text{IsNext}) + (1-y) \log P(\text{NotNext})\right]" />
        <p>
          Later work (RoBERTa) showed NSP is not necessary and can even hurt performance.
        </p>

        <h3>Perplexity</h3>
        <p>
          The standard metric for language model quality is perplexity — the exponentiated average negative log-likelihood:
        </p>
        <BlockMath math="\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})\right)" />
        <p>
          Perplexity can be interpreted as the effective number of equally likely next tokens. Lower is better.
          A perplexity of 1 means the model perfectly predicts every token. GPT-3 achieves ~20-30 perplexity on common benchmarks.
        </p>

        <h3>Cross-Entropy Loss</h3>
        <p>
          The training loss at each position is the cross-entropy between the one-hot true token and the predicted distribution:
        </p>
        <BlockMath math="H(p, q) = -\sum_{v \in V} p(v) \log q(v) = -\log q(x_t)" />
        <p>
          since <InlineMath math="p" /> is a one-hot vector. Minimizing cross-entropy is equivalent to maximizing log-likelihood.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Causal Language Modeling with HuggingFace</h3>
        <CodeBlock
          language="python"
          title="causal_lm.py"
          code={`from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# Compute loss on a sequence (this is what pre-training minimizes)
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss        # cross-entropy loss
    logits = outputs.logits    # (1, seq_len, vocab_size)

perplexity = torch.exp(loss)
print(f"Loss: {loss.item():.4f}")
print(f"Perplexity: {perplexity.item():.2f}")

# See what the model predicts at each position
import torch.nn.functional as F
probs = F.softmax(logits[0], dim=-1)
for i in range(1, inputs["input_ids"].shape[1]):
    token_id = inputs["input_ids"][0, i]
    predicted_id = probs[i-1].argmax()
    prob = probs[i-1, token_id].item()
    print(f"  After '{tokenizer.decode(inputs['input_ids'][0, :i])}' -> "
          f"predicted: '{tokenizer.decode(predicted_id)}', "
          f"actual: '{tokenizer.decode(token_id)}', "
          f"P(actual)={prob:.4f}")`}
        />

        <h3>Masked Language Modeling with BERT</h3>
        <CodeBlock
          language="python"
          title="masked_lm.py"
          code={`from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

# Mask a word and predict it
text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # (1, seq_len, vocab_size)

# Find the [MASK] position
mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

# Top 5 predictions for the masked position
mask_logits = logits[0, mask_idx[0]]
top5 = torch.topk(mask_logits, 5)
print(f"Input: {text}")
print("Top 5 predictions for [MASK]:")
for token_id, score in zip(top5.indices, top5.values):
    word = tokenizer.decode(token_id)
    print(f"  {word}: {score.item():.4f}")
# paris: 18.28, lyon: 12.15, lille: 10.50, ...`}
        />

        <h3>Pre-training a Small Causal LM from Scratch</h3>
        <CodeBlock
          language="python"
          title="pretrain_from_scratch.py"
          code={`from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import load_dataset

# Define a small model architecture
config = GPT2Config(
    vocab_size=50257,
    n_positions=512,
    n_embd=256,       # small embedding dim
    n_layer=4,        # 4 transformer layers
    n_head=4,         # 4 attention heads
)
model = GPT2LMHeadModel(config)
print(f"Parameters: {model.num_parameters():,}")  # ~5M params

# Load and tokenize data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Data collator shifts labels for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # False = causal LM (labels are shifted input_ids)
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./small-gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=1000,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

trainer.train()`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Data quality dominates</strong>: Frontier LLMs are trained on heavily curated data mixes. The ratio of code vs. text vs. math data significantly affects reasoning ability. Garbage in, garbage out — at trillion-token scale.</li>
          <li><strong>Scaling laws</strong>: Kaplan et al. (2020) and Hoffmann et al. (2022, &quot;Chinchilla&quot;) established that loss decreases as a power law with compute, data, and parameters. Chinchilla showed most models were under-trained — you need roughly 20 tokens per parameter.</li>
          <li><strong>Tokenizer determines ceiling</strong>: If the tokenizer fragments important words into many tokens, the model must use multiple positions to represent a single concept, wasting capacity. Tokenizer design is pre-training decision #1.</li>
          <li><strong>Learning rate schedule</strong>: Most LLM pre-training uses a warmup phase (linear increase to peak LR) followed by cosine decay to near-zero. The peak learning rate and warmup duration are critical hyperparameters.</li>
          <li><strong>Context length training</strong>: Models are typically pre-trained on shorter contexts (2K-4K tokens) for efficiency, then extended to longer contexts (32K-128K) in a second phase with techniques like RoPE scaling or ALiBi.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Confusing pre-training with fine-tuning</strong>: Pre-training is the initial phase on massive unlabeled data. Fine-tuning adapts the pre-trained model to a specific task with much less data. They use different learning rates, data, and objectives.</li>
          <li><strong>Thinking MLM models can generate text</strong>: BERT predicts masked tokens in parallel, not autoregressively. You cannot sample coherent text from BERT — use causal LMs (GPT-style) for generation.</li>
          <li><strong>Underestimating data deduplication</strong>: Duplicate data in pre-training causes memorization, inflated benchmarks, and wasted compute. Near-dedup (MinHash) is essential at scale.</li>
          <li><strong>Ignoring the label shift in causal LM</strong>: In causal LM, the label for position <InlineMath math="t" /> is the token at position <InlineMath math="t+1" />. The HuggingFace DataCollator handles this automatically, but implementing it yourself requires the shift: <code>labels = input_ids[1:]</code>.</li>
          <li><strong>Evaluating on training data</strong>: Language models can memorize training text. Always evaluate perplexity on held-out data, and check for contamination with your downstream benchmarks.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Compare causal language modeling (GPT-style) with masked language modeling (BERT-style). When would you choose each? Why has causal LM become dominant for large language models?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Causal LM (Autoregressive)</strong>:
            <ul>
              <li>Predicts <InlineMath math="P(x_t \mid x_1, \ldots, x_{t-1})" /> — left-to-right, one token at a time.</li>
              <li>Natural for generation: just keep sampling next tokens. This is why GPT, LLaMA, and Claude use it.</li>
              <li>Sees only past context at each position (unidirectional), which limits representation power for understanding tasks.</li>
            </ul>
          </li>
          <li>
            <strong>Masked LM (Bidirectional)</strong>:
            <ul>
              <li>Randomly masks 15% of tokens and predicts them from full bidirectional context.</li>
              <li>Richer representations for understanding tasks because each position sees both left and right context.</li>
              <li>Cannot generate text autoregressively — it predicts masked positions in parallel, not sequentially.</li>
            </ul>
          </li>
          <li>
            <strong>Why causal LM won</strong>:
            <ul>
              <li>Generative capability: the same model can answer questions, write code, summarize, translate — any text-to-text task.</li>
              <li>In-context learning: causal LMs can learn from examples in the prompt (few-shot), eliminating the need for task-specific fine-tuning.</li>
              <li>Scaling: causal LMs scale more smoothly — every token in the sequence provides a training signal (not just the 15% masked tokens).</li>
              <li>BERT-style models are still used for embedding, retrieval, and classification where bidirectional context matters and generation is not needed.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Radford et al. (2018) &quot;Improving Language Understanding by Generative Pre-Training&quot;</strong> — The original GPT paper that introduced causal LM pre-training + fine-tuning.</li>
          <li><strong>Devlin et al. (2019) &quot;BERT: Pre-training of Deep Bidirectional Transformers&quot;</strong> — Introduced masked language modeling and set off the pre-training revolution.</li>
          <li><strong>Hoffmann et al. (2022) &quot;Training Compute-Optimal Large Language Models&quot;</strong> — The Chinchilla paper on scaling laws and optimal data/parameter ratios.</li>
          <li><strong>Touvron et al. (2023) &quot;LLaMA: Open and Efficient Foundation Language Models&quot;</strong> — Detailed recipe for pre-training open-source LLMs.</li>
          <li><strong>Andrej Karpathy&apos;s &quot;Let&apos;s reproduce GPT-2 (124M)&quot;</strong> — YouTube video walking through the full pre-training process from scratch.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
