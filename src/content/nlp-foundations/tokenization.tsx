"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Tokenization() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Before any language model can process text, it needs to break the text into smaller pieces called <strong>tokens</strong>.
          Tokenization is the very first step in every NLP pipeline — it converts raw strings into sequences of integers that a model
          can work with. The choice of tokenization strategy profoundly affects vocabulary size, model performance, and how well
          the model handles rare or unseen words.
        </p>
        <p>
          The simplest approach is <strong>word-level tokenization</strong>: split on whitespace and punctuation. But this creates
          enormous vocabularies (English alone has hundreds of thousands of words), and any word not in the vocabulary becomes an
          unknown &lt;UNK&gt; token. Misspellings, compound words, and morphological variants all break.
        </p>
        <p>
          The modern solution is <strong>subword tokenization</strong>. Instead of treating each word as atomic, we break words into
          meaningful sub-units. The word &quot;unhappiness&quot; might become [&quot;un&quot;, &quot;happi&quot;, &quot;ness&quot;].
          This gives us a compact vocabulary (typically 30K-50K tokens) that can still represent any possible text — including words
          never seen during training — by composing subword pieces. The three dominant algorithms are <strong>BPE</strong> (used by GPT),
          <strong>WordPiece</strong> (used by BERT), and <strong>SentencePiece</strong> (used by T5 and LLaMA).
        </p>
        <p>
          At the character level, the vocabulary is tiny (a few hundred characters), but sequences become very long and the model
          must learn to compose characters into meaning. Subword methods hit the sweet spot between character-level flexibility
          and word-level semantic density.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Byte Pair Encoding (BPE) Algorithm</h3>
        <p>
          BPE starts with a vocabulary of individual characters and iteratively merges the most frequent adjacent pair.
          Given a corpus with token frequencies, at each step:
        </p>
        <BlockMath math="\text{pair}^* = \arg\max_{(a, b) \in V \times V} \text{count}(a, b)" />
        <p>
          We merge <InlineMath math="a" /> and <InlineMath math="b" /> into a new symbol <InlineMath math="ab" /> and add it to the
          vocabulary <InlineMath math="V" />. This repeats for a fixed number of merge operations (e.g., 50,000 merges).
        </p>

        <h3>WordPiece</h3>
        <p>
          WordPiece (used in BERT) is similar but chooses merges based on <strong>likelihood</strong> rather than raw frequency.
          It selects the pair that maximizes the language model likelihood of the training corpus:
        </p>
        <BlockMath math="\text{pair}^* = \arg\max_{(a, b)} \frac{P(ab)}{P(a) \cdot P(b)}" />
        <p>
          This is equivalent to choosing the pair with the highest pointwise mutual information (PMI). Subwords that co-occur
          more than chance predicts get merged first.
        </p>

        <h3>Unigram Language Model (SentencePiece)</h3>
        <p>
          The unigram model takes a different approach: start with a <strong>large</strong> vocabulary and prune it down. Given a
          sentence <InlineMath math="X" />, the best tokenization maximizes:
        </p>
        <BlockMath math="X^* = \arg\max_{x \in S(X)} \sum_{i=1}^{|x|} \log P(x_i)" />
        <p>
          where <InlineMath math="S(X)" /> is the set of all possible segmentations and <InlineMath math="P(x_i)" /> are unigram
          probabilities estimated via EM. Tokens that contribute least to the overall likelihood are pruned until the desired
          vocabulary size is reached.
        </p>

        <h3>Vocabulary Size Tradeoff</h3>
        <p>
          Let <InlineMath math="|V|" /> be vocabulary size and <InlineMath math="L" /> the average sequence length:
        </p>
        <ul>
          <li>Larger <InlineMath math="|V|" /> means shorter sequences (<InlineMath math="L \downarrow" />) but a bigger embedding matrix <InlineMath math="\in \mathbb{R}^{|V| \times d}" /></li>
          <li>Smaller <InlineMath math="|V|" /> means longer sequences (<InlineMath math="L \uparrow" />) and more compute for self-attention <InlineMath math="O(L^2 d)" /></li>
        </ul>
      </TopicSection>

      <TopicSection type="code">
        <h3>BPE from Scratch</h3>
        <CodeBlock
          language="python"
          title="bpe_from_scratch.py"
          code={`from collections import Counter, defaultdict

def get_pair_counts(vocab):
    """Count frequency of adjacent symbol pairs across vocabulary."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_pair(pair, vocab):
    """Merge all occurrences of a symbol pair in the vocabulary."""
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

def learn_bpe(corpus, num_merges=10):
    """Learn BPE merge rules from a corpus."""
    # Initialize: split each word into characters + end-of-word marker
    word_freqs = Counter(corpus.split())
    vocab = {}
    for word, freq in word_freqs.items():
        symbols = " ".join(list(word)) + " </w>"
        vocab[symbols] = freq

    merges = []
    for i in range(num_merges):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_pair(best_pair, vocab)
        merges.append(best_pair)
        print(f"Merge {i+1}: {best_pair} -> {''.join(best_pair)}")

    return merges, vocab

# Example
corpus = "low low low low low lower lower newest newest newest widest widest"
merges, final_vocab = learn_bpe(corpus, num_merges=10)
print("\\nFinal vocabulary:")
for token, freq in sorted(final_vocab.items(), key=lambda x: -x[1]):
    print(f"  {token}: {freq}")`}
        />

        <h3>Using HuggingFace Tokenizers</h3>
        <CodeBlock
          language="python"
          title="hf_tokenizers.py"
          code={`from transformers import AutoTokenizer

# GPT-2 uses BPE
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
text = "Tokenization is surprisingly important!"
tokens = gpt2_tok.tokenize(text)
ids = gpt2_tok.encode(text)
print(f"GPT-2 BPE tokens: {tokens}")
print(f"GPT-2 token IDs:  {ids}")
# GPT-2 BPE tokens: ['Token', 'ization', ' is', ' surprisingly', ' important', '!']

# BERT uses WordPiece
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = bert_tok.tokenize(text)
ids = bert_tok.encode(text)
print(f"\\nBERT WordPiece tokens: {tokens}")
print(f"BERT token IDs:        {ids}")
# BERT WordPiece tokens: ['token', '##ization', 'is', 'surprisingly', 'important', '!']
# Note: ## prefix indicates a continuation subword

# T5 / LLaMA use SentencePiece
t5_tok = AutoTokenizer.from_pretrained("t5-small")
tokens = t5_tok.tokenize(text)
ids = t5_tok.encode(text)
print(f"\\nT5 SentencePiece tokens: {tokens}")
print(f"T5 token IDs:            {ids}")

# Compare vocabulary sizes
print(f"\\nVocab sizes: GPT-2={gpt2_tok.vocab_size}, "
      f"BERT={bert_tok.vocab_size}, T5={t5_tok.vocab_size}")

# Decode back to text
print(f"\\nDecoded: {gpt2_tok.decode(ids)}")`}
        />

        <h3>Training a Custom BPE Tokenizer</h3>
        <CodeBlock
          language="python"
          title="train_tokenizer.py"
          code={`from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Build a BPE tokenizer from scratch
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    show_progress=True,
)

# Train on your own files
tokenizer.train(files=["my_corpus.txt"], trainer=trainer)

# Use the trained tokenizer
output = tokenizer.encode("Hello, custom tokenizer!")
print(f"Tokens: {output.tokens}")
print(f"IDs:    {output.ids}")

# Save and reload
tokenizer.save("my_tokenizer.json")
loaded = Tokenizer.from_file("my_tokenizer.json")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Vocabulary size sweet spot</strong>: Most modern LLMs use 32K-100K tokens. GPT-2 uses 50,257, LLaMA uses 32,000, GPT-4 uses ~100K. Larger vocabs improve compression (fewer tokens per text) but increase embedding table memory.</li>
          <li><strong>Byte-level BPE</strong>: GPT-2 introduced byte-level BPE, which operates on UTF-8 bytes rather than Unicode characters. This guarantees any text can be tokenized (no UNK tokens), even binary data or rare scripts.</li>
          <li><strong>Tokenization affects cost</strong>: API pricing is per-token. The same text can cost 2-3x more with a poorly matched tokenizer. Code, non-English text, and mathematical notation are typically tokenized inefficiently.</li>
          <li><strong>Whitespace handling matters</strong>: SentencePiece treats the input as a raw stream (whitespace is just another character). BPE/WordPiece typically split on whitespace first. This affects how the model handles code, URLs, and formatted text.</li>
          <li><strong>Special tokens</strong>: Every tokenizer has special tokens ([CLS], [SEP], &lt;bos&gt;, &lt;eos&gt;, &lt;pad&gt;) that serve structural roles. Never forget to account for them when calculating sequence length budgets.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Assuming one token equals one word</strong>: A single word can be multiple tokens (&quot;indescribable&quot; might be 3 tokens), and common phrases can be single tokens. Always check with the actual tokenizer.</li>
          <li><strong>Ignoring tokenizer-model mismatch</strong>: Using a BERT tokenizer with a GPT model (or vice versa) will produce garbage. The tokenizer and model must be trained together or be explicitly compatible.</li>
          <li><strong>Forgetting the fertility problem</strong>: Non-English languages often get poor tokenization because the BPE merges were learned primarily on English text. &quot;Hello&quot; is 1 token, but its Japanese equivalent might be 5+ tokens.</li>
          <li><strong>Not handling special tokens in fine-tuning</strong>: When adding new special tokens (like &lt;tool_call&gt;), you must resize the embedding matrix with <code>model.resize_token_embeddings()</code>.</li>
          <li><strong>Truncation bugs</strong>: When your input exceeds max sequence length, naive truncation might cut in the middle of a subword or remove critical context. Use the tokenizer&apos;s built-in truncation (truncation=True) instead of slicing strings.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain the BPE algorithm step by step. Why do modern LLMs use subword tokenization instead of word-level or character-level?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>BPE Algorithm</strong>:
            <ul>
              <li>Start with a vocabulary of individual characters (or bytes).</li>
              <li>Count the frequency of every adjacent pair of tokens in the corpus.</li>
              <li>Merge the most frequent pair into a single new token and add it to the vocabulary.</li>
              <li>Repeat for a fixed number of merges (e.g., 50,000). The merge order becomes the encoding rules.</li>
              <li>To tokenize new text at inference time, apply the learned merges in the same order (greedy left-to-right).</li>
            </ul>
          </li>
          <li>
            <strong>Why subword beats word-level</strong>:
            <ul>
              <li>Word-level creates huge vocabularies (300K+ words in English alone), wasting parameters on rare words.</li>
              <li>Any out-of-vocabulary word becomes &lt;UNK&gt;, losing all information.</li>
              <li>Subword methods decompose rare words into known pieces: &quot;unhelpfulness&quot; becomes [&quot;un&quot;, &quot;help&quot;, &quot;ful&quot;, &quot;ness&quot;], preserving morphological information.</li>
            </ul>
          </li>
          <li>
            <strong>Why subword beats character-level</strong>:
            <ul>
              <li>Character sequences are very long, making self-attention expensive (<InlineMath math="O(L^2)" />).</li>
              <li>The model must learn to compose characters into meaningful units from scratch, requiring much more data and capacity.</li>
              <li>Subword tokens carry more semantic density per position, enabling longer effective context.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Sennrich et al. (2016) &quot;Neural Machine Translation of Rare Words with Subword Units&quot;</strong> — The paper that introduced BPE to NLP.</li>
          <li><strong>Kudo (2018) &quot;Subword Regularization&quot;</strong> — Introduces the unigram language model approach used in SentencePiece.</li>
          <li><strong>HuggingFace Tokenizers library documentation</strong> — Fast, production-quality implementations of BPE, WordPiece, and Unigram.</li>
          <li><strong>Karpathy &quot;Let&apos;s build the GPT Tokenizer&quot;</strong> — YouTube video building a BPE tokenizer from scratch, explaining every design decision.</li>
          <li><strong>The tiktoken library</strong> — OpenAI&apos;s fast BPE tokenizer used by GPT-4. Great for understanding token counts and costs.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
