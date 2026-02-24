"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function PromptEngineering() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Prompt engineering is <strong>programming in natural language</strong>. Instead of writing code that
          tells a computer exactly what to do step by step, you write instructions in plain English (or any language)
          that guide a large language model to produce the output you want. The wording matters enormously — a small
          change in phrasing can be the difference between a useless response and a brilliant one.
        </p>
        <p>
          There are three foundational prompting strategies. <strong>Zero-shot</strong> prompting gives the model
          a task with no examples — you just describe what you want. <strong>Few-shot</strong> prompting includes
          a handful of input-output examples in the prompt so the model learns the pattern by analogy.
          <strong> Chain-of-thought (CoT)</strong> prompting asks the model to &quot;think step by step,&quot;
          which dramatically improves performance on reasoning tasks like math, logic, and multi-step problems.
        </p>
        <p>
          Why does wording matter so much? LLMs are autoregressive models — they predict one token at a time based on
          everything that came before. The prompt is the starting context that shapes the entire probability distribution
          over possible completions. A well-crafted prompt steers the model toward the region of its knowledge that
          is most relevant to your task. A poorly crafted prompt leaves the model guessing about your intent, leading
          to generic or off-target responses.
        </p>
        <p>
          Prompt engineering has become a critical skill because it is the primary interface for working with LLMs.
          Before fine-tuning, before building complex pipelines, the first question is always: can you get the model
          to do what you want with the right prompt?
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Autoregressive Token Probability</h3>
        <p>
          An autoregressive language model generates text by predicting one token at a time. Given a
          prompt <InlineMath math="x_1, x_2, \ldots, x_n" />, the probability of the next
          token <InlineMath math="x_{n+1}" /> is:
        </p>
        <BlockMath math="P(x_{n+1} \mid x_1, x_2, \ldots, x_n) = \text{softmax}(z_{n+1})_{x_{n+1}}" />
        <p>
          where <InlineMath math="z_{n+1} \in \mathbb{R}^{|V|}" /> is the logit vector over the
          vocabulary <InlineMath math="V" />. The full sequence probability decomposes as:
        </p>
        <BlockMath math="P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})" />
        <p>
          This is why the prompt matters: every token in the prompt shifts the conditional distribution
          for all subsequent tokens.
        </p>

        <h3>Temperature Scaling</h3>
        <p>
          Temperature <InlineMath math="\tau" /> controls the sharpness of the output distribution by
          scaling the logits before applying softmax:
        </p>
        <BlockMath math="P(x_t = w) = \frac{\exp(z_w / \tau)}{\sum_{w' \in V} \exp(z_{w'} / \tau)}" />
        <p>
          When <InlineMath math="\tau \to 0" />, the distribution collapses to a point mass on the
          highest-probability token (deterministic, greedy decoding). When <InlineMath math="\tau = 1" />,
          you get the model&apos;s native distribution. When <InlineMath math="\tau > 1" />, the distribution
          becomes more uniform (more random, more &quot;creative&quot;).
        </p>

        <h3>Top-k Sampling</h3>
        <p>
          Restrict sampling to the <InlineMath math="k" /> tokens with highest probability:
        </p>
        <BlockMath math="P'(x_t = w) = \begin{cases} \frac{P(x_t = w)}{\sum_{w' \in V_k} P(x_t = w')} & \text{if } w \in V_k \\ 0 & \text{otherwise} \end{cases}" />
        <p>
          where <InlineMath math="V_k" /> is the set of <InlineMath math="k" /> highest-probability tokens.
        </p>

        <h3>Top-p (Nucleus) Sampling</h3>
        <p>
          Instead of a fixed <InlineMath math="k" />, select the smallest set of tokens whose cumulative
          probability exceeds threshold <InlineMath math="p" />:
        </p>
        <BlockMath math="V_p = \arg\min_{V' \subseteq V} |V'| \quad \text{subject to} \quad \sum_{w \in V'} P(x_t = w) \geq p" />
        <p>
          This adapts dynamically: when the model is confident, few tokens are sampled; when uncertain,
          many tokens are considered. Typical values: <InlineMath math="p = 0.9" /> or <InlineMath math="p = 0.95" />.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Zero-Shot vs Few-Shot Classification</h3>
        <CodeBlock
          language="python"
          title="zero_few_shot.py"
          code={`from transformers import pipeline

# Load a text-generation model
generator = pipeline("text-generation", model="gpt2-large", max_new_tokens=20)

# ---- ZERO-SHOT: No examples, just the task description ----
zero_shot_prompt = """Classify the following customer review as Positive, Negative, or Neutral.

Review: "The product arrived on time and works exactly as described. Very happy!"
Classification:"""

result = generator(zero_shot_prompt, do_sample=False)
print("Zero-shot:", result[0]["generated_text"].split("Classification:")[-1].strip())

# ---- FEW-SHOT: Provide examples so the model learns the pattern ----
few_shot_prompt = """Classify the following customer reviews as Positive, Negative, or Neutral.

Review: "Absolutely love this! Best purchase I've made all year."
Classification: Positive

Review: "Arrived broken and customer service was unhelpful."
Classification: Negative

Review: "It's okay. Does what it says, nothing special."
Classification: Neutral

Review: "The product arrived on time and works exactly as described. Very happy!"
Classification:"""

result = generator(few_shot_prompt, do_sample=False)
print("Few-shot:", result[0]["generated_text"].split("Classification:")[-1].strip())

# ---- PROGRAMMATIC FEW-SHOT BUILDER ----
def build_few_shot_prompt(task_description, examples, query):
    """
    Build a few-shot prompt from structured examples.

    Args:
        task_description: what the model should do
        examples: list of (input_text, output_text) tuples
        query: the new input to classify
    """
    prompt = task_description + "\\n\\n"
    for inp, out in examples:
        prompt += f"Input: {inp}\\nOutput: {out}\\n\\n"
    prompt += f"Input: {query}\\nOutput:"
    return prompt

examples = [
    ("I need to reset my password", "Account Access"),
    ("When will my order arrive?", "Shipping"),
    ("I want a refund for my purchase", "Billing"),
    ("The app keeps crashing on startup", "Technical Issue"),
]

prompt = build_few_shot_prompt(
    "Classify the customer support ticket into a category.",
    examples,
    "I was charged twice for the same item"
)
print(prompt)`}
        />

        <h3>Chain-of-Thought Prompting for Math Problems</h3>
        <CodeBlock
          language="python"
          title="chain_of_thought.py"
          code={`import re

def chain_of_thought_prompt(question):
    """
    Build a chain-of-thought prompt that forces step-by-step reasoning.
    """
    return f"""Solve the following math problem step by step.
Show your reasoning at each step before giving the final answer.

Question: {question}

Let me solve this step by step:"""


def extract_answer(response, pattern=r"(?:final answer|answer is)[:\\s]*([\\d,.]+)"):
    """Extract the final numerical answer from a CoT response."""
    match = re.search(pattern, response.lower())
    if match:
        return match.group(1).replace(",", "")
    # Fallback: find the last number in the response
    numbers = re.findall(r"[\\d,]+\\.?\\d*", response)
    return numbers[-1].replace(",", "") if numbers else None


# Compare standard vs chain-of-thought prompting
standard_prompt = """Answer the following question with just the number.

Question: A store has 45 apples. They sell 12 in the morning and receive
a shipment of 30 more. Then they sell 18 in the afternoon. How many
apples do they have at the end of the day?

Answer:"""

cot_prompt = chain_of_thought_prompt(
    "A store has 45 apples. They sell 12 in the morning and receive "
    "a shipment of 30 more. Then they sell 18 in the afternoon. How many "
    "apples do they have at the end of the day?"
)

# The CoT version will produce reasoning like:
# Step 1: Start with 45 apples
# Step 2: Sell 12 in the morning: 45 - 12 = 33
# Step 3: Receive 30 more: 33 + 30 = 63
# Step 4: Sell 18 in the afternoon: 63 - 18 = 45
# Final answer: 45

# ---- ZERO-SHOT CoT (just add "Let's think step by step") ----
zero_shot_cot = f"""Question: If a train travels at 60 mph for 2.5 hours, then at
90 mph for 1.5 hours, what is the total distance traveled?

Let's think step by step."""

print("Standard prompt:")
print(standard_prompt)
print("\\nChain-of-thought prompt:")
print(cot_prompt)`}
        />

        <h3>Structured Output Extraction with System Prompts</h3>
        <CodeBlock
          language="python"
          title="structured_output.py"
          code={`import json
import re

def build_json_extraction_prompt(text, schema_description, example_output):
    """
    Build a prompt that extracts structured data as JSON.

    Args:
        text: the input text to extract from
        schema_description: describes the expected fields
        example_output: a dict showing the expected JSON structure
    """
    system = """You are a data extraction assistant. Extract information from
the provided text and return it as valid JSON. Only include information
explicitly stated in the text. Use null for missing fields."""

    user = f"""Extract the following fields from the text below:
{schema_description}

Example output format:
{json.dumps(example_output, indent=2)}

Text: \\"{text}\\"

JSON output:"""

    return {"system": system, "user": user}


# Example: Extract structured data from job postings
schema = """- company: the company name
- title: the job title
- location: where the job is based
- salary_min: minimum salary (integer, or null)
- salary_max: maximum salary (integer, or null)
- remote: whether remote work is available (boolean)"""

example = {
    "company": "Acme Corp",
    "title": "Software Engineer",
    "location": "San Francisco, CA",
    "salary_min": 120000,
    "salary_max": 180000,
    "remote": True,
}

text = "Senior ML Engineer at DataFlow Inc. Based in NYC, with hybrid remote \
option. Compensation range $150,000 - $220,000 plus equity."

prompt = build_json_extraction_prompt(text, schema, example)
print("System:", prompt["system"])
print("\\nUser:", prompt["user"])


def validate_json_output(response_text):
    """Parse and validate JSON from model response."""
    # Try to find JSON in the response
    json_match = re.search(r'\\{[^{}]*\\}', response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return parsed
        except json.JSONDecodeError:
            return None
    return None


# ---- PROMPT CHAINING: Multi-step pipeline ----
def classification_then_extraction(text):
    """
    Step 1: Classify the document type
    Step 2: Use type-specific extraction prompt
    """
    # Step 1: Classify
    classify_prompt = f"""Classify this text into exactly one category:
[job_posting, product_review, news_article, email]

Text: \\"{text}\\"
Category:"""

    # In production, send classify_prompt to the model and get category
    category = "job_posting"  # simulated response

    # Step 2: Extract based on category
    extraction_schemas = {
        "job_posting": schema,  # defined above
        "product_review": "- product, rating, pros, cons, recommendation",
        "news_article": "- headline, date, source, summary, entities",
        "email": "- sender, recipient, subject, action_items, urgency",
    }

    extract_prompt = build_json_extraction_prompt(
        text, extraction_schemas[category], example
    )
    return classify_prompt, extract_prompt

step1, step2 = classification_then_extraction(text)
print("\\n--- Step 1 (Classify) ---")
print(step1)
print("\\n--- Step 2 (Extract) ---")
print(step2["user"])`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>System prompts set the persona and rules</strong>: Use the system prompt to define the model&apos;s role, constraints, and output format. This is separate from the user message and carries higher weight in most APIs. Example: &quot;You are a medical coding assistant. Only use ICD-10 codes. If uncertain, say so.&quot;</li>
          <li><strong>Temperature tuning</strong>: Use <InlineMath math="\tau = 0" /> (or near zero) for factual, deterministic tasks like classification, extraction, and code generation. Use <InlineMath math="\tau = 0.7\text{--}1.0" /> for creative tasks like brainstorming and writing. Never use high temperature for tasks requiring accuracy.</li>
          <li><strong>Output format control</strong>: Explicitly specify the format you want: &quot;Respond with only a JSON object,&quot; &quot;Answer with Yes or No,&quot; or &quot;Return a Python list.&quot; If the model produces free text when you need structured data, your prompt is too vague.</li>
          <li><strong>Prompt chaining</strong>: Break complex tasks into multiple LLM calls. First classify, then extract. First outline, then write. First draft, then critique. Each step gets a focused prompt, and the output of one step becomes input to the next.</li>
          <li><strong>Few-shot example selection</strong>: Choose examples that are diverse (cover edge cases), relevant (similar to the expected inputs), and correctly labeled. 3-5 examples usually suffice. More examples improve consistency but cost tokens.</li>
          <li><strong>Delimiters and structure</strong>: Use clear delimiters (triple backticks, XML tags, markdown headers) to separate instructions from data. This prevents prompt injection and helps the model distinguish between what it should follow and what it should process.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Too much instruction</strong>: Overloading the prompt with rules, caveats, and edge cases can confuse the model. Start simple. Add constraints only when the model fails on specific cases. A 50-word prompt often outperforms a 500-word one.</li>
          <li><strong>Not providing examples</strong>: Telling the model &quot;classify sentiment&quot; without showing what your labels look like leads to inconsistent outputs. Few-shot examples are the single most effective prompting technique for consistency.</li>
          <li><strong>Ignoring token limits</strong>: The prompt plus the expected output must fit within the model&apos;s context window. If your few-shot examples consume 90% of the context, the model has no room to generate a good response. Always calculate: prompt tokens + max output tokens &le; context window.</li>
          <li><strong>Not testing edge cases</strong>: A prompt that works for happy-path inputs may fail on ambiguous, adversarial, or empty inputs. Test with the hardest cases first: &quot;What if the input is blank?&quot; &quot;What if the input contradicts the instructions?&quot;</li>
          <li><strong>Using high temperature for deterministic tasks</strong>: If you need the same input to always produce the same output (classification, extraction, code), set temperature to 0. Non-zero temperature introduces randomness that makes outputs unreproducible.</li>
          <li><strong>Forgetting to specify what NOT to do</strong>: Models tend to be verbose. If you want concise output, say &quot;Do not include explanations.&quot; If you want JSON only, say &quot;Do not include any text outside the JSON object.&quot; Negative instructions are surprisingly effective.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Design a prompt pipeline for classifying customer support tickets into categories, assigning priority, and routing to the right team. How would you handle edge cases and evaluate quality?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li>
            <strong>Step 1 &mdash; Classification prompt</strong>:
            <ul>
              <li>Use few-shot prompting with 3-5 examples per category (Billing, Technical, Shipping, Account, General).</li>
              <li>Include ambiguous examples in the few-shot set to teach the model how to handle borderline cases.</li>
              <li>Force structured output: &quot;Respond with exactly one category from the list above.&quot;</li>
              <li>Set temperature to 0 for deterministic classification.</li>
            </ul>
          </li>
          <li>
            <strong>Step 2 &mdash; Priority assignment</strong>:
            <ul>
              <li>Chain the category result into a second prompt that assigns priority (P0-P3).</li>
              <li>Define priority criteria in the system prompt: P0 = service outage affecting many users, P1 = individual user blocked, P2 = inconvenience, P3 = general inquiry.</li>
              <li>Use chain-of-thought: &quot;First assess impact, then urgency, then assign priority.&quot;</li>
            </ul>
          </li>
          <li>
            <strong>Step 3 &mdash; Routing</strong>:
            <ul>
              <li>Map (category, priority) to team using deterministic rules (no LLM needed for this step).</li>
              <li>P0 tickets get escalated regardless of category.</li>
            </ul>
          </li>
          <li>
            <strong>Edge cases</strong>:
            <ul>
              <li>Multi-label tickets (billing AND technical): classify as the primary issue, note secondary in metadata.</li>
              <li>Tickets in other languages: add instruction &quot;The ticket may be in any language. Classify based on the content regardless of language.&quot;</li>
              <li>Adversarial inputs: add guardrails — &quot;If the input is not a customer support ticket, respond with category: Invalid.&quot;</li>
            </ul>
          </li>
          <li>
            <strong>Evaluation</strong>:
            <ul>
              <li>Build a labeled test set of 200+ tickets with ground-truth categories and priorities.</li>
              <li>Measure accuracy, precision/recall per category, and confusion matrix.</li>
              <li>Track agreement rate between LLM classification and human agents over time.</li>
              <li>Monitor for distribution drift: if a new issue type emerges, update the few-shot examples.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>OpenAI Prompt Engineering Guide</strong> &mdash; Comprehensive guide covering best practices, strategies, and tactics for effective prompting.</li>
          <li><strong>Anthropic Prompt Library</strong> &mdash; Curated collection of effective prompts for common tasks with explanations of why they work.</li>
          <li><strong>Wei et al. (2022) &quot;Chain-of-Thought Prompting Elicits Reasoning in Large Language Models&quot;</strong> &mdash; The foundational paper showing that adding &quot;Let&apos;s think step by step&quot; dramatically improves reasoning performance.</li>
          <li><strong>Brown et al. (2020) &quot;Language Models are Few-Shot Learners&quot;</strong> &mdash; The GPT-3 paper that demonstrated in-context learning and the power of few-shot prompting.</li>
          <li><strong>Kojima et al. (2022) &quot;Large Language Models are Zero-Shot Reasoners&quot;</strong> &mdash; Shows that zero-shot chain-of-thought (just appending &quot;Let&apos;s think step by step&quot;) works surprisingly well without any examples.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
