"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function ControlFlow() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Control flow is how you tell Python <strong>which code to run and when</strong>. The three
          fundamental building blocks are <strong>conditionals</strong> (<code>if</code>/<code>elif</code>/<code>else</code>),
          <strong>loops</strong> (<code>for</code> and <code>while</code>), and <strong>comprehensions</strong> — Python&apos;s
          compact syntax for building new collections by transforming existing ones. In data science, you
          rarely write raw <code>for</code> loops over data (that&apos;s what NumPy and pandas are for), but
          you constantly use control flow to orchestrate pipelines, handle edge cases, filter data, and
          manage experiment configurations.
        </p>
        <p>
          Python also has <strong>generators</strong> — functions that <code>yield</code> values one at a time
          instead of returning an entire collection. Generators are <strong>lazy</strong>: they produce values
          only when asked, which means they can process datasets that don&apos;t fit in memory. When you call
          <code>range(10_000_000)</code>, Python doesn&apos;t create a list of ten million integers — it
          creates a generator-like object that produces them one by one. This lazy evaluation pattern is
          fundamental to working with large-scale data.
        </p>
        <p>
          <strong>Comprehensions</strong> are arguably Python&apos;s most distinctive feature. A list
          comprehension like <code>[x**2 for x in range(10) if x % 2 == 0]</code> is not just shorter than
          an equivalent loop — it&apos;s often <strong>faster</strong> because the iteration happens at the C
          level inside CPython. Dict comprehensions, set comprehensions, and generator expressions follow the
          same pattern and are idiomatic Python that every data scientist should read and write fluently.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Loop Complexity Analysis</h3>
        <p>
          Understanding time complexity helps you decide when a Python loop is acceptable versus when you
          need vectorized operations. A single <code>for</code> loop over <InlineMath math="n" /> items
          is <InlineMath math="O(n)" />:
        </p>
        <BlockMath math="T(n) = c \cdot n" />
        <p>
          Nested loops multiply: two nested loops over <InlineMath math="n" /> items
          give <InlineMath math="O(n^2)" />, which becomes prohibitive for large datasets. For a dataset
          of <InlineMath math="n = 10^6" /> rows:
        </p>
        <BlockMath math="O(n) \approx 10^6 \text{ ops} \quad \text{vs} \quad O(n^2) \approx 10^{12} \text{ ops}" />
        <p>
          This is why vectorized NumPy operations (implemented in C) outperform Python loops by 10-100x for
          the same algorithmic complexity. A Python <code>for</code> loop has high constant overhead per
          iteration (type checking, reference counting, bytecode dispatch), while NumPy operates on
          contiguous memory blocks with CPU-optimized instructions.
        </p>

        <h3>Generator Memory Complexity</h3>
        <p>
          A list stores all <InlineMath math="n" /> elements simultaneously, requiring <InlineMath math="O(n)" /> memory.
          A generator stores only its current state (local variables and instruction pointer),
          requiring <InlineMath math="O(1)" /> memory regardless of how many values it produces:
        </p>
        <BlockMath math="\text{List: } O(n) \text{ space} \quad \text{Generator: } O(1) \text{ space}" />
        <p>
          This makes generators essential for streaming data processing, reading large files line by line,
          and building data pipelines that chain transformations without materializing intermediate results.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Conditionals and Pattern Matching</h3>
        <CodeBlock
          language="python"
          title="conditionals.py"
          code={`# Standard if/elif/else
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

# Ternary (conditional expression) — use for simple cases
grade = "pass" if score >= 60 else "fail"

# Structural pattern matching (Python 3.10+)
# Much cleaner than chains of isinstance() checks
def process_config(config):
    match config:
        case {"model": "xgboost", "n_estimators": n}:
            print(f"XGBoost with {n} trees")
        case {"model": "linear", "regularization": reg}:
            print(f"Linear model with {reg} regularization")
        case {"model": str(name), **rest}:
            print(f"Unknown model: {name}, params: {rest}")
        case _:
            raise ValueError("Invalid config")

process_config({"model": "xgboost", "n_estimators": 100})
# Output: XGBoost with 100 trees

# Guard clauses (match + if)
match score:
    case x if x >= 90:
        print("Excellent")
    case x if x >= 70:
        print("Good")
    case _:
        print("Needs improvement")`}
        />

        <h3>Loops and Iteration Patterns</h3>
        <CodeBlock
          language="python"
          title="loops.py"
          code={`# for loop with enumerate (get index + value)
features = ["age", "income", "education"]
for i, feat in enumerate(features):
    print(f"Feature {i}: {feat}")

# zip — iterate over multiple sequences in parallel
names = ["Alice", "Bob", "Charlie"]
scores = [92, 87, 95]
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# zip with unequal lengths — stops at shortest
# Use itertools.zip_longest to pad with fillvalue
from itertools import zip_longest
for a, b in zip_longest([1, 2, 3], [10, 20], fillvalue=0):
    print(a, b)  # 1 10, 2 20, 3 0

# dict iteration patterns
params = {"lr": 0.01, "epochs": 100, "batch_size": 32}
for key, value in params.items():
    print(f"{key} = {value}")

# while loop with break — useful for convergence checks
import random
tolerance = 0.001
loss = 1.0
epoch = 0
while loss > tolerance:
    loss *= random.uniform(0.5, 0.99)  # simulate training
    epoch += 1
    if epoch > 10000:
        print("Did not converge")
        break
else:
    # The else clause runs ONLY if the loop completed without break
    print(f"Converged in {epoch} epochs, final loss: {loss:.6f}")`}
        />

        <h3>Comprehensions</h3>
        <CodeBlock
          language="python"
          title="comprehensions.py"
          code={`# List comprehension — the workhorse
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With filtering
even_squares = [x**2 for x in range(10) if x % 2 == 0]
# [0, 4, 16, 36, 64]

# Nested comprehension (flatten a matrix)
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [val for row in matrix for val in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Dict comprehension
word_lengths = {word: len(word) for word in ["python", "data", "science"]}
# {'python': 6, 'data': 4, 'science': 7}

# Invert a dictionary
inverted = {v: k for k, v in word_lengths.items()}

# Set comprehension
unique_lengths = {len(word) for word in ["python", "data", "science", "code"]}
# {4, 6, 7}

# Nested dict comprehension — feature engineering pattern
raw_data = {"age": [25, 30, 35], "income": [50000, 60000, 70000]}
normalized = {
    col: [(v - min(vals)) / (max(vals) - min(vals)) for v in vals]
    for col, vals in raw_data.items()
}
# {'age': [0.0, 0.5, 1.0], 'income': [0.0, 0.5, 1.0]}`}
        />

        <h3>Generators and Lazy Evaluation</h3>
        <CodeBlock
          language="python"
          title="generators.py"
          code={`# Generator expression — like list comp but with parentheses
# Does NOT create the list in memory
sum_squares = sum(x**2 for x in range(1_000_000))
# Computes lazily — only one value in memory at a time

# Generator function with yield
def fibonacci():
    """Infinite Fibonacci sequence — O(1) memory."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Take first 10 values
from itertools import islice
fib_10 = list(islice(fibonacci(), 10))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Real DS use case: streaming CSV processing
def read_large_csv(filepath, chunk_size=10000):
    """Process a large CSV without loading it all into memory."""
    import csv
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk  # Don't forget the last partial chunk

# Pipeline pattern: chain generators
def parse_rows(chunks):
    for chunk in chunks:
        for row in chunk:
            yield {k: float(v) if v.replace(".", "").isdigit() else v
                   for k, v in row.items()}

def filter_valid(rows):
    for row in rows:
        if row.get("price", 0) > 0:
            yield row

# Nothing executes until you consume the pipeline:
# chunks = read_large_csv("huge_file.csv")
# parsed = parse_rows(chunks)
# valid = filter_valid(parsed)
# for row in valid:  # NOW it starts reading the file
#     process(row)

# yield from — delegate to a sub-generator
def flatten(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)  # recursively delegate
        else:
            yield item

print(list(flatten([1, [2, [3, 4], 5], 6])))
# [1, 2, 3, 4, 5, 6]`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Avoid Python loops on data — vectorize instead</strong>: If you&apos;re iterating row-by-row over
            a DataFrame, you&apos;re almost certainly doing it wrong. Use <code>df.apply()</code> as a last resort,
            and prefer <code>df[&quot;col&quot;].map()</code> or fully vectorized pandas/NumPy operations.
          </li>
          <li>
            <strong>Use comprehensions for data transformation</strong>: Building feature name lists, cleaning
            column names, filtering configs — comprehensions are the Pythonic way. But if a comprehension exceeds
            one line, switch to a regular loop for readability.
          </li>
          <li>
            <strong>Generators for data loading</strong>: PyTorch <code>DataLoader</code>, TensorFlow
            <code>tf.data.Dataset</code>, and pandas <code>read_csv(chunksize=N)</code> all use lazy iteration
            under the hood. Understanding generators helps you build custom data pipelines.
          </li>
          <li>
            <strong>Pattern matching for config handling</strong>: ML experiments often have complex config
            dicts. <code>match/case</code> (Python 3.10+) is much cleaner than nested <code>if/elif</code>
            chains for dispatching on config structure.
          </li>
          <li>
            <strong>Use <code>for/else</code> for search patterns</strong>: The <code>else</code> clause
            on a <code>for</code> loop runs if the loop completed without hitting <code>break</code>. It&apos;s
            perfect for &quot;find or fail&quot; patterns.
          </li>
          <li>
            <strong>itertools is your friend</strong>: <code>chain</code>, <code>product</code>,
            <code>combinations</code>, <code>groupby</code>, <code>accumulate</code> — learn these for hyperparameter
            grid search, feature combinations, and grouped aggregations.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Modifying a list while iterating over it</strong>: <code>for x in my_list: my_list.remove(x)</code> skips
            elements because the iterator&apos;s index advances while the list shrinks. Iterate over a copy instead:
            <code>for x in my_list[:]: ...</code>
          </li>
          <li>
            <strong>Exhausted generators</strong>: A generator can only be iterated <strong>once</strong>.
            After the first <code>for</code> loop consumes it, a second loop gets nothing. If you need multiple
            passes, convert to a list first or re-create the generator.
          </li>
          <li>
            <strong>Overusing comprehensions</strong>: A comprehension with nested loops, multiple conditions,
            and a complex expression is <em>worse</em> than a regular loop. If you can&apos;t read the
            comprehension in 5 seconds, refactor it.
          </li>
          <li>
            <strong>Using <code>range(len(...))</code></strong>: Instead of <code>for i in range(len(items))</code>,
            use <code>for i, item in enumerate(items)</code>. It&apos;s more Pythonic, less error-prone, and just as fast.
          </li>
          <li>
            <strong>Forgetting that <code>dict.keys()</code> returns a view, not a list</strong>: In Python 3,
            <code>dict.keys()</code> is a dynamic view. If you modify the dict during iteration, you get a
            <code>RuntimeError</code>. Convert to a list first: <code>for k in list(d.keys()): ...</code>
          </li>
          <li>
            <strong>Accidentally creating a tuple instead of a generator</strong>: There is no &quot;tuple
            comprehension.&quot; <code>(x**2 for x in range(10))</code> creates a <strong>generator expression</strong>,
            not a tuple. Use <code>tuple(x**2 for x in range(10))</code> for a tuple.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Write a function <code>flatten_dict</code> that takes an arbitrarily nested
          dictionary and returns a flat dictionary with dot-separated keys. For example:
        </p>
        <CodeBlock
          language="python"
          code={`input_dict = {
    "model": {
        "name": "xgboost",
        "params": {
            "n_estimators": 100,
            "max_depth": 6
        }
    },
    "data": {
        "path": "/data/train.csv"
    }
}

# Expected output:
# {
#     "model.name": "xgboost",
#     "model.params.n_estimators": 100,
#     "model.params.max_depth": 6,
#     "data.path": "/data/train.csv"
# }`}
        />
        <p><strong>Solution:</strong></p>
        <CodeBlock
          language="python"
          title="flatten_dict.py"
          code={`def flatten_dict(d, parent_key="", sep="."):
    """Flatten a nested dict with dot-separated keys.

    Uses a generator to yield key-value pairs lazily,
    then collects them into a dict.
    """
    items = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            # Recurse into nested dicts
            items.extend(flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

# Test it
config = {
    "model": {
        "name": "xgboost",
        "params": {"n_estimators": 100, "max_depth": 6}
    },
    "data": {"path": "/data/train.csv"}
}

flat = flatten_dict(config)
print(flat)
# {'model.name': 'xgboost', 'model.params.n_estimators': 100,
#  'model.params.max_depth': 6, 'data.path': '/data/train.csv'}

# Alternative: iterative version using a stack (avoids recursion limit)
def flatten_dict_iterative(d, sep="."):
    result = {}
    stack = [("", d)]
    while stack:
        prefix, current = stack.pop()
        for key, value in current.items():
            new_key = f"{prefix}{sep}{key}" if prefix else key
            if isinstance(value, dict):
                stack.append((new_key, value))
            else:
                result[new_key] = value
    return result`}
        />
        <p>
          This question tests recursion, dict iteration, string manipulation, and the ability to handle
          edge cases. A follow-up might ask: &quot;How would you handle lists inside the dict?&quot; or
          &quot;What&apos;s the time complexity?&quot; (Answer: <InlineMath math="O(n)" /> where <InlineMath math="n" /> is the
          total number of leaf key-value pairs, since each pair is visited exactly once.)
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li>
            <strong>Python docs: Compound Statements</strong> — <code>docs.python.org/3/reference/compound_stmts.html</code> — the
            official spec for <code>if</code>, <code>for</code>, <code>while</code>, <code>match</code>
          </li>
          <li>
            <strong>PEP 572 — Assignment Expressions</strong> — The walrus operator (<code>:=</code>)
            rationale and use cases
          </li>
          <li>
            <strong>PEP 634/635/636 — Structural Pattern Matching</strong> — The full specification for
            <code>match/case</code> with tutorials and examples
          </li>
          <li>
            <strong>David Beazley &quot;Generator Tricks for Systems Programmers&quot;</strong> — The classic
            talk/tutorial on advanced generator patterns and pipeline composition
          </li>
          <li>
            <strong>itertools documentation</strong> — <code>docs.python.org/3/library/itertools.html</code> — recipes
            section has production-ready patterns for grouping, windowing, and combining iterators
          </li>
        </ul>
      </TopicSection>
    </div>
  );
}
