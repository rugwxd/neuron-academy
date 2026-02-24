"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function FunctionsAndClosures() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Functions in Python are <strong>first-class objects</strong>. This means a function is a value — you
          can assign it to a variable, pass it as an argument to another function, return it from a function,
          and store it in a data structure. This is not just an academic detail; it&apos;s the foundation of
          callbacks, decorators, and the functional programming patterns that permeate data science code
          (think <code>df.apply(my_func)</code>, <code>map(transform, data)</code>, or registering custom
          loss functions in PyTorch).
        </p>
        <p>
          Python supports <strong>*args</strong> (variable positional arguments) and <strong>**kwargs</strong> (variable
          keyword arguments), which let you write flexible functions that accept any number of inputs. This is
          how libraries like scikit-learn pass hyperparameters through wrapper classes without knowing every
          possible parameter in advance. <strong>Lambda</strong> expressions create small anonymous functions
          inline — useful for quick transformations but not a replacement for named functions when logic gets
          complex.
        </p>
        <p>
          <strong>Closures</strong> are functions that remember the variables from the scope where they were
          defined, even after that scope has exited. <strong>Decorators</strong> are the most practical
          application of closures: they wrap a function to add behavior (logging, timing, caching, access
          control) without modifying the function&apos;s code. Python&apos;s built-in <code>@property</code>,
          <code>@staticmethod</code>, <code>@functools.lru_cache</code>, and <code>@torch.no_grad()</code> are
          all decorators. Understanding how they work under the hood — a decorator is just a function that takes
          a function and returns a new function — unlocks a powerful pattern for writing clean, DRY code.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Function Composition</h3>
        <p>
          In mathematics, function composition <InlineMath math="(f \circ g)(x) = f(g(x))" /> applies <InlineMath math="g" /> first,
          then <InlineMath math="f" />. Data pipelines are essentially function composition chains:
        </p>
        <BlockMath math="\text{pipeline}(x) = f_n(f_{n-1}(\cdots f_2(f_1(x))\cdots))" />
        <p>
          Decorators reverse the visual order — the outermost decorator is applied last:
        </p>
        <BlockMath math="\texttt{@A @B @C def f} \quad \Rightarrow \quad f = A(B(C(f)))" />

        <h3>Memoization and Dynamic Programming</h3>
        <p>
          Memoization (caching function results) transforms exponential-time recursive algorithms into
          polynomial-time ones. For the classic Fibonacci example, naive recursion
          has <InlineMath math="O(2^n)" /> time complexity:
        </p>
        <BlockMath math="T(n) = T(n-1) + T(n-2) + O(1) \Rightarrow T(n) = O(\phi^n) \text{ where } \phi = \frac{1+\sqrt{5}}{2}" />
        <p>
          With memoization via <code>@functools.lru_cache</code>, each subproblem is computed exactly once,
          reducing complexity to:
        </p>
        <BlockMath math="T(n) = O(n) \text{ time}, \quad O(n) \text{ space}" />
        <p>
          This same principle applies to caching expensive feature computations or API calls in data
          science pipelines.
        </p>

        <h3>Recursion Depth</h3>
        <p>
          Python&apos;s default recursion limit is 1000. Each recursive call adds a frame to the call stack,
          consuming <InlineMath math="O(d)" /> memory where <InlineMath math="d" /> is the recursion depth. Python does
          not have tail-call optimization, so converting deep recursion to iteration is sometimes necessary.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Function Signatures: *args, **kwargs, and Keyword-Only</h3>
        <CodeBlock
          language="python"
          title="function_signatures.py"
          code={`# Basic function with type hints
def normalize(values: list[float], method: str = "minmax") -> list[float]:
    """Normalize a list of values."""
    if method == "minmax":
        lo, hi = min(values), max(values)
        return [(v - lo) / (hi - lo) for v in values]
    elif method == "zscore":
        mean = sum(values) / len(values)
        std = (sum((v - mean)**2 for v in values) / len(values)) ** 0.5
        return [(v - mean) / std for v in values]
    raise ValueError(f"Unknown method: {method}")

# *args — collect extra positional arguments into a tuple
def log_metrics(*metrics: float, prefix: str = ""):
    for i, m in enumerate(metrics):
        print(f"{prefix}metric_{i}: {m:.4f}")

log_metrics(0.95, 0.87, 0.92, prefix="val_")
# val_metric_0: 0.9500
# val_metric_1: 0.8700
# val_metric_2: 0.9200

# **kwargs — collect extra keyword arguments into a dict
def create_model(model_type: str, **hyperparams):
    print(f"Creating {model_type} with params: {hyperparams}")
    # Pass kwargs through to the actual model constructor
    # This pattern is used EVERYWHERE in sklearn/pytorch

create_model("xgboost", n_estimators=100, max_depth=6, learning_rate=0.1)

# Keyword-only arguments (after *)
def train(data, *, epochs: int, lr: float):
    """epochs and lr MUST be passed as keyword arguments."""
    print(f"Training for {epochs} epochs at lr={lr}")

train([1, 2, 3], epochs=10, lr=0.01)
# train([1, 2, 3], 10, 0.01)  # TypeError!

# Positional-only arguments (before /) — Python 3.8+
def distance(x1, y1, x2, y2, /):
    """Arguments must be positional — prevents confusing kwarg names."""
    return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

print(distance(0, 0, 3, 4))  # 5.0
# distance(x1=0, y1=0, x2=3, y2=4)  # TypeError!`}
        />

        <h3>Lambda, Map, Filter, and Functional Patterns</h3>
        <CodeBlock
          language="python"
          title="functional_patterns.py"
          code={`# Lambda — anonymous function for simple expressions
square = lambda x: x ** 2
print(square(5))  # 25

# Common use: sorting with a key function
models = [
    {"name": "xgboost", "accuracy": 0.92},
    {"name": "rf", "accuracy": 0.89},
    {"name": "svm", "accuracy": 0.95},
]
sorted_models = sorted(models, key=lambda m: m["accuracy"], reverse=True)
print([m["name"] for m in sorted_models])  # ['svm', 'xgboost', 'rf']

# map() — apply function to every element (lazy!)
features = ["  Age ", " Income", "Education "]
cleaned = list(map(str.strip, features))
# ['Age', 'Income', 'Education']

# filter() — keep elements where function returns True
scores = [92, 45, 78, 31, 88, 67, 55]
passing = list(filter(lambda s: s >= 60, scores))
# [92, 78, 88, 67]

# functools.reduce — fold a sequence into a single value
from functools import reduce
product = reduce(lambda a, b: a * b, [1, 2, 3, 4, 5])
# 120 (= 5!)

# Partial application — fix some arguments
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)
print(square(5), cube(3))  # 25 27

# In pandas: partial is great for parameterized transforms
import pandas as pd
def clip_column(df, col, lower, upper):
    df[col] = df[col].clip(lower, upper)
    return df

clip_age = partial(clip_column, col="age", lower=0, upper=120)`}
        />

        <h3>Closures and Decorators</h3>
        <CodeBlock
          language="python"
          title="closures_decorators.py"
          code={`# CLOSURE — a function that captures variables from its enclosing scope
def make_multiplier(factor):
    """Returns a function that multiplies by 'factor'."""
    def multiplier(x):
        return x * factor  # 'factor' is captured from outer scope
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)
print(double(5))   # 10
print(triple(5))   # 15
# 'factor' lives on even after make_multiplier() returned!

# DECORATOR — a function that wraps another function
import time
import functools

def timer(func):
    """Decorator that prints execution time."""
    @functools.wraps(func)  # Preserves original function's name and docstring
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def train_model(epochs):
    """Train a dummy model."""
    total = sum(i**2 for i in range(epochs * 100000))
    return total

result = train_model(10)
# train_model took 0.2341s

# DECORATOR WITH ARGUMENTS — needs an extra layer
def retry(max_attempts=3, delay=1.0):
    """Retry a function on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"Attempt {attempt} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def fetch_data(url):
    """Simulate an unreliable API call."""
    import random
    if random.random() < 0.5:
        raise ConnectionError("Server unavailable")
    return {"data": [1, 2, 3]}

# BUILT-IN DECORATORS YOU SHOULD KNOW
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    """O(n) with memoization instead of O(2^n) without."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))  # 354224848179261915075 — instant!
print(fibonacci.cache_info())  # Shows hits, misses, etc.`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Use <code>functools.wraps</code> in every decorator</strong>: Without it, the decorated
            function loses its <code>__name__</code>, <code>__doc__</code>, and <code>__module__</code>, which
            breaks logging, debugging, and introspection tools.
          </li>
          <li>
            <strong><code>**kwargs</code> for pass-through configuration</strong>: Libraries like scikit-learn
            use <code>**kwargs</code> extensively so wrapper classes can forward hyperparameters to underlying
            estimators without knowing them in advance. Follow this pattern in your own pipeline code.
          </li>
          <li>
            <strong><code>@lru_cache</code> for expensive computations</strong>: Feature engineering often
            involves repeatedly computing the same value. Cache it. But beware: cached values stay in memory,
            and the arguments must be hashable (no lists or dicts — use tuples and frozensets).
          </li>
          <li>
            <strong>Lambda for quick pandas transforms</strong>: <code>df[&quot;col&quot;].apply(lambda x: x.strip().lower())</code> is
            a common pattern. But for anything more complex, define a named function — it&apos;s easier to test
            and debug.
          </li>
          <li>
            <strong>Decorator stacking for ML experiments</strong>: Combine <code>@timer</code>,
            <code>@retry</code>, and <code>@log_results</code> decorators to build robust experiment
            infrastructure without cluttering your core logic.
          </li>
          <li>
            <strong>Use <code>partial</code> over lambda for callbacks</strong>:
            <code>partial(func, arg1=val)</code> is more readable and inspectable than
            <code>lambda x: func(x, arg1=val)</code>.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Mutable default arguments (again)</strong>: <code>def f(x, cache=&#123;&#125;)</code> shares one dict.
            Use <code>None</code> as the default and create a new dict inside the function.
          </li>
          <li>
            <strong>Lambda late-binding in loops</strong>: <code>[lambda: i for i in range(5)]</code> creates
            5 functions that ALL return 4 (the final value of <code>i</code>). Fix with a default
            argument: <code>[lambda i=i: i for i in range(5)]</code>.
          </li>
          <li>
            <strong>Forgetting <code>@functools.wraps</code></strong>: Without it, <code>func.__name__</code>
            returns <code>&quot;wrapper&quot;</code> instead of the original name, causing confusion in logs,
            stack traces, and documentation generators.
          </li>
          <li>
            <strong>Overusing lambda</strong>: If your lambda has an <code>if/else</code>, a function call chain,
            or is longer than about 40 characters, use a named function. Lambdas are for trivial expressions only.
          </li>
          <li>
            <strong>Confusing <code>*args</code> unpacking with <code>*</code> in a signature</strong>:
            <code>def f(*args)</code> packs arguments into a tuple. <code>f(*my_list)</code> unpacks a list into
            separate arguments. They&apos;re complementary operations that look confusingly similar.
          </li>
          <li>
            <strong>Closures over loop variables</strong>: Similar to the lambda issue — if you create closures
            inside a loop, they all capture the <em>same</em> variable (by reference, not by value). The value
            at call time will be the loop&apos;s final value unless you bind it as a default argument.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Implement a <code>@cache_with_ttl(seconds)</code> decorator that caches
          function results but invalidates the cache after a specified number of seconds. The function takes
          only hashable arguments.
        </p>
        <p><strong>Solution:</strong></p>
        <CodeBlock
          language="python"
          title="cache_with_ttl.py"
          code={`import time
import functools

def cache_with_ttl(seconds):
    """Decorator that caches results with a time-to-live.

    This demonstrates:
    1. Decorator with arguments (3 layers of nesting)
    2. Closures capturing the cache dict and TTL
    3. *args/**kwargs forwarding
    4. functools.wraps for metadata preservation
    """
    def decorator(func):
        cache = {}  # Captured by closure: {args_key: (result, timestamp)}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hashable key from the arguments
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()

            # Check if we have a valid cached result
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < seconds:
                    print(f"Cache HIT for {func.__name__}{args}")
                    return result
                else:
                    print(f"Cache EXPIRED for {func.__name__}{args}")

            # Compute and cache the result
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            print(f"Cache MISS for {func.__name__}{args}")
            return result

        # Expose cache for testing/debugging
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        return wrapper
    return decorator

# Usage
@cache_with_ttl(seconds=2)
def fetch_stock_price(ticker):
    """Simulate an expensive API call."""
    time.sleep(0.1)  # Simulate network latency
    return {"ticker": ticker, "price": 150.0 + hash(ticker) % 50}

# First call — cache miss
result1 = fetch_stock_price("AAPL")
# Cache MISS for fetch_stock_price('AAPL',)

# Second call — cache hit (fast!)
result2 = fetch_stock_price("AAPL")
# Cache HIT for fetch_stock_price('AAPL',)

# Wait for TTL to expire
time.sleep(2.1)

# Third call — cache expired, recompute
result3 = fetch_stock_price("AAPL")
# Cache EXPIRED for fetch_stock_price('AAPL',)

# Verify metadata is preserved
print(fetch_stock_price.__name__)  # "fetch_stock_price" (not "wrapper")
print(fetch_stock_price.__doc__)   # "Simulate an expensive API call."`}
        />
        <p>
          This question tests closures (the <code>cache</code> dict lives in the decorator&apos;s scope),
          decorator mechanics (three levels of nesting for parameterized decorators),
          <code>*args/**kwargs</code> forwarding, and practical software engineering skills. A follow-up
          might ask about thread safety (add a <code>threading.Lock</code>) or LRU eviction (limit cache
          size and evict the oldest entry).
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li>
            <strong>Fluent Python, Chapter 7 &amp; 9</strong> — Luciano Ramalho covers first-class functions,
            closures, and decorators with exceptional clarity
          </li>
          <li>
            <strong>PEP 3102 — Keyword-Only Arguments</strong> — The rationale behind the <code>*</code> separator
            in function signatures
          </li>
          <li>
            <strong>PEP 570 — Positional-Only Parameters</strong> — Explains the <code>/</code> syntax and when to use it
          </li>
          <li>
            <strong>functools documentation</strong> — <code>docs.python.org/3/library/functools.html</code> — covers
            <code>lru_cache</code>, <code>partial</code>, <code>reduce</code>, <code>wraps</code>,
            <code>singledispatch</code>, and <code>cache</code>
          </li>
          <li>
            <strong>Real Python: Primer on Python Decorators</strong> — A thorough tutorial with progressive
            examples from basic to advanced decorator patterns
          </li>
        </ul>
      </TopicSection>
    </div>
  );
}
