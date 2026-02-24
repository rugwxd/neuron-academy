"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function VariablesAndTypes() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          In Python, a <strong>variable</strong> is a name that points to an object in memory. Unlike
          statically-typed languages like Java or C++, Python is <strong>dynamically typed</strong> — you
          don&apos;t declare a variable&apos;s type; the interpreter figures it out from the value you assign.
          When you write <code>x = 42</code>, Python creates an integer object with value 42 and binds the
          name <code>x</code> to it. If you later write <code>x = &quot;hello&quot;</code>, the name now
          points to a completely different object — a string. The old integer object gets garbage-collected
          if nothing else references it.
        </p>
        <p>
          Python&apos;s core <strong>data types</strong> include <code>int</code>, <code>float</code>,
          <code>bool</code>, <code>str</code>, <code>list</code>, <code>tuple</code>, <code>dict</code>,
          <code>set</code>, and <code>NoneType</code>. The distinction between <strong>mutable</strong> types
          (lists, dicts, sets) and <strong>immutable</strong> types (ints, floats, strings, tuples) is one
          of the most important concepts in Python. Mutable objects can be changed in place; immutable
          objects cannot — any &quot;modification&quot; creates a new object.
        </p>
        <p>
          <strong>Operators</strong> in Python are syntactic sugar for special methods (dunder methods).
          When you write <code>a + b</code>, Python actually calls <code>a.__add__(b)</code>. This is why
          <code>+</code> can concatenate strings, add numbers, and merge lists — each type defines its own
          <code>__add__</code>. Understanding this is crucial because libraries like NumPy and pandas
          override these operators to work element-wise on arrays and DataFrames, which is the foundation
          of vectorized computation in data science.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Numeric Representation and Precision</h3>
        <p>
          Python <code>int</code> objects have <strong>arbitrary precision</strong> — they can be as large as
          your memory allows. Floats, however, follow the IEEE 754 double-precision standard, storing numbers
          in 64 bits: 1 sign bit, 11 exponent bits, and 52 mantissa bits. A float can represent values in the
          range:
        </p>
        <BlockMath math="\pm 2.2 \times 10^{-308} \text{ to } \pm 1.8 \times 10^{308}" />
        <p>
          with approximately <InlineMath math="15{-}17" /> significant decimal digits of precision. This leads to
          the classic floating-point surprise:
        </p>
        <BlockMath math="0.1 + 0.2 = 0.30000000000000004" />
        <p>
          because <InlineMath math="0.1" /> cannot be represented exactly in base-2. The machine epsilon for
          double precision is:
        </p>
        <BlockMath math="\epsilon = 2^{-52} \approx 2.22 \times 10^{-16}" />
        <p>
          This matters in data science when comparing floating-point results. Use <code>np.isclose()</code> or
          <code>math.isclose()</code> instead of <code>==</code>.
        </p>

        <h3>Hashing and Dictionary Lookup</h3>
        <p>
          Python dictionaries and sets use <strong>hash tables</strong>. The average-case time complexity for
          lookup, insertion, and deletion is:
        </p>
        <BlockMath math="O(1) \text{ average}, \quad O(n) \text{ worst-case (hash collisions)}" />
        <p>
          Only <strong>hashable</strong> (immutable) objects can be dictionary keys or set members. That&apos;s
          why lists can&apos;t be dict keys but tuples can. The hash function maps an object to an integer index
          in the underlying array, and Python resolves collisions via open addressing.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Variables, Types, and Mutability</h3>
        <CodeBlock
          language="python"
          title="variables_basics.py"
          code={`# Dynamic typing — the same name can point to different types
x = 42          # int
x = 3.14        # float (previous int is garbage-collected)
x = "hello"     # str

# Check type at runtime
print(type(x))          # <class 'str'>
print(isinstance(x, str))  # True

# Multiple assignment
a, b, c = 1, 2.0, "three"

# Immutable vs mutable — the KEY distinction
# Immutable: int, float, str, tuple, frozenset, bool
name = "Alice"
# name[0] = "B"  # TypeError! Strings are immutable

# Mutable: list, dict, set
scores = [90, 85, 92]
scores[0] = 95  # This works — lists are mutable

# GOTCHA: aliasing with mutable objects
a = [1, 2, 3]
b = a           # b points to the SAME list object
b.append(4)
print(a)        # [1, 2, 3, 4] — a is affected!
print(a is b)   # True — same object in memory

# Fix: make a copy
c = a.copy()    # shallow copy — new list, same elements
c.append(5)
print(a)        # [1, 2, 3, 4] — a is NOT affected
print(a is c)   # False — different objects

# Deep copy for nested structures
import copy
nested = [[1, 2], [3, 4]]
shallow = nested.copy()
deep = copy.deepcopy(nested)

shallow[0][0] = 99
print(nested[0][0])  # 99 — shallow copy shares inner lists!

deep[1][0] = 99
print(nested[1][0])  # 3 — deep copy is fully independent`}
        />

        <h3>Operators and Type Coercion</h3>
        <CodeBlock
          language="python"
          title="operators.py"
          code={`# Arithmetic operators
print(7 / 2)    # 3.5   — true division (always returns float)
print(7 // 2)   # 3     — floor division
print(7 % 2)    # 1     — modulo
print(2 ** 10)  # 1024  — exponentiation

# Comparison operators
print(1 == True)   # True  — bool is a subclass of int
print(1 is True)   # False — different objects in memory

# Chained comparisons (Pythonic!)
x = 5
print(1 < x < 10)      # True — equivalent to (1 < x) and (x < 10)
print(1 < x < 3)       # False

# Logical operators: and, or, not (short-circuit!)
# 'and' returns the first falsy value, or the last value
print(0 and "hello")    # 0
print(1 and "hello")    # "hello"

# 'or' returns the first truthy value, or the last value
print(0 or "default")   # "default"
print("value" or "default")  # "value"

# Falsy values: 0, 0.0, "", [], {}, set(), None, False
# Everything else is truthy

# Bitwise operators (useful for feature flags, masks)
flags = 0b1010
mask  = 0b1100
print(bin(flags & mask))   # 0b1000 — AND
print(bin(flags | mask))   # 0b1110 — OR
print(bin(flags ^ mask))   # 0b0110 — XOR

# Walrus operator (:=) — assign and use in one expression (Python 3.8+)
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
if (n := len(data)) > 5:
    print(f"List has {n} elements, processing...")

# Unpacking operators
first, *middle, last = [1, 2, 3, 4, 5]
print(first, middle, last)  # 1 [2, 3, 4] 5`}
        />

        <h3>Data Science Types in Practice</h3>
        <CodeBlock
          language="python"
          title="ds_types.py"
          code={`import numpy as np
import pandas as pd

# NumPy arrays are typed — much faster than Python lists
arr = np.array([1.0, 2.0, 3.0])  # float64 by default
print(arr.dtype)  # float64

# Operators are element-wise on arrays (not Python lists!)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)    # [5 7 9]  — element-wise add
print(a * b)    # [4 10 18] — element-wise multiply
print(a @ b)    # 32 — dot product

# Compare with Python lists
print([1, 2, 3] + [4, 5, 6])  # [1, 2, 3, 4, 5, 6] — concatenation!
print([1, 2, 3] * 2)          # [1, 2, 3, 1, 2, 3] — repetition!

# Pandas dtypes extend NumPy
df = pd.DataFrame({
    "name": ["Alice", "Bob"],          # object (string)
    "age": [30, 25],                    # int64
    "score": [95.5, 87.3],             # float64
    "passed": [True, False],            # bool
    "date": pd.to_datetime(["2024-01-01", "2024-06-15"])  # datetime64
})
print(df.dtypes)

# Nullable integer type (handles NaN in integer columns)
s = pd.array([1, 2, None, 4], dtype=pd.Int64Dtype())
print(s)  # [1, 2, <NA>, 4] — no silent float conversion`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Use type hints in production DS code</strong>: <code>def process(data: pd.DataFrame) -&gt; np.ndarray</code> makes
            your code self-documenting and enables IDE autocompletion and static analysis with mypy.
          </li>
          <li>
            <strong>Know your dtypes in pandas</strong>: A column of integers with even one <code>NaN</code> gets
            silently upcast to <code>float64</code>. Use <code>pd.Int64Dtype()</code> for nullable integers.
          </li>
          <li>
            <strong>Avoid <code>==</code> for float comparison</strong>: Use <code>np.isclose(a, b)</code> or
            <code>np.allclose(a, b)</code> when comparing arrays of floats, especially after chain calculations.
          </li>
          <li>
            <strong>Prefer tuples over lists for fixed-size data</strong>: Tuples use less memory, are hashable
            (can be dict keys), and signal &quot;this data shouldn&apos;t change.&quot;
          </li>
          <li>
            <strong>Use <code>dict</code> for O(1) lookups</strong>: If you find yourself scanning a list
            repeatedly, convert it to a dict or set. This is a constant pattern in feature engineering pipelines.
          </li>
          <li>
            <strong>String formatting</strong>: Use f-strings (<code>f&quot;value: &#123;x:.2f&#125;&quot;</code>) — they&apos;re
            the fastest and most readable option. Avoid <code>%</code> formatting and <code>.format()</code> in new code.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Mutable default arguments</strong>: <code>def f(items=[])</code> shares ONE list across all
            calls. Use <code>def f(items=None)</code> and then <code>items = items or []</code> inside the function.
          </li>
          <li>
            <strong>Confusing <code>is</code> with <code>==</code></strong>: <code>is</code> checks object identity
            (same memory address), <code>==</code> checks value equality. Use <code>is</code> only for <code>None</code>,
            <code>True</code>, <code>False</code>.
          </li>
          <li>
            <strong>Integer caching gotcha</strong>: Python caches small integers (-5 to 256), so
            <code>a = 256; b = 256; a is b</code> is <code>True</code>, but <code>a = 257; b = 257; a is b</code> may
            be <code>False</code>. Never rely on <code>is</code> for number comparisons.
          </li>
          <li>
            <strong>Shallow copy surprise</strong>: <code>list.copy()</code> and <code>[:]</code> only copy one level deep.
            Nested mutable objects (lists of lists) still share references. Use <code>copy.deepcopy()</code> for nested structures.
          </li>
          <li>
            <strong>Silent type coercion in pandas</strong>: Inserting a <code>None</code> into an integer Series
            silently converts the entire column to <code>float64</code>. Always check <code>df.dtypes</code> after data
            cleaning.
          </li>
          <li>
            <strong>String concatenation in loops</strong>: Strings are immutable, so <code>s += &quot;x&quot;</code> in a
            loop creates a new string every iteration — <InlineMath math="O(n^2)" /> total. Use <code>&quot;&quot;.join(parts)</code> instead.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> What is the output of the following code? Explain why.
        </p>
        <CodeBlock
          language="python"
          title="interview_question.py"
          code={`def append_to(element, target=[]):
    target.append(element)
    return target

print(append_to(1))
print(append_to(2))
print(append_to(3))`}
        />
        <p><strong>Solution:</strong></p>
        <p>
          The output is:
        </p>
        <CodeBlock
          language="python"
          code={`[1]
[1, 2]
[1, 2, 3]`}
        />
        <p>
          Most people expect each call to return a single-element list, but that&apos;s not what happens.
          The default argument <code>target=[]</code> is evaluated <strong>once</strong> — when the function
          is defined, not each time it&apos;s called. All calls that don&apos;t provide <code>target</code>
          share the <strong>same list object</strong>. Each call mutates that shared list.
        </p>
        <p>
          You can verify this by inspecting <code>append_to.__defaults__</code>, which holds the default
          values. After three calls, it shows <code>([1, 2, 3],)</code>.
        </p>
        <p><strong>The fix:</strong></p>
        <CodeBlock
          language="python"
          title="fixed_version.py"
          code={`def append_to(element, target=None):
    if target is None:
        target = []  # New list created on each call
    target.append(element)
    return target

print(append_to(1))  # [1]
print(append_to(2))  # [2]
print(append_to(3))  # [3]`}
        />
        <p>
          This is one of the most common Python interview questions because it tests understanding of
          object identity, mutability, and how default arguments work under the hood. The general rule:
          <strong>never use a mutable object as a default argument</strong>.
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li>
            <strong>Python Data Model</strong> — Chapter 1 of Luciano Ramalho&apos;s <em>Fluent Python</em> (2nd ed.) — the
            definitive guide to how Python objects actually work
          </li>
          <li>
            <strong>&quot;What the f*ck Python&quot;</strong> — A curated collection of surprising Python behaviors
            and gotchas with explanations: <code>github.com/satwikkansal/wtfpython</code>
          </li>
          <li>
            <strong>IEEE 754 Floating Point</strong> — David Goldberg&apos;s classic paper &quot;What Every Computer
            Scientist Should Know About Floating-Point Arithmetic&quot;
          </li>
          <li>
            <strong>Python docs: Built-in Types</strong> — <code>docs.python.org/3/library/stdtypes.html</code> — the
            authoritative reference for all built-in types
          </li>
          <li>
            <strong>NumPy dtype documentation</strong> — <code>numpy.org/doc/stable/reference/arrays.dtypes.html</code> — essential
            for understanding how data is stored in arrays
          </li>
        </ul>
      </TopicSection>
    </div>
  );
}
