"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function ClassesAndOOP() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Object-oriented programming (OOP) in Python is about organizing code around <strong>objects</strong> that
          bundle data (attributes) and behavior (methods) together. A <strong>class</strong> is a blueprint;
          an <strong>instance</strong> is a specific object created from that blueprint. You use OOP in data
          science more than you might think: every scikit-learn estimator is a class (with <code>.fit()</code> and
          <code>.predict()</code>), every PyTorch model subclasses <code>nn.Module</code>, and every pandas
          DataFrame is an object with hundreds of methods.
        </p>
        <p>
          Python&apos;s OOP has a distinctive character. Unlike Java or C++, Python uses <strong>duck typing</strong> —
          &quot;if it walks like a duck and quacks like a duck, it&apos;s a duck.&quot; You don&apos;t need to
          declare interfaces or abstract types for polymorphism to work; if an object has the right methods,
          it works. Python also has <strong>dunder (double underscore) methods</strong> — special methods
          like <code>__init__</code>, <code>__repr__</code>, <code>__add__</code>, <code>__len__</code> — that
          let your classes integrate seamlessly with Python&apos;s syntax and built-in functions.
        </p>
        <p>
          <strong>Inheritance</strong> lets you build specialized classes on top of general ones.
          <strong>Multiple inheritance</strong> is supported but should be used carefully — Python resolves
          method lookup using the <strong>Method Resolution Order (MRO)</strong>, which follows the C3
          linearization algorithm. In practice, data scientists mostly encounter inheritance when subclassing
          library base classes (like <code>torch.nn.Module</code> or <code>sklearn.base.BaseEstimator</code>)
          rather than designing deep class hierarchies. The modern Python philosophy is
          &quot;composition over inheritance&quot; — prefer objects that <em>contain</em> other objects over
          deep inheritance trees.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Method Resolution Order (MRO) and C3 Linearization</h3>
        <p>
          When a class inherits from multiple parents, Python must determine the order in which to search
          for methods. The <strong>C3 linearization</strong> algorithm produces a consistent, monotonic ordering.
          For a class <InlineMath math="C" /> with parents <InlineMath math="B_1, B_2, \ldots, B_n" />:
        </p>
        <BlockMath math="L(C) = C + \text{merge}(L(B_1), L(B_2), \ldots, L(B_n), [B_1, B_2, \ldots, B_n])" />
        <p>
          The merge operation takes the first element of the first list that does not appear in the tail of
          any other list, adds it to the result, and removes it from all lists. This guarantees that:
        </p>
        <ul>
          <li>A class always appears before its parents.</li>
          <li>The order specified in the class definition is preserved.</li>
          <li>If two classes inherit from the same parent, the parent appears only once and after both children.</li>
        </ul>

        <h3>Computational Complexity of Class Operations</h3>
        <p>
          Attribute lookup in Python uses a dictionary under the hood (<code>__dict__</code>). Instance attribute
          access is <InlineMath math="O(1)" /> average-case. Method resolution walks the MRO chain, which
          is <InlineMath math="O(d)" /> where <InlineMath math="d" /> is the depth of the inheritance
          hierarchy — but Python caches this, so repeated lookups are effectively <InlineMath math="O(1)" />.
        </p>
        <p>
          Using <code>__slots__</code> replaces the per-instance <code>__dict__</code> with a fixed-size
          array, reducing memory usage from roughly <InlineMath math="O(k)" /> for a dict
          of <InlineMath math="k" /> attributes to a compact struct with no hash table overhead. For a million
          instances with 5 attributes each, <code>__slots__</code> can save 40-50% of memory.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Classes, Dunder Methods, and Properties</h3>
        <CodeBlock
          language="python"
          title="class_fundamentals.py"
          code={`class Dataset:
    """A simple dataset container demonstrating core OOP patterns."""

    # Class variable — shared across all instances
    default_split = 0.8

    def __init__(self, features, labels, name="unnamed"):
        """Constructor — called when you create an instance."""
        # Instance variables — unique to each instance
        self.features = features
        self.labels = labels
        self.name = name
        self._is_shuffled = False  # Convention: underscore = "private"

    def __repr__(self):
        """Unambiguous string for debugging (shown in REPL)."""
        return f"Dataset(name='{self.name}', n={len(self)}, features={self.features.shape[1]})"

    def __str__(self):
        """User-friendly string (used by print())."""
        return f"Dataset '{self.name}' with {len(self)} samples"

    def __len__(self):
        """Enable len(dataset)."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Enable dataset[0] and dataset[10:20] indexing."""
        return self.features[idx], self.labels[idx]

    def __add__(self, other):
        """Enable dataset1 + dataset2 to concatenate."""
        import numpy as np
        return Dataset(
            features=np.vstack([self.features, other.features]),
            labels=np.concatenate([self.labels, other.labels]),
            name=f"{self.name}+{other.name}",
        )

    def __eq__(self, other):
        """Enable dataset1 == dataset2."""
        import numpy as np
        return (np.array_equal(self.features, other.features)
                and np.array_equal(self.labels, other.labels))

    @property
    def shape(self):
        """Computed attribute — access as dataset.shape, not dataset.shape()."""
        return self.features.shape

    @property
    def is_shuffled(self):
        return self._is_shuffled

    def split(self, ratio=None):
        """Split into train and test sets."""
        ratio = ratio or self.default_split
        n = int(len(self) * ratio)
        return (
            Dataset(self.features[:n], self.labels[:n], f"{self.name}_train"),
            Dataset(self.features[n:], self.labels[n:], f"{self.name}_test"),
        )

    @classmethod
    def from_csv(cls, filepath, label_col="target"):
        """Alternative constructor — create Dataset from a CSV file."""
        import pandas as pd
        import numpy as np
        df = pd.read_csv(filepath)
        labels = df[label_col].values
        features = df.drop(columns=[label_col]).values
        return cls(features, labels, name=filepath.split("/")[-1])

    @staticmethod
    def normalize(array):
        """Utility — doesn't need self or cls."""
        return (array - array.mean(axis=0)) / array.std(axis=0)

# Usage
import numpy as np
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

ds = Dataset(X, y, name="my_data")
print(repr(ds))      # Dataset(name='my_data', n=100, features=5)
print(len(ds))       # 100
print(ds.shape)      # (100, 5)
print(ds[0])         # (array([...]), 0)
train, test = ds.split(0.8)`}
        />

        <h3>Inheritance and Polymorphism</h3>
        <CodeBlock
          language="python"
          title="inheritance.py"
          code={`from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Abstract base class — cannot be instantiated directly."""

    def __init__(self, name):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y):
        """Subclasses MUST implement this."""
        pass

    @abstractmethod
    def predict(self, X):
        """Subclasses MUST implement this."""
        pass

    def fit_predict(self, X, y):
        """Concrete method that uses abstract methods — Template Method pattern."""
        self.fit(X, y)
        return self.predict(X)

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"


class MeanPredictor(BaseModel):
    """Predicts the mean of training labels — useful baseline."""

    def fit(self, X, y):
        self.mean_ = np.mean(y)
        self.is_fitted = True
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class KNNPredictor(BaseModel):
    """Simple k-nearest neighbors from scratch."""

    def __init__(self, name, k=5):
        super().__init__(name)  # Call parent's __init__
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.is_fitted = True
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_idx = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_idx]
            predictions.append(np.mean(nearest_labels))
        return np.array(predictions)


# Polymorphism in action — same interface, different behavior
def evaluate(model: BaseModel, X_train, y_train, X_test, y_test):
    """Works with ANY subclass of BaseModel."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = np.mean((y_test - preds) ** 2)
    print(f"{model}: MSE = {mse:.4f}")
    return mse

X = np.random.randn(200, 3)
y = X @ np.array([1.5, -2.0, 0.5]) + np.random.randn(200) * 0.1

models = [MeanPredictor("baseline"), KNNPredictor("knn", k=5)]
for m in models:
    evaluate(m, X[:150], y[:150], X[150:], y[150:])

# Check MRO
print(KNNPredictor.__mro__)
# (KNNPredictor, BaseModel, ABC, object)`}
        />

        <h3>Dataclasses and Modern Python OOP</h3>
        <CodeBlock
          language="python"
          title="modern_oop.py"
          code={`from dataclasses import dataclass, field
from typing import Optional

# dataclass eliminates boilerplate __init__, __repr__, __eq__
@dataclass
class ExperimentConfig:
    model_name: str
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    tags: list[str] = field(default_factory=list)  # Mutable default done right!
    notes: Optional[str] = None

    def __post_init__(self):
        """Validation after __init__ runs."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")

# Auto-generates __init__, __repr__, __eq__
config = ExperimentConfig("xgboost", learning_rate=0.01, tags=["baseline"])
print(config)
# ExperimentConfig(model_name='xgboost', learning_rate=0.01, epochs=100,
#                  batch_size=32, tags=['baseline'], notes=None)

# Frozen dataclass — immutable (like a named tuple but better)
@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @property
    def norm(self):
        return (self.x**2 + self.y**2) ** 0.5

p = Point(3.0, 4.0)
print(p.norm)  # 5.0
# p.x = 10  # FrozenInstanceError!

# __slots__ for memory-efficient classes
class SparseFeature:
    """Millions of these? Use __slots__ to save memory."""
    __slots__ = ("index", "value")

    def __init__(self, index: int, value: float):
        self.index = index
        self.value = value

    def __repr__(self):
        return f"SparseFeature({self.index}, {self.value})"

# With __slots__:  ~56 bytes per instance
# Without __slots__: ~152 bytes per instance (dict overhead)
# For 10M instances: 560MB vs 1.52GB

# Protocol-based duck typing (Python 3.8+)
from typing import Protocol, runtime_checkable

@runtime_checkable
class Fittable(Protocol):
    def fit(self, X, y) -> "Fittable": ...
    def predict(self, X): ...

# Any class with fit() and predict() satisfies this Protocol
# No inheritance required — true structural typing
print(isinstance(MeanPredictor("test"), Fittable))  # True`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Subclass library base classes, don&apos;t reinvent them</strong>: Inherit
            from <code>sklearn.base.BaseEstimator</code> and relevant mixins (<code>ClassifierMixin</code>,
            <code>RegressorMixin</code>) to get <code>get_params()</code>, <code>set_params()</code>,
            <code>score()</code>, and compatibility with sklearn pipelines and grid search for free.
          </li>
          <li>
            <strong>Use <code>dataclass</code> for experiment configs</strong>: Instead of passing dicts around,
            define a dataclass. You get type checking, default values, immutability (<code>frozen=True</code>),
            and automatic <code>__repr__</code> for logging.
          </li>
          <li>
            <strong>Implement <code>__repr__</code> on every class you write</strong>: When debugging a
            pipeline at 2 AM, <code>Model(name=&apos;xgb&apos;, fitted=True, n_features=50)</code> is infinitely
            more helpful than <code>&lt;Model object at 0x7f...&gt;</code>.
          </li>
          <li>
            <strong>Prefer composition over deep inheritance</strong>: Instead of <code>class MyPipeline(TransformerMixin, RegressorMixin, BaseEstimator)</code>,
            consider a class that <em>contains</em> a transformer and a regressor as attributes. Deep hierarchies
            are hard to reason about and fragile.
          </li>
          <li>
            <strong><code>__slots__</code> for high-volume objects</strong>: If you create millions of small
            objects (feature records, graph nodes, tokens), <code>__slots__</code> can cut memory use in half.
          </li>
          <li>
            <strong>Context managers for resource handling</strong>: Implement <code>__enter__</code> and
            <code>__exit__</code> (or use <code>@contextmanager</code>) for database connections, file handles,
            GPU memory management, and temporary directory creation.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Forgetting <code>super().__init__()</code> in subclasses</strong>: If your subclass
            defines <code>__init__</code> but doesn&apos;t call <code>super().__init__()</code>, the parent&apos;s
            initialization is skipped entirely. Attributes set in the parent&apos;s <code>__init__</code> will be
            missing, causing subtle <code>AttributeError</code>s later.
          </li>
          <li>
            <strong>Class variables vs. instance variables</strong>: <code>class Foo: items = []</code> creates
            ONE list shared by all instances. To give each instance its own list, assign it
            in <code>__init__</code>: <code>self.items = []</code>.
          </li>
          <li>
            <strong>Using inheritance when composition would be simpler</strong>: &quot;Is-a&quot; vs. &quot;has-a&quot; — a
            <code>DataPipeline</code> is not a type of <code>list</code>; it <em>has</em> a list of steps.
            Inheriting from <code>list</code> exposes every list method, many of which make no sense for your class.
          </li>
          <li>
            <strong>Mutable class attributes in dataclasses</strong>: <code>@dataclass</code> with a
            default <code>list</code> or <code>dict</code> raises <code>ValueError</code> on purpose. Use
            <code>field(default_factory=list)</code> — this is Python protecting you from the shared-mutable-default bug.
          </li>
          <li>
            <strong>Overriding <code>__eq__</code> without <code>__hash__</code></strong>: If you
            define <code>__eq__</code>, Python sets <code>__hash__</code> to <code>None</code>, making your
            objects unhashable (can&apos;t be dict keys or set members). If you need both, define <code>__hash__</code>
            explicitly.
          </li>
          <li>
            <strong>Diamond inheritance confusion</strong>: With multiple inheritance, the MRO determines
            which parent&apos;s method gets called. Always use <code>super()</code> instead of calling parent
            classes by name to ensure the MRO is followed correctly.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Design a <code>Matrix</code> class from scratch that supports addition,
          multiplication, transpose, and pretty-printing using dunder methods. It should work with Python&apos;s
          built-in syntax: <code>A + B</code>, <code>A @ B</code>, <code>A.T</code>, <code>print(A)</code>.
        </p>
        <p><strong>Solution:</strong></p>
        <CodeBlock
          language="python"
          title="matrix_class.py"
          code={`class Matrix:
    """A 2D matrix with operator overloading via dunder methods."""

    def __init__(self, data: list[list[float]]):
        if not data or not data[0]:
            raise ValueError("Matrix cannot be empty")
        if len(set(len(row) for row in data)) != 1:
            raise ValueError("All rows must have the same length")
        self._data = [row[:] for row in data]  # Defensive copy
        self.rows = len(data)
        self.cols = len(data[0])

    def __repr__(self):
        return f"Matrix({self.rows}x{self.cols})"

    def __str__(self):
        col_widths = [
            max(len(f"{self._data[r][c]:.2f}") for r in range(self.rows))
            for c in range(self.cols)
        ]
        lines = []
        for row in self._data:
            formatted = [f"{val:>{col_widths[j]}.2f}" for j, val in enumerate(row)]
            lines.append("[ " + "  ".join(formatted) + " ]")
        return "\\n".join(lines)

    def __getitem__(self, key):
        """Enable A[i, j] and A[i] indexing."""
        if isinstance(key, tuple):
            i, j = key
            return self._data[i][j]
        return self._data[key]

    def __eq__(self, other):
        return isinstance(other, Matrix) and self._data == other._data

    def __add__(self, other):
        """A + B — element-wise addition."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Shape mismatch: {self.rows}x{self.cols} vs {other.rows}x{other.cols}")
        return Matrix([
            [self._data[i][j] + other._data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __matmul__(self, other):
        """A @ B — matrix multiplication."""
        if self.cols != other.rows:
            raise ValueError(f"Cannot multiply {self.rows}x{self.cols} @ {other.rows}x{other.cols}")
        return Matrix([
            [
                sum(self._data[i][k] * other._data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def __mul__(self, scalar):
        """A * 3 — scalar multiplication."""
        return Matrix([[val * scalar for val in row] for row in self._data])

    def __rmul__(self, scalar):
        """3 * A — scalar multiplication (reversed)."""
        return self.__mul__(scalar)

    @property
    def T(self):
        """Transpose — access as A.T (like numpy)."""
        return Matrix([
            [self._data[i][j] for i in range(self.rows)]
            for j in range(self.cols)
        ])

    def __len__(self):
        return self.rows

# Demo
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A:")
print(A)
# [ 1.00  2.00 ]
# [ 3.00  4.00 ]

print("\\nA + B:")
print(A + B)
# [  6.00   8.00 ]
# [ 10.00  12.00 ]

print("\\nA @ B:")
print(A @ B)
# [ 19.00  22.00 ]
# [ 43.00  50.00 ]

print("\\nA.T:")
print(A.T)
# [ 1.00  3.00 ]
# [ 2.00  4.00 ]

print("\\n3 * A:")
print(3 * A)
# [ 3.00   6.00 ]
# [ 9.00  12.00 ]`}
        />
        <p>
          This question tests understanding of dunder methods (<code>__add__</code>, <code>__matmul__</code>,
          <code>__rmul__</code>, <code>__repr__</code> vs. <code>__str__</code>), properties, defensive copying,
          and input validation. A common follow-up is: &quot;Why did you implement both <code>__mul__</code>
          and <code>__rmul__</code>?&quot; Answer: <code>__mul__</code> handles <code>A * 3</code>, but
          for <code>3 * A</code>, Python first tries <code>int.__mul__(3, A)</code>, which returns
          <code>NotImplemented</code>, so Python falls back to <code>A.__rmul__(3)</code>.
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li>
            <strong>Fluent Python, Part IV (Object-Oriented Idioms)</strong> — Chapters 11-14 cover the Python data
            model, inheritance, operator overloading, and abstract base classes in depth
          </li>
          <li>
            <strong>Python Data Model documentation</strong> — <code>docs.python.org/3/reference/datamodel.html</code> — the
            complete reference for all dunder methods and how Python uses them
          </li>
          <li>
            <strong>Raymond Hettinger &quot;Super Considered Super&quot;</strong> — The definitive talk on how
            <code>super()</code> and the MRO work in Python, with practical cooperative multiple inheritance
            patterns
          </li>
          <li>
            <strong>PEP 557 — Data Classes</strong> — The full specification for <code>@dataclass</code>
            including <code>field()</code>, <code>__post_init__</code>, inheritance, and frozen instances
          </li>
          <li>
            <strong>Design Patterns in Python</strong> — Brandon Rhodes&apos; website
            (<code>python-patterns.guide</code>) covers Gang of Four patterns adapted for idiomatic Python
          </li>
        </ul>
      </TopicSection>
    </div>
  );
}
