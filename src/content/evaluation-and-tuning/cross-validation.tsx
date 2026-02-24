"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function CrossValidation() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          You trained a model, and it gets 95% accuracy on your test set. Great? Maybe — but what if you got lucky with the split? What if the 20% of data you held out happened to be easy examples? <strong>Cross-validation</strong> solves this by testing on <em>every</em> part of the data, not just one arbitrary slice.
        </p>
        <p>
          In <strong>k-fold cross-validation</strong>, you split your data into <InlineMath math="k" /> equal parts (folds). You train on <InlineMath math="k-1" /> folds and test on the remaining one. Repeat this <InlineMath math="k" /> times, each time using a different fold as the test set. Your final score is the average across all <InlineMath math="k" /> runs. This gives you both a more reliable performance estimate and a measure of variability (standard deviation across folds).
        </p>
        <p>
          Different problems need different CV strategies. <strong>Stratified k-fold</strong> preserves the class distribution in each fold — essential for imbalanced datasets. <strong>Time series CV</strong> respects temporal order — you never train on future data. <strong>Nested CV</strong> uses an inner loop for hyperparameter tuning and an outer loop for unbiased evaluation — the gold standard for model selection.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>k-Fold Cross-Validation</h3>
        <p>Partition data <InlineMath math="D" /> into <InlineMath math="k" /> disjoint folds <InlineMath math="D_1, D_2, \ldots, D_k" />:</p>
        <BlockMath math="\text{CV}(k) = \frac{1}{k}\sum_{i=1}^{k} L\left(f_{D \setminus D_i}, D_i\right)" />
        <p>where <InlineMath math="f_{D \setminus D_i}" /> is the model trained on everything except fold <InlineMath math="i" />, and <InlineMath math="L" /> is the loss evaluated on fold <InlineMath math="i" />.</p>

        <h3>Bias-Variance of the CV Estimator</h3>
        <p>The CV estimate itself has bias and variance:</p>
        <ul>
          <li><strong>Bias</strong>: Each fold trains on <InlineMath math="\frac{k-1}{k}n" /> samples. Fewer than the full dataset, so the estimate is pessimistic. Higher <InlineMath math="k" /> reduces bias (LOOCV has minimal bias).</li>
          <li><strong>Variance</strong>: LOOCV (<InlineMath math="k = n" />) has high variance because the <InlineMath math="n" /> training sets overlap almost entirely. <InlineMath math="k = 5" /> or <InlineMath math="k = 10" /> balances bias and variance.</li>
        </ul>
        <BlockMath math="\text{Var}[\text{CV}] \approx \frac{1}{k}\text{Var}[L] + \frac{k-1}{k}\text{Cov}[L_i, L_j]" />

        <h3>Nested Cross-Validation</h3>
        <p>Outer loop for evaluation, inner loop for model selection:</p>
        <BlockMath math="\text{Score}_{\text{outer}} = \frac{1}{k_{\text{outer}}}\sum_{i=1}^{k_{\text{outer}}} L\left(f^*_{D \setminus D_i}, D_i\right)" />
        <p>where <InlineMath math="f^*_{D \setminus D_i}" /> is the best model found via inner CV on <InlineMath math="D \setminus D_i" />. This gives an <strong>unbiased estimate</strong> of the generalization error of the entire model-selection procedure.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Standard k-Fold and Stratified k-Fold</h3>
        <CodeBlock
          language="python"
          title="cross_validation_basics.py"
          code={`import numpy as np
from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold, RepeatedStratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    weights=[0.7, 0.3],  # imbalanced
    random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Basic k-fold (doesn't preserve class balance!)
kf_scores = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=42),
                            scoring="f1")
print(f"KFold F1: {kf_scores.mean():.4f} (+/- {kf_scores.std():.4f})")

# Stratified k-fold (preserves class proportions in each fold)
skf_scores = cross_val_score(model, X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=42),
                             scoring="f1")
print(f"Stratified F1: {skf_scores.mean():.4f} (+/- {skf_scores.std():.4f})")

# Repeated stratified k-fold (most robust estimate)
rskf_scores = cross_val_score(
    model, X, y,
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42),
    scoring="f1"
)
print(f"Repeated Stratified F1: {rskf_scores.mean():.4f} (+/- {rskf_scores.std():.4f})")`}
        />

        <h3>Time Series Cross-Validation</h3>
        <CodeBlock
          language="python"
          title="time_series_cv.py"
          code={`from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
import numpy as np

# Simulated time series: never train on future data!
np.random.seed(42)
n = 500
X = np.random.randn(n, 5)
y = np.cumsum(np.random.randn(n))  # random walk

model = Ridge(alpha=1.0)
tscv = TimeSeriesSplit(n_splits=5)

for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
    print(f"Fold {i+1}: train [{train_idx[0]}-{train_idx[-1]}], "
          f"test [{test_idx[0]}-{test_idx[-1]}], R² = {score:.4f}")
# Each fold uses a growing training window with a forward test window`}
        />

        <h3>Nested Cross-Validation</h3>
        <CodeBlock
          language="python"
          title="nested_cv.py"
          code={`from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=20, random_state=42)

# Inner CV: tune hyperparameters
pipe = make_pipeline(StandardScaler(), SVC())
param_grid = {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1, 1]}
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="accuracy")

# Outer CV: unbiased performance estimate
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring="accuracy")

print(f"Nested CV accuracy: {nested_scores.mean():.4f} (+/- {nested_scores.std():.4f}")

# Compare to non-nested (optimistic!)
grid_search.fit(X, y)
print(f"Non-nested best CV accuracy: {grid_search.best_score_:.4f}")
# Non-nested is always >= nested (selection bias)`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>k = 5 or k = 10 is standard</strong>: Good balance of bias and variance for the CV estimator. LOOCV is rarely worth the compute unless your dataset is tiny (&lt; 100 samples).</li>
          <li><strong>Always shuffle before splitting</strong>: Unless your data has temporal ordering. Set <code>shuffle=True</code> and a <code>random_state</code> for reproducibility.</li>
          <li><strong>Stratification is non-negotiable for classification</strong>: Especially with imbalanced classes. A fold with no positive examples is useless.</li>
          <li><strong>Use nested CV for reporting final performance</strong>: If you tune hyperparameters with CV and then report the same CV score, you&apos;re overfitting to the validation set. Nested CV fixes this.</li>
          <li><strong>Time series data requires expanding or sliding windows</strong>: Never use random k-fold on temporal data — it leaks future information into training.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Feature engineering before cross-validation</strong>: If you fit a scaler, select features, or compute statistics on the <em>entire</em> dataset before splitting, you leak information. All preprocessing must happen <em>inside</em> each fold (use <code>Pipeline</code>).</li>
          <li><strong>Reporting best fold score instead of mean</strong>: The mean across folds is your estimate, not the best single fold.</li>
          <li><strong>Using CV score for both tuning and evaluation</strong>: This produces optimistic estimates. Use nested CV or a completely held-out test set.</li>
          <li><strong>Random k-fold on time series</strong>: Training on 2023 data and testing on 2021 data is data leakage. Always use time-aware splits.</li>
          <li><strong>Using too many folds on small datasets</strong>: LOOCV on 50 samples has very high variance. Consider repeated stratified k-fold instead.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You have a dataset with 10,000 samples, an imbalanced target (5% positive), and several hyperparameters to tune. Design a complete evaluation strategy.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Hold out a final test set</strong> (20%, stratified) that is never touched during development.</li>
          <li><strong>Nested CV on the remaining 80%</strong>:
            <ul>
              <li><strong>Outer loop</strong>: 5-fold stratified CV for unbiased performance estimation.</li>
              <li><strong>Inner loop</strong>: 5-fold stratified CV for hyperparameter tuning (e.g., via <code>GridSearchCV</code> or <code>RandomizedSearchCV</code>).</li>
            </ul>
          </li>
          <li><strong>Use appropriate metrics</strong>: Not accuracy (95% by always predicting negative). Use F1, PR-AUC, or a custom metric aligned with the business cost.</li>
          <li><strong>Report confidence intervals</strong>: From the outer CV folds, compute mean +/- 2 standard deviations.</li>
          <li><strong>Final sanity check</strong>: Retrain on all development data with the best hyperparameters, evaluate once on the held-out test set. This is your reported number.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Hastie et al. (ESL) Chapter 7</strong> — Model Assessment and Selection. The definitive treatment of CV, AIC, BIC.</li>
          <li><strong>Varma &amp; Simon (2006) &quot;Bias in error estimation when using cross-validation for model selection&quot;</strong> — Why nested CV is necessary.</li>
          <li><strong>Arlot &amp; Celisse (2010) &quot;A survey of cross-validation procedures for model selection&quot;</strong> — Comprehensive review of CV variants.</li>
          <li><strong>scikit-learn Cross-validation guide</strong> — All splitter classes, cross_val_score, cross_val_predict.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
