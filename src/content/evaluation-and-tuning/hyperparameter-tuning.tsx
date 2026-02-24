"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function HyperparameterTuning() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Model <strong>parameters</strong> are learned from data (weights, splits). <strong>Hyperparameters</strong> are set <em>before</em> training and control <em>how</em> the model learns: the learning rate, number of trees, regularization strength, number of layers. The model can&apos;t optimize these itself — that&apos;s your job.
        </p>
        <p>
          The simplest approach is <strong>grid search</strong>: define a grid of all combinations and try each one. It&apos;s thorough but wasteful — if you have 5 hyperparameters with 4 values each, that&apos;s <InlineMath math="4^5 = 1024" /> experiments. <strong>Random search</strong> is surprisingly better: by sampling random combinations, you cover the same effective range with far fewer trials. This works because most hyperparameters don&apos;t matter equally — random search is more likely to explore the important dimensions thoroughly.
        </p>
        <p>
          <strong>Bayesian optimization</strong> is the smartest approach: it builds a probabilistic model of the objective function (e.g., a Gaussian process) and uses it to decide which hyperparameters to try next. It focuses on promising regions and avoids wasting evaluations on clearly bad configurations. Libraries like Optuna and scikit-optimize make this easy.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>The Hyperparameter Optimization Problem</h3>
        <BlockMath math="\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda} \in \Lambda} \frac{1}{k}\sum_{i=1}^{k} L\left(f_{\boldsymbol{\lambda}}^{(i)}, D_i^{\text{val}}\right)" />
        <p>where <InlineMath math="\boldsymbol{\lambda}" /> is the hyperparameter vector, <InlineMath math="f_{\boldsymbol{\lambda}}^{(i)}" /> is the model trained with those hyperparameters on training fold <InlineMath math="i" />, and <InlineMath math="D_i^{\text{val}}" /> is the validation fold.</p>

        <h3>Why Random Search Beats Grid Search</h3>
        <p>
          Consider two hyperparameters where only one matters. Grid search with 9 trials gives you 3 unique values for the important parameter. Random search with 9 trials gives you 9 unique values. The probability of finding a good value scales much better:
        </p>
        <BlockMath math="P(\text{finding top 5\% region}) = 1 - (1 - 0.05)^n" />
        <p>With <InlineMath math="n = 60" /> random trials, you have a 95% chance of sampling from the top 5% of any single hyperparameter (Bergstra &amp; Bengio, 2012).</p>

        <h3>Bayesian Optimization</h3>
        <p>Model the objective as a Gaussian process: <InlineMath math="f(\boldsymbol{\lambda}) \sim \mathcal{GP}(\mu, K)" /></p>
        <p>Select the next point by maximizing an <strong>acquisition function</strong>:</p>
        <BlockMath math="\boldsymbol{\lambda}_{\text{next}} = \arg\max_{\boldsymbol{\lambda}} \alpha(\boldsymbol{\lambda})" />
        <p><strong>Expected Improvement (EI)</strong>:</p>
        <BlockMath math="\text{EI}(\boldsymbol{\lambda}) = \mathbb{E}\left[\max(0, f^* - f(\boldsymbol{\lambda}))\right]" />
        <p>where <InlineMath math="f^*" /> is the best observed value. EI balances <strong>exploitation</strong> (try near the best known point) and <strong>exploration</strong> (try uncertain regions).</p>

        <h3>Tree-structured Parzen Estimator (TPE)</h3>
        <p>Used by Optuna. Instead of modeling <InlineMath math="P(y \mid \boldsymbol{\lambda})" />, model:</p>
        <BlockMath math="p(\boldsymbol{\lambda} \mid y) = \begin{cases} \ell(\boldsymbol{\lambda}) & \text{if } y < y^* \\ g(\boldsymbol{\lambda}) & \text{if } y \geq y^* \end{cases}" />
        <p>Maximize <InlineMath math="\ell(\boldsymbol{\lambda}) / g(\boldsymbol{\lambda})" /> — the ratio of &quot;good&quot; to &quot;bad&quot; density.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Grid Search</h3>
        <CodeBlock
          language="python"
          title="grid_search.py"
          code={`from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=2000, n_features=20, random_state=42)

model = GradientBoostingClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.8, 1.0],
}
# Total: 3 * 3 * 3 * 2 = 54 combinations x 5 folds = 270 fits

grid = GridSearchCV(
    model, param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")
print(f"Best F1: {grid.best_score_:.4f}")`}
        />

        <h3>Random Search</h3>
        <CodeBlock
          language="python"
          title="random_search.py"
          code={`from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    "n_estimators": randint(50, 500),        # discrete uniform [50, 500)
    "max_depth": randint(2, 10),             # discrete uniform [2, 10)
    "learning_rate": uniform(0.001, 0.3),    # continuous uniform [0.001, 0.301)
    "subsample": uniform(0.6, 0.4),          # continuous uniform [0.6, 1.0)
    "min_samples_leaf": randint(1, 20),
}

random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions,
    n_iter=100,  # 100 random combinations (vs 54 for grid above)
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="f1",
    n_jobs=-1,
    random_state=42,
)
random_search.fit(X, y)
print(f"Best params: {random_search.best_params_}")
print(f"Best F1: {random_search.best_score_:.4f}")`}
        />

        <h3>Bayesian Optimization with Optuna</h3>
        <CodeBlock
          language="python"
          title="bayesian_optuna.py"
          code={`import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }

    model = GradientBoostingClassifier(**params, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    return scores.mean()

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Optuna extras: visualize optimization history
# optuna.visualization.plot_optimization_history(study)
# optuna.visualization.plot_param_importances(study)`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Start with random search</strong>: It&apos;s simple, parallelizable, and surprisingly effective. Use it to narrow down the search space before switching to Bayesian optimization for fine-tuning.</li>
          <li><strong>Use log-uniform for learning rates and regularization</strong>: These parameters span orders of magnitude. Searching [0.001, 0.01, 0.1] uniformly wastes trials. Use <code>log=True</code> in Optuna or <code>loguniform</code> in scipy.</li>
          <li><strong>Early stopping in the inner loop</strong>: For boosting models, use early stopping to find the right n_estimators rather than tuning it directly. This saves enormous computation.</li>
          <li><strong>Don&apos;t tune too many hyperparameters at once</strong>: Focus on the 3-5 that matter most. For XGBoost: learning_rate, max_depth, n_estimators, subsample, colsample_bytree.</li>
          <li><strong>Optuna&apos;s pruning</strong>: Use <code>MedianPruner</code> to terminate unpromising trials early. This can cut total compute by 50% or more.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Overfitting the validation set</strong>: Running 10,000 trials means your validation score is optimistic. Use nested CV or a held-out test set for final evaluation.</li>
          <li><strong>Grid search on continuous parameters</strong>: A grid with learning_rate in [0.01, 0.1, 1.0] misses everything in between. Use random or Bayesian search for continuous hyperparameters.</li>
          <li><strong>Not setting a random seed</strong>: Without fixed seeds, your results aren&apos;t reproducible. Set random_state in both the model and the CV splitter.</li>
          <li><strong>Tuning on the test set</strong>: If you adjust hyperparameters based on test set performance, the test set is no longer an unbiased estimate. Only use the test set once, at the very end.</li>
          <li><strong>Ignoring runtime</strong>: A model with 0.1% better F1 but 10x slower inference may not be worth it in production. Include latency constraints in your search.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You have a limited compute budget of 50 model evaluations to tune 6 hyperparameters. How do you allocate those evaluations, and why would you choose random search over grid search?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Grid search is infeasible</strong>: With 6 hyperparameters and even just 3 values each, grid search needs <InlineMath math="3^6 = 729" /> evaluations — far over budget. With 50 evaluations, grid search can only test ~1.9 values per dimension (<InlineMath math="50^{1/6} \approx 1.9" />), which is useless.</li>
          <li><strong>Random search explores each dimension independently</strong>: 50 random trials give you 50 unique values for <em>each</em> hyperparameter. If only 2 of the 6 actually matter (the &quot;effective dimensionality&quot; is low), random search efficiently covers the important subspace.</li>
          <li><strong>Better approach — Bayesian optimization</strong>:
            <ul>
              <li>Use the first 10-15 evaluations as random initialization.</li>
              <li>Use the remaining 35-40 evaluations with TPE (Optuna) or GP-based Bayesian optimization.</li>
              <li>Enable pruning to terminate bad trials early, effectively getting more than 50 evaluations.</li>
            </ul>
          </li>
          <li><strong>Even better — use domain knowledge</strong>: Fix hyperparameters you know are unimportant (e.g., max_features for small feature counts). Reduce the effective search to 3-4 critical parameters, making 50 trials more than sufficient.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Bergstra &amp; Bengio (2012) &quot;Random Search for Hyper-Parameter Optimization&quot;</strong> — The paper that proved random search dominates grid search.</li>
          <li><strong>Snoek et al. (2012) &quot;Practical Bayesian Optimization of Machine Learning Algorithms&quot;</strong> — The foundational paper for BO in ML.</li>
          <li><strong>Optuna documentation</strong> — Modern, flexible hyperparameter framework with pruning, visualization, and distributed tuning.</li>
          <li><strong>Feurer &amp; Hutter (2019) &quot;Hyperparameter Optimization&quot;</strong> — Comprehensive survey chapter from the AutoML book.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
