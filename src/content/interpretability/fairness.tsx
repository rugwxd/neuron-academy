"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Fairness() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A hiring model that rejects more women than men — even when they are equally qualified — is
          not just a PR problem, it is a legal and ethical failure. <strong>Fairness in machine learning</strong> asks
          a deceptively simple question: does this model treat different groups equitably?
        </p>
        <p>
          The challenge is that &quot;fair&quot; has many rigorous definitions, and they often
          <strong> conflict with each other</strong>. A model can satisfy demographic parity (equal acceptance
          rates across groups) while violating equalized odds (equal error rates across groups). Impossibility
          theorems prove that you cannot satisfy all fairness criteria simultaneously unless the model is
          perfect or the groups have identical base rates.
        </p>
        <p>
          There are two broad perspectives. <strong>Group fairness</strong> requires that statistical measures
          (acceptance rates, false positive rates) are equal across protected groups defined by attributes like
          race, gender, or age. <strong>Individual fairness</strong> requires that similar individuals receive
          similar predictions — &quot;treating like cases alike.&quot; Responsible AI means understanding these
          tradeoffs and making deliberate, documented choices about which criteria matter most for a given
          application.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Group Fairness Metrics</h3>
        <p>
          Let <InlineMath math="A \in \{0, 1\}" /> be the protected attribute (e.g., gender),
          <InlineMath math="Y \in \{0, 1\}" /> be the true label, and <InlineMath math="\hat{Y} \in \{0, 1\}" /> be
          the model&apos;s prediction.
        </p>

        <p><strong>1. Demographic Parity (Statistical Parity)</strong></p>
        <p>
          The model&apos;s positive prediction rate should be equal across groups:
        </p>
        <BlockMath math="P(\hat{Y} = 1 \mid A = 0) = P(\hat{Y} = 1 \mid A = 1)" />
        <p>
          This criterion ignores the true label entirely. A random coin flip satisfies demographic parity
          but is useless. It is most appropriate when historical labels may themselves be biased.
        </p>

        <p><strong>2. Equalized Odds</strong></p>
        <p>
          The model should have equal true positive rates <em>and</em> equal false positive rates across groups:
        </p>
        <BlockMath math="P(\hat{Y} = 1 \mid Y = y, A = 0) = P(\hat{Y} = 1 \mid Y = y, A = 1) \quad \forall\, y \in \{0, 1\}" />
        <p>
          This means the model is equally accurate for both groups conditional on the true outcome. It allows
          different base rates but demands equal error rates.
        </p>

        <p><strong>3. Equal Opportunity</strong></p>
        <p>
          A relaxation of equalized odds that only requires equal true positive rates:
        </p>
        <BlockMath math="P(\hat{Y} = 1 \mid Y = 1, A = 0) = P(\hat{Y} = 1 \mid Y = 1, A = 1)" />
        <p>
          This ensures that qualified individuals in both groups have the same chance of receiving a
          positive prediction.
        </p>

        <p><strong>4. Calibration (Sufficiency)</strong></p>
        <p>
          Among individuals who receive the same predicted score <InlineMath math="s" />, the actual positive
          rate should be the same across groups:
        </p>
        <BlockMath math="P(Y = 1 \mid \hat{S} = s, A = 0) = P(Y = 1 \mid \hat{S} = s, A = 1) \quad \forall\, s" />
        <p>
          A score of 0.8 should mean an 80% chance of being positive regardless of group membership.
        </p>

        <h3>Disparate Impact (80% Rule)</h3>
        <p>
          The disparate impact ratio is a legal standard from employment law. A selection process has
          disparate impact if the selection rate for the disadvantaged group is less than 80% of the
          advantaged group&apos;s rate:
        </p>
        <BlockMath math="\text{DI} = \frac{P(\hat{Y} = 1 \mid A = 0)}{P(\hat{Y} = 1 \mid A = 1)} \geq 0.8" />
        <p>
          Values below 0.8 are considered evidence of discrimination under U.S. employment law (EEOC guidelines).
        </p>

        <h3>Impossibility Theorem (Chouldechova, 2017)</h3>
        <p>
          When the base rates differ between groups (<InlineMath math="P(Y=1|A=0) \neq P(Y=1|A=1)" />),
          it is <strong>mathematically impossible</strong> to simultaneously satisfy calibration, equal false
          positive rates, and equal false negative rates — unless the classifier is perfect. This means
          fairness always involves tradeoffs.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Computing Fairness Metrics from Scratch</h3>
        <CodeBlock
          language="python"
          title="fairness_metrics.py"
          code={`import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def compute_fairness_metrics(y_true, y_pred, protected_attr):
    """
    Compute comprehensive fairness metrics for binary classification.

    Args:
        y_true: array of true labels (0/1)
        y_pred: array of predicted labels (0/1)
        protected_attr: array of group membership (0/1)

    Returns:
        dict of fairness metrics
    """
    results = {}
    groups = np.unique(protected_attr)

    group_metrics = {}
    for g in groups:
        mask = protected_attr == g
        y_t, y_p = y_true[mask], y_pred[mask]
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()

        group_metrics[g] = {
            "selection_rate": y_p.mean(),
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0,  # recall
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "base_rate": y_t.mean(),
            "count": len(y_t),
        }

    g0, g1 = groups[0], groups[1]
    m0, m1 = group_metrics[g0], group_metrics[g1]

    # Demographic Parity: equal selection rates
    results["demographic_parity_diff"] = abs(
        m0["selection_rate"] - m1["selection_rate"]
    )

    # Disparate Impact (80% rule)
    if m1["selection_rate"] > 0:
        results["disparate_impact"] = (
            m0["selection_rate"] / m1["selection_rate"]
        )
    else:
        results["disparate_impact"] = float("inf")

    # Equalized Odds: equal TPR and FPR
    results["equal_opportunity_diff"] = abs(m0["tpr"] - m1["tpr"])
    results["fpr_diff"] = abs(m0["fpr"] - m1["fpr"])
    results["equalized_odds_diff"] = max(
        results["equal_opportunity_diff"],
        results["fpr_diff"]
    )

    # Print summary
    print("=" * 60)
    print("FAIRNESS AUDIT REPORT")
    print("=" * 60)
    for g in groups:
        m = group_metrics[g]
        print(f"\\nGroup {g} (n={m['count']}):")
        print(f"  Base rate:      {m['base_rate']:.3f}")
        print(f"  Selection rate: {m['selection_rate']:.3f}")
        print(f"  TPR (recall):   {m['tpr']:.3f}")
        print(f"  FPR:            {m['fpr']:.3f}")
        print(f"  Precision:      {m['precision']:.3f}")

    print(f"\\n--- Fairness Metrics ---")
    print(f"Demographic Parity Diff:  {results['demographic_parity_diff']:.3f}")
    print(f"Disparate Impact Ratio:   {results['disparate_impact']:.3f}")
    di_ok = "PASS" if results["disparate_impact"] >= 0.8 else "FAIL"
    print(f"  80% Rule:               {di_ok}")
    print(f"Equal Opportunity Diff:   {results['equal_opportunity_diff']:.3f}")
    print(f"Equalized Odds Diff:      {results['equalized_odds_diff']:.3f}")

    return results, group_metrics

# --- Example usage ---
np.random.seed(42)
n = 2000
gender = np.random.binomial(1, 0.5, n)  # 0=female, 1=male
# Simulate biased labels: men have higher base rate
y_true = np.random.binomial(1, 0.4 + 0.15 * gender, n)
# Simulate a biased classifier
y_pred = np.random.binomial(1, 0.35 + 0.2 * gender, n)

results, gm = compute_fairness_metrics(y_true, y_pred, gender)`}
        />

        <h3>Detecting and Mitigating Bias with Fairlearn</h3>
        <CodeBlock
          language="python"
          title="fairlearn_mitigation.py"
          code={`import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer

# --- Generate synthetic data with a protected attribute ---
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=10,
    random_state=42
)
# Simulate a protected attribute correlated with features
protected = (X[:, 0] > 0).astype(int)

X_train, X_test, y_train, y_test, prot_train, prot_test = (
    train_test_split(X, y, protected, test_size=0.3, random_state=42)
)

# --- Train unconstrained model ---
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred_unconstrained = model.predict(X_test)

# --- Audit fairness ---
metrics = {
    "accuracy": accuracy_score,
    "selection_rate": selection_rate,
    "tpr": true_positive_rate,
    "fpr": false_positive_rate,
}

mf = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred_unconstrained,
    sensitive_features=prot_test,
)

print("=== Unconstrained Model ===")
print(mf.by_group.to_string())
print(f"\\nDemographic Parity Diff: "
      f"{demographic_parity_difference(y_test, y_pred_unconstrained, sensitive_features=prot_test):.3f}")
print(f"Equalized Odds Diff: "
      f"{equalized_odds_difference(y_test, y_pred_unconstrained, sensitive_features=prot_test):.3f}")

# --- Post-processing: ThresholdOptimizer for Equalized Odds ---
postprocessor = ThresholdOptimizer(
    estimator=model,
    constraints="equalized_odds",
    objective="accuracy_score",
    prefit=True,
)
postprocessor.fit(X_train, y_train, sensitive_features=prot_train)
y_pred_fair = postprocessor.predict(X_test, sensitive_features=prot_test)

mf_fair = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred_fair,
    sensitive_features=prot_test,
)

print("\\n=== After ThresholdOptimizer (Equalized Odds) ===")
print(mf_fair.by_group.to_string())
print(f"\\nDemographic Parity Diff: "
      f"{demographic_parity_difference(y_test, y_pred_fair, sensitive_features=prot_test):.3f}")
print(f"Equalized Odds Diff: "
      f"{equalized_odds_difference(y_test, y_pred_fair, sensitive_features=prot_test):.3f}")

# --- Accuracy tradeoff ---
acc_before = accuracy_score(y_test, y_pred_unconstrained)
acc_after = accuracy_score(y_test, y_pred_fair)
print(f"\\nAccuracy tradeoff: {acc_before:.3f} -> {acc_after:.3f} "
      f"(delta: {acc_after - acc_before:+.3f})")`}
        />

        <h3>Threshold Adjustment for Equalized Odds from Scratch</h3>
        <CodeBlock
          language="python"
          title="threshold_adjustment.py"
          code={`import numpy as np
from sklearn.metrics import roc_curve

def equalize_odds_thresholds(y_true, y_scores, protected,
                             target_fpr=None):
    """
    Find group-specific thresholds to achieve equalized odds.

    The idea: instead of using a single threshold for all groups,
    use different thresholds per group so that TPR and FPR are
    equalized across groups.

    Args:
        y_true: true binary labels
        y_scores: model predicted probabilities
        protected: binary group membership
        target_fpr: desired FPR (if None, uses the overall ROC)

    Returns:
        dict mapping group -> optimal threshold
    """
    groups = np.unique(protected)
    group_thresholds = {}

    # Step 1: Compute ROC curves per group
    group_rocs = {}
    for g in groups:
        mask = protected == g
        fpr, tpr, thresholds = roc_curve(y_true[mask], y_scores[mask])
        group_rocs[g] = (fpr, tpr, thresholds)

    # Step 2: Choose target operating point
    if target_fpr is None:
        # Use the overall model's FPR at default threshold (0.5)
        overall_pred = (y_scores >= 0.5).astype(int)
        negatives = y_true == 0
        target_fpr = overall_pred[negatives].mean()
    print(f"Target FPR: {target_fpr:.3f}")

    # Step 3: For each group, find threshold closest to target FPR
    for g in groups:
        fpr, tpr, thresholds = group_rocs[g]
        # Find index closest to target FPR
        idx = np.argmin(np.abs(fpr - target_fpr))
        group_thresholds[g] = thresholds[idx]
        print(f"  Group {g}: threshold={thresholds[idx]:.3f}, "
              f"FPR={fpr[idx]:.3f}, TPR={tpr[idx]:.3f}")

    return group_thresholds

def predict_with_group_thresholds(y_scores, protected,
                                   group_thresholds):
    """Apply group-specific thresholds."""
    y_pred = np.zeros_like(y_scores, dtype=int)
    for g, thresh in group_thresholds.items():
        mask = protected == g
        y_pred[mask] = (y_scores[mask] >= thresh).astype(int)
    return y_pred

# --- Example ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 3000
X = np.random.randn(n, 10)
protected = (X[:, 0] > 0).astype(int)
# Biased outcome: group 1 has higher base rate
y = np.random.binomial(1, 0.3 + 0.2 * protected + 0.1 * X[:, 1])

X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(
    X, y, protected, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_tr, y_tr)
scores = clf.predict_proba(X_te)[:, 1]

# Standard single threshold
y_pred_std = (scores >= 0.5).astype(int)
print("=== Single Threshold (0.5) ===")
for g in [0, 1]:
    mask = p_te == g
    tpr = y_pred_std[mask & (y_te == 1)].mean()
    fpr = y_pred_std[mask & (y_te == 0)].mean()
    print(f"  Group {g}: TPR={tpr:.3f}, FPR={fpr:.3f}")

# Group-specific thresholds for equalized FPR
print("\\n=== Equalized Odds Thresholds ===")
thresholds = equalize_odds_thresholds(y_te, scores, p_te)
y_pred_fair = predict_with_group_thresholds(scores, p_te, thresholds)

for g in [0, 1]:
    mask = p_te == g
    tpr = y_pred_fair[mask & (y_te == 1)].mean()
    fpr = y_pred_fair[mask & (y_te == 0)].mean()
    print(f"  Group {g}: TPR={tpr:.3f}, FPR={fpr:.3f}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always start with a fairness audit</strong>: Before deploying any model that affects people, compute demographic parity, equalized odds, and calibration across all protected groups. This is not optional — it is increasingly a legal requirement (EU AI Act, NYC Local Law 144).</li>
          <li><strong>Choose the right fairness criterion for the domain</strong>: In lending, calibration matters most (a score of 0.8 should mean the same default probability for everyone). In hiring, equal opportunity is often prioritized (equal recall across groups). In criminal justice, equalized odds (equal error rates) is critical.</li>
          <li><strong>Post-processing is the simplest mitigation</strong>: Using group-specific thresholds (ThresholdOptimizer) requires no model retraining and provides the most transparent fairness-accuracy tradeoff. Start here before trying in-processing or pre-processing methods.</li>
          <li><strong>Document your tradeoffs</strong>: The impossibility theorem means you <em>will</em> sacrifice something. Document which fairness criteria you prioritize, why, and the quantitative tradeoff with accuracy. This is essential for regulatory compliance and stakeholder trust.</li>
          <li><strong>Intersectionality matters</strong>: A model can be fair with respect to gender and fair with respect to race individually, yet still discriminate against Black women specifically. Always check intersections of protected attributes.</li>
          <li><strong>Fairness is not a one-time check</strong>: Population demographics shift over time. Monitor fairness metrics in production continuously, not just at deployment.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Removing protected attributes and assuming fairness</strong>: This is called &quot;fairness through unawareness&quot; and it almost never works. Other features (zip code, name patterns, education history) are often proxies for protected attributes. The model can still discriminate without directly seeing the protected attribute.</li>
          <li><strong>Optimizing for one metric while ignoring others</strong>: Achieving perfect demographic parity can dramatically worsen calibration or accuracy for all groups. Always report multiple fairness metrics together and understand the tradeoffs.</li>
          <li><strong>Using biased historical labels as ground truth</strong>: If past hiring decisions were biased, training on them perpetuates bias. The &quot;true label&quot; in your dataset may itself be unfair. Consider whether demographic parity (which ignores labels) is more appropriate when labels are suspect.</li>
          <li><strong>Ignoring base rate differences</strong>: If 30% of group A and 50% of group B are truly positive, perfect demographic parity requires over-selecting from A or under-selecting from B — hurting overall accuracy. Understand the base rates before choosing fairness criteria.</li>
          <li><strong>Not accounting for sample size</strong>: Fairness metrics on small subgroups are noisy. A TPR difference of 0.05 between groups of 10,000 is meaningful; the same difference between groups of 50 may be pure noise. Always report confidence intervals.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Your credit scoring model has equal accuracy for men and women, but the false positive rate is 2x higher for women. How do you diagnose and fix this?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Diagnosis</strong>: A higher false positive rate for women means the model incorrectly approves more women who will default. This violates equalized odds. First, check the base rates — if women have a lower default rate, a single threshold naturally produces different FPRs. Examine the score distributions: if the distributions for men and women are shifted, a global threshold cuts them differently.</li>
          <li><strong>Root cause analysis</strong>: Check whether features like income, employment type, or zip code serve as proxies for gender. Compute SHAP values conditioned on gender to identify which features drive disparate predictions.</li>
          <li><strong>Mitigation options</strong>:
            <ul>
              <li><em>Post-processing</em>: Use group-specific classification thresholds to equalize FPR across genders. This is the simplest approach and preserves the model&apos;s ranking ability within each group.</li>
              <li><em>In-processing</em>: Retrain with a fairness constraint (e.g., adversarial debiasing or a regularization term penalizing FPR difference). This can find better accuracy-fairness tradeoffs than post-processing.</li>
              <li><em>Pre-processing</em>: Transform the feature space to remove correlation with gender (e.g., learning fair representations).</li>
            </ul>
          </li>
          <li><strong>Validation</strong>: After mitigation, verify that fixing FPR disparity did not create new disparities in TPR or calibration. Report the full fairness dashboard, not just the metric you optimized.</li>
          <li><strong>Regulatory note</strong>: In lending, the Equal Credit Opportunity Act (ECOA) prohibits discrimination. Document the entire analysis and remediation for compliance.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Chouldechova (2017) &quot;Fair prediction with disparate impact: A study of bias in recidivism prediction instruments&quot;</strong> — Proves the impossibility of simultaneously satisfying calibration and equal error rates across groups.</li>
          <li><strong>Hardt, Price, &amp; Srebro (2016) &quot;Equality of Opportunity in Supervised Learning&quot;</strong> — Introduces equalized odds and equal opportunity criteria with post-processing methods.</li>
          <li><strong>Barocas, Hardt, &amp; Narayanan &quot;Fairness and Machine Learning&quot; (free online textbook)</strong> — The definitive reference covering fairness definitions, impossibility results, and mitigation strategies.</li>
          <li><strong>Fairlearn documentation (fairlearn.org)</strong> — Microsoft&apos;s open-source toolkit for assessing and improving fairness, with extensive tutorials and API documentation.</li>
          <li><strong>Mehrabi et al. (2021) &quot;A Survey on Bias and Fairness in Machine Learning&quot;</strong> — Comprehensive survey covering sources of bias, fairness metrics, and debiasing techniques across the ML pipeline.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
