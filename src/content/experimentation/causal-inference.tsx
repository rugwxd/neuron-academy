"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function CausalInference() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Correlation is not causation — you&apos;ve heard it a thousand times. But <strong>causal
          inference</strong> gives you the tools to actually go from correlation to causation, even
          without running a randomized experiment. This matters enormously because many of the
          most important questions in data science <em>cannot</em> be answered with A/B tests:
          Did this policy change reduce churn? Does education increase earnings? Did the ad
          campaign cause the sales lift?
        </p>
        <p>
          The fundamental problem of causal inference is that you can never observe the
          <strong>counterfactual</strong> — what would have happened to the treated group if they
          had not been treated. You see someone who took a drug and recovered, but you can never
          see the same person at the same time not taking the drug. Every causal inference method
          is, at its core, a strategy for constructing a credible counterfactual.
        </p>
        <p>
          <strong>Directed Acyclic Graphs (DAGs)</strong> formalize causal relationships and tell
          you what to control for (and what NOT to control for — collider bias is a real trap).
          <strong>Propensity score methods</strong> create pseudo-randomized comparisons from
          observational data by matching treated and untreated units that had similar
          probabilities of treatment. <strong>Difference-in-differences (DiD)</strong> exploits
          the timing of an intervention by comparing the change in treated vs. untreated
          groups over time.
        </p>
        <p>
          These methods all require untestable assumptions. The power of causal inference lies
          not in eliminating assumptions, but in making them <em>explicit</em> so they can be
          scrutinized, debated, and stress-tested.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Potential Outcomes Framework (Rubin Causal Model)</h3>
        <p>
          For each unit <InlineMath math="i" />, define two potential outcomes:
        </p>
        <ul>
          <li><InlineMath math="Y_i(1)" /> — outcome if treated</li>
          <li><InlineMath math="Y_i(0)" /> — outcome if not treated (counterfactual)</li>
        </ul>
        <p>The <strong>individual treatment effect</strong> is <InlineMath math="\tau_i = Y_i(1) - Y_i(0)" />, but we only observe one of these.</p>
        <p>The <strong>Average Treatment Effect (ATE)</strong> is:</p>
        <BlockMath math="\tau_{\text{ATE}} = E[Y_i(1) - Y_i(0)] = E[Y_i(1)] - E[Y_i(0)]" />
        <p>The <strong>Average Treatment Effect on the Treated (ATT)</strong> is:</p>
        <BlockMath math="\tau_{\text{ATT}} = E[Y_i(1) - Y_i(0) \mid T_i = 1]" />

        <h3>Selection Bias</h3>
        <p>The naive comparison is biased when treatment is not random:</p>
        <BlockMath math="E[Y \mid T=1] - E[Y \mid T=0] = \underbrace{\tau_{\text{ATT}}}_{\text{causal effect}} + \underbrace{E[Y(0) \mid T=1] - E[Y(0) \mid T=0]}_{\text{selection bias}}" />
        <p>Selection bias is the difference in baseline outcomes between treated and untreated — the reason treated people would have had different outcomes <em>even without treatment</em>.</p>

        <h3>Directed Acyclic Graphs (DAGs)</h3>
        <p>A DAG encodes causal assumptions. Three fundamental structures:</p>
        <ul>
          <li><strong>Chain</strong>: <InlineMath math="X \to M \to Y" /> — M mediates the effect. Controlling for M blocks the causal path.</li>
          <li><strong>Fork (Confounder)</strong>: <InlineMath math="X \leftarrow C \to Y" /> — C confounds the X-Y relationship. <strong>Must control for C.</strong></li>
          <li><strong>Collider</strong>: <InlineMath math="X \to C \leftarrow Y" /> — C is a common effect. <strong>Must NOT control for C</strong> — doing so induces a spurious association between X and Y.</li>
        </ul>
        <p><strong>Backdoor criterion</strong>: To estimate the causal effect of X on Y, block all &quot;backdoor paths&quot; (non-causal paths from X to Y) by conditioning on the right set of variables.</p>

        <h3>Propensity Score</h3>
        <p>The propensity score is the probability of receiving treatment given covariates:</p>
        <BlockMath math="e(X_i) = P(T_i = 1 \mid X_i)" />
        <p><strong>Key theorem</strong> (Rosenbaum &amp; Rubin, 1983): If treatment assignment is <strong>strongly ignorable</strong> given <InlineMath math="X" />:</p>
        <BlockMath math="Y(0), Y(1) \perp\!\!\!\perp T \mid X \quad \implies \quad Y(0), Y(1) \perp\!\!\!\perp T \mid e(X)" />
        <p>Conditioning on the scalar propensity score is sufficient to remove confounding — you don&apos;t need to match on all covariates individually.</p>

        <h3>Difference-in-Differences (DiD)</h3>
        <p>
          Compare the change over time between a treated group and a control group:
        </p>
        <BlockMath math="\hat{\tau}_{\text{DiD}} = (\bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}}) - (\bar{Y}_{C,\text{post}} - \bar{Y}_{C,\text{pre}})" />
        <p>The key assumption is <strong>parallel trends</strong>: absent treatment, both groups would have followed the same trajectory. In regression form:</p>
        <BlockMath math="Y_{it} = \beta_0 + \beta_1 \cdot \text{Treat}_i + \beta_2 \cdot \text{Post}_t + \beta_3 \cdot (\text{Treat}_i \times \text{Post}_t) + \epsilon_{it}" />
        <p><InlineMath math="\beta_3" /> is the DiD estimator — the causal effect of the treatment.</p>
      </TopicSection>

      <TopicSection type="code">
        <CodeBlock
          language="python"
          title="propensity_score_matching.py"
          code={`import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats

np.random.seed(42)

# ============================================================
# Simulate observational data with confounding
# ============================================================
n = 2000

# Covariates
age = np.random.normal(40, 12, n)
income = np.random.normal(50000, 15000, n)

# Treatment assignment depends on covariates (confounding!)
# Older, wealthier people are more likely to get treated
logit = -3 + 0.03 * age + 0.00004 * income
prob_treat = 1 / (1 + np.exp(-logit))
treatment = np.random.binomial(1, prob_treat)

# Outcome depends on covariates AND treatment
# True treatment effect = 5.0
true_effect = 5.0
y = 20 + 0.3 * age + 0.0002 * income + true_effect * treatment + np.random.normal(0, 5, n)

df = pd.DataFrame({
    'age': age, 'income': income,
    'treatment': treatment, 'outcome': y
})

# Naive comparison (BIASED due to confounding)
naive_effect = (df[df.treatment == 1].outcome.mean() -
                df[df.treatment == 0].outcome.mean())
print(f"True effect:  {true_effect:.2f}")
print(f"Naive effect: {naive_effect:.2f} (biased!)")

# ============================================================
# Propensity Score Matching
# ============================================================

# Step 1: Estimate propensity scores
X = df[['age', 'income']].values
ps_model = LogisticRegression()
ps_model.fit(X, treatment)
df['propensity'] = ps_model.predict_proba(X)[:, 1]

# Step 2: Match treated to control using nearest-neighbor on PS
treated = df[df.treatment == 1]
control = df[df.treatment == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['propensity']].values)
distances, indices = nn.kneighbors(treated[['propensity']].values)

matched_control = control.iloc[indices.flatten()]

# Step 3: Compute ATT from matched pairs
att_psm = treated.outcome.values.mean() - matched_control.outcome.values.mean()
print(f"\\nPSM estimate: {att_psm:.2f}")

# Step 4: Check covariate balance after matching
print("\\nCovariate balance (standardized mean difference):")
for col in ['age', 'income']:
    smd_before = ((treated[col].mean() - control[col].mean()) /
                  np.sqrt((treated[col].var() + control[col].var()) / 2))
    smd_after = ((treated[col].mean() - matched_control[col].mean()) /
                 np.sqrt((treated[col].var() + matched_control[col].var()) / 2))
    print(f"  {col}: before={smd_before:.3f}, after={smd_after:.3f}")`}
        />

        <CodeBlock
          language="python"
          title="ipw_estimation.py"
          code={`import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

# (Using same df from above)
# ============================================================
# Inverse Propensity Weighting (IPW)
# ============================================================

# IPW estimates ATE by reweighting observations
# Weight for treated: 1/e(X), weight for control: 1/(1-e(X))

e = df['propensity'].values
t = df['treatment'].values
y = df['outcome'].values

# Horvitz-Thompson estimator
ate_ipw = (np.mean(t * y / e) - np.mean((1 - t) * y / (1 - e)))
print(f"IPW estimate (ATE): {ate_ipw:.2f}")

# Doubly robust estimator (combines regression + IPW)
# Robust to misspecification of either the outcome model OR propensity model
from sklearn.linear_model import LinearRegression

X = df[['age', 'income']].values

# Fit outcome model
outcome_model = LinearRegression()
outcome_model.fit(X[t == 0], y[t == 0])
mu0_hat = outcome_model.predict(X)

outcome_model.fit(X[t == 1], y[t == 1])
mu1_hat = outcome_model.predict(X)

# Doubly robust ATE
dr_ate = np.mean(
    t * (y - mu1_hat) / e + mu1_hat
) - np.mean(
    (1 - t) * (y - mu0_hat) / (1 - e) + mu0_hat
)
print(f"Doubly robust estimate (ATE): {dr_ate:.2f}")
print(f"True effect: 5.00")`}
        />

        <CodeBlock
          language="python"
          title="difference_in_differences.py"
          code={`import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)

# ============================================================
# Difference-in-Differences
# ============================================================
# Scenario: a city implements a new policy in Jan 2024.
# We have monthly data for treated city and control city.

n_months = 24  # 12 pre, 12 post
treatment_start = 12

# Treated city: baseline trend + treatment effect of 8 units
treated_pre = 50 + np.arange(12) * 0.5 + np.random.normal(0, 2, 12)
treated_post = 50 + np.arange(12, 24) * 0.5 + 8 + np.random.normal(0, 2, 12)

# Control city: same baseline trend, no treatment
control_pre = 45 + np.arange(12) * 0.5 + np.random.normal(0, 2, 12)
control_post = 45 + np.arange(12, 24) * 0.5 + np.random.normal(0, 2, 12)

# Build dataframe
df = pd.DataFrame({
    'outcome': np.concatenate([treated_pre, treated_post,
                                control_pre, control_post]),
    'treated': np.array([1]*24 + [0]*24),
    'post': np.array([0]*12 + [1]*12 + [0]*12 + [1]*12),
    'time': np.tile(np.arange(24), 2),
    'city': ['treated']*24 + ['control']*24
})
df['treat_x_post'] = df['treated'] * df['post']

# DiD regression
model = smf.ols('outcome ~ treated + post + treat_x_post + time', data=df).fit()
print("=== Difference-in-Differences ===")
print(f"DiD estimate (treat_x_post): {model.params['treat_x_post']:.3f}")
print(f"95% CI: ({model.conf_int().loc['treat_x_post'][0]:.3f}, "
      f"{model.conf_int().loc['treat_x_post'][1]:.3f})")
print(f"p-value: {model.pvalues['treat_x_post']:.4f}")
print(f"True effect: 8.00")

# Manual DiD calculation
did_manual = ((treated_post.mean() - treated_pre.mean()) -
              (control_post.mean() - control_pre.mean()))
print(f"\\nManual DiD: {did_manual:.3f}")

# Check parallel trends assumption (pre-treatment)
print("\\n--- Parallel Trends Check (pre-treatment slopes) ---")
from scipy import stats
slope_t, _, _, _, _ = stats.linregress(np.arange(12), treated_pre)
slope_c, _, _, _, _ = stats.linregress(np.arange(12), control_pre)
print(f"Treated pre-trend slope: {slope_t:.3f}")
print(f"Control pre-trend slope: {slope_c:.3f}")
print(f"Slopes similar? {'Yes' if abs(slope_t - slope_c) < 0.3 else 'No'}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Draw the DAG first</strong>: Before running any analysis, draw out your assumed causal structure. This forces you to be explicit about what you&apos;re assuming and helps identify what variables to control for. Use the <InlineMath math="\texttt{dowhy}" /> library in Python for formal causal analysis.</li>
          <li><strong>Propensity score overlap is critical</strong>: If treated and control groups have non-overlapping propensity scores, no method can rescue you — there are no comparable counterfactuals. Always plot the propensity score distributions and trim observations with extreme scores (below 0.05 or above 0.95).</li>
          <li><strong>DiD is everywhere in tech</strong>: Launched a feature in one market but not another? DiD. Rolled out a policy change on a specific date? DiD. It&apos;s the most practical causal inference tool for evaluating interventions that can&apos;t be A/B tested.</li>
          <li><strong>Sensitivity analysis is mandatory</strong>: All observational causal methods rely on the &quot;no unmeasured confounders&quot; assumption, which is untestable. Use Rosenbaum bounds or the E-value to quantify how strong unmeasured confounding would need to be to overturn your conclusion.</li>
          <li><strong>Doubly robust estimators</strong>: Combine propensity scores with outcome regression. You get a consistent estimate if <em>either</em> the propensity model or the outcome model is correct (but not necessarily both). This provides insurance against model misspecification.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Controlling for a collider</strong>: If you control for a variable that is a common <em>effect</em> of treatment and outcome, you introduce bias rather than removing it. Example: conditioning on &quot;got promoted&quot; when studying the effect of education on salary — promotion is affected by both, making it a collider.</li>
          <li><strong>Controlling for a mediator</strong>: If you want the total effect of X on Y and you control for a variable M on the causal path <InlineMath math="X \to M \to Y" />, you block the very effect you&apos;re trying to estimate. Only control for mediators if you want the <em>direct</em> effect.</li>
          <li><strong>Assuming parallel trends without checking</strong>: DiD is only valid if treated and control groups would have followed the same trajectory without treatment. Always plot pre-treatment trends for both groups. If they diverge before treatment, DiD is suspect.</li>
          <li><strong>Propensity score matching without balance checks</strong>: After matching, verify that covariates are balanced (standardized mean differences &lt; 0.1). Matching on the propensity score doesn&apos;t guarantee balance on individual covariates — it&apos;s an approximation.</li>
          <li><strong>Confusing prediction with causal inference</strong>: A model that predicts outcome well (high R-squared) does not mean its coefficients are causal. Prediction requires different variable selection than causal estimation. You can have a great prediction model with terrible causal estimates and vice versa.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Your company launches a premium feature. Users who adopt it have 40% higher retention. The PM concludes the feature <em>causes</em> better retention and wants to push all users to adopt. What&apos;s wrong with this reasoning, and how would you estimate the true causal effect?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>The problem: selection bias.</strong> Users who voluntarily adopt a premium feature are likely already more engaged, more tech-savvy, and more committed to the product. They would have had higher retention <em>even without</em> the feature. The 40% difference confounds the causal effect with selection.</li>
          <li><strong>Draw the DAG:</strong> User engagement <InlineMath math="\to" /> Feature adoption, User engagement <InlineMath math="\to" /> Retention, Feature adoption <InlineMath math="\to" /> Retention (?). Engagement is a confounder.</li>
          <li><strong>Ideal: A/B test.</strong> Randomly offer the feature to some users and withhold it from others. The intent-to-treat (ITT) effect controls for selection because randomization eliminates confounding.</li>
          <li><strong>If no A/B test is possible:</strong>
            <ul>
              <li><strong>Propensity score matching:</strong> Match adopters with non-adopters who had similar engagement levels, tenure, and demographics. The matched comparison estimates the ATT.</li>
              <li><strong>Instrumental variable:</strong> Find something that affects adoption but not retention directly (e.g., a random promotional email about the feature). Use 2SLS to estimate the causal effect.</li>
              <li><strong>DiD:</strong> If the feature launched at a specific time, compare the change in retention for adopters vs. non-adopters, before and after launch.</li>
            </ul>
          </li>
          <li><strong>Key insight:</strong> The true causal effect is almost certainly smaller than 40%. Selection bias inflates naive estimates of feature impact.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Cunningham (2021) &quot;Causal Inference: The Mixtape&quot;</strong> — Free online textbook covering DiD, RDD, IV, and synthetic control with code examples. Accessible and entertaining.</li>
          <li><strong>Hernan &amp; Robins (2020) &quot;Causal Inference: What If&quot;</strong> — Rigorous treatment from an epidemiology perspective. Available free online from Harvard.</li>
          <li><strong>Pearl (2009) &quot;Causality&quot;</strong> — The foundational work on causal DAGs, do-calculus, and structural causal models. Dense but revolutionary.</li>
          <li><strong>Angrist &amp; Pischke (2009) &quot;Mostly Harmless Econometrics&quot;</strong> — The econometrics Bible for practical causal inference: IV, DiD, RDD.</li>
          <li><strong>Microsoft DoWhy library</strong> — Python library for causal inference that integrates DAGs, identification, estimation, and refutation in a single pipeline.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
