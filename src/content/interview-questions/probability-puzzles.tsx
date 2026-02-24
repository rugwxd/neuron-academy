"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function ProbabilityPuzzles() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Probability puzzles are a staple of data science, quant, and ML engineering interviews. They test
          your ability to think rigorously about uncertainty, apply first principles (Bayes&apos; theorem,
          conditional probability, linearity of expectation), and avoid common cognitive traps. The problems
          below range from warm-ups to brain-benders, each with a complete solution.
        </p>
        <p>
          The key to solving these problems is not memorization — it&apos;s having a <strong>systematic
          approach</strong>. For any probability puzzle: (1) Define the sample space clearly. (2) Identify
          what you&apos;re conditioning on. (3) Check if Bayes&apos; theorem, linearity of expectation,
          or symmetry can simplify the problem. (4) Verify with a quick simulation if time allows.
        </p>
        <p>
          These 20+ problems cover the most frequently tested patterns: conditional probability, Bayes&apos;
          theorem, expected value, counting, geometric/binomial distributions, and classic paradoxes.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Puzzle 1: The Monty Hall Problem</h3>
        <p>
          You pick door 1. The host (who knows what&apos;s behind each door) opens door 3, showing a goat.
          Should you switch to door 2?
        </p>
        <p><strong>Solution:</strong> Yes — switching wins with probability <InlineMath math="2/3" />.</p>
        <BlockMath math="P(\text{car at 2} \mid \text{host opens 3}) = \frac{P(\text{host opens 3} \mid \text{car at 2}) \cdot P(\text{car at 2})}{P(\text{host opens 3})}" />
        <BlockMath math="= \frac{1 \cdot \frac{1}{3}}{\frac{1}{2}} = \frac{2}{3}" />
        <p>
          The host&apos;s action is <em>not random</em> — they always reveal a goat. This asymmetry is what
          makes switching advantageous.
        </p>

        <h3>Puzzle 2: Birthday Problem</h3>
        <p>How many people are needed so there&apos;s a 50% chance two share a birthday?</p>
        <p><strong>Solution:</strong> 23 people. Compute the complement (no shared birthdays):</p>
        <BlockMath math="P(\text{no match among } n) = \prod_{i=0}^{n-1} \frac{365-i}{365} = \frac{365!}{365^n (365-n)!}" />
        <BlockMath math="P(\text{no match among 23}) \approx 0.493 \quad \Rightarrow \quad P(\text{match}) \approx 0.507" />

        <h3>Puzzle 3: Unfair Coin to Fair Outcome</h3>
        <p>You have a biased coin with <InlineMath math="P(H) = p \neq 0.5" />. How do you generate a fair 50/50 outcome?</p>
        <p><strong>Solution (von Neumann&apos;s trick):</strong> Flip twice. HT → &quot;heads&quot;, TH → &quot;tails&quot;, HH or TT → repeat.</p>
        <BlockMath math="P(HT) = p(1-p) = P(TH) \quad \Rightarrow \quad \text{fair outcome}" />
        <p>Expected flips: <InlineMath math="\frac{2}{2p(1-p)}" /> (for <InlineMath math="p = 0.7" />, about 4.8 flips per outcome).</p>

        <h3>Puzzle 4: Two Children Problem</h3>
        <p>A family has two children. One of them is a boy. What is the probability both are boys?</p>
        <p><strong>Solution:</strong> Sample space given at least one boy: {"{BB, BG, GB}"}. So <InlineMath math="P(BB) = 1/3" />.</p>
        <p><strong>Subtle variant:</strong> &quot;I met one of their children and it was a boy&quot; → <InlineMath math="P(BB) = 1/2" />. The difference is <em>how</em> you learned the information.</p>

        <h3>Puzzle 5: Expected Value of Max of Two Dice</h3>
        <p>Roll two fair dice. What is <InlineMath math="E[\max(D_1, D_2)]" />?</p>
        <p><strong>Solution:</strong> Use the CDF approach:</p>
        <BlockMath math="P(\max \leq k) = P(D_1 \leq k) \cdot P(D_2 \leq k) = \left(\frac{k}{6}\right)^2" />
        <BlockMath math="P(\max = k) = \frac{k^2 - (k-1)^2}{36} = \frac{2k-1}{36}" />
        <BlockMath math="E[\max] = \sum_{k=1}^{6} k \cdot \frac{2k-1}{36} = \frac{1 + 6 + 15 + 28 + 45 + 66}{36} = \frac{161}{36} \approx 4.47" />

        <h3>Puzzle 6: Coupon Collector Problem</h3>
        <p>There are <InlineMath math="n" /> distinct coupons. How many do you need to buy (randomly) to collect all <InlineMath math="n" />?</p>
        <BlockMath math="E[T] = n \sum_{i=1}^{n} \frac{1}{i} = n \cdot H_n \approx n \ln n + 0.577n" />
        <p>For <InlineMath math="n = 50" /> stickers: <InlineMath math="E[T] \approx 50 \cdot \ln 50 + 29 \approx 225" /> purchases.</p>

        <h3>Puzzle 7: Conditional Probability — Disease Testing</h3>
        <p>
          A disease affects 1 in 10,000 people. A test has 99% sensitivity and 99% specificity.
          You test positive. What is the probability you have the disease?
        </p>
        <BlockMath math="P(D|+) = \frac{P(+|D) \cdot P(D)}{P(+|D) \cdot P(D) + P(+|\neg D) \cdot P(\neg D)}" />
        <BlockMath math="= \frac{0.99 \times 0.0001}{0.99 \times 0.0001 + 0.01 \times 0.9999} = \frac{0.000099}{0.010098} \approx 0.98\%" />
        <p>
          Despite a 99% accurate test, the probability of disease given a positive test is less than 1%.
          The low base rate dominates.
        </p>

        <h3>Puzzle 8: Geometric Distribution — First Success</h3>
        <p>A server succeeds with probability <InlineMath math="p = 0.3" /> on each attempt. What is the expected number of attempts until the first success?</p>
        <BlockMath math="E[X] = \frac{1}{p} = \frac{1}{0.3} \approx 3.33 \text{ attempts}" />

        <h3>Puzzle 9: Broken Stick Triangle</h3>
        <p>Break a stick at two uniformly random points. What is the probability the three pieces form a triangle?</p>
        <p><strong>Solution:</strong> The triangle inequality requires each piece to be less than half the stick length. The probability is <InlineMath math="1/4" />.</p>
        <BlockMath math="P(\text{triangle}) = P\left(X < \frac{1}{2}, Y - X < \frac{1}{2}, 1 - Y < \frac{1}{2}\right) = \frac{1}{4}" />

        <h3>Puzzle 10: Gambler&apos;s Ruin</h3>
        <p>
          You start with $<InlineMath math="k" /> and repeatedly bet $1 on a fair coin flip. You stop when you reach $<InlineMath math="N" /> or $0. What is the probability of reaching $<InlineMath math="N" />?
        </p>
        <BlockMath math="P(\text{reach } N \mid \text{start at } k) = \frac{k}{N}" />
        <p>Starting with $20, target $100: <InlineMath math="P = 20/100 = 20\%" />.</p>

        <h3>Puzzle 11: Expected Rolls to See All Faces of a Die</h3>
        <p>A special case of the coupon collector with <InlineMath math="n = 6" />:</p>
        <BlockMath math="E = 6\left(1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \frac{1}{5} + \frac{1}{6}\right) = 6 \times 2.45 = 14.7" />

        <h3>Puzzle 12: Matching Problem (Derangements)</h3>
        <p><InlineMath math="n" /> people put hats in a box, then each draws one randomly. Expected number of people who get their own hat?</p>
        <BlockMath math="E[\text{matches}] = \sum_{i=1}^{n} P(\text{person } i \text{ gets own hat}) = n \cdot \frac{1}{n} = 1" />
        <p>By linearity of expectation, the answer is always 1, regardless of <InlineMath math="n" />.</p>

        <h3>Puzzle 13: Envelope Paradox (Two Envelopes)</h3>
        <p>
          Two envelopes: one has <InlineMath math="x" />, the other has <InlineMath math="2x" />. You pick one and see amount <InlineMath math="a" />.
          Should you switch?
        </p>
        <p>
          <strong>The naive argument (wrong):</strong> The other envelope has <InlineMath math="a/2" /> or <InlineMath math="2a" /> with equal probability, so <InlineMath math="E = a/2 \cdot 0.5 + 2a \cdot 0.5 = 1.25a" />. Always switch.
        </p>
        <p>
          <strong>The resolution:</strong> The error is in assuming <InlineMath math="P(\text{other} = 2a) = P(\text{other} = a/2) = 0.5" /> regardless of <InlineMath math="a" />. Without a prior on <InlineMath math="x" />, the problem is ill-defined. With a proper prior, the symmetry breaks.
        </p>

        <h3>Puzzle 14: Random Walk Return Probability</h3>
        <p>Starting at origin, each step +1 or -1 with probability 1/2. In 1D, will you return to origin?</p>
        <p><strong>Answer:</strong> With probability 1 (almost surely). In 2D, also probability 1. In 3D and above, not guaranteed (<InlineMath math="P \approx 0.34" /> for 3D).</p>

        <h3>Puzzle 15: Secretary Problem / Optimal Stopping</h3>
        <p>
          Interview <InlineMath math="n" /> candidates sequentially. After each, hire or reject irrevocably. Maximize probability of hiring the best.
        </p>
        <p><strong>Solution:</strong> Reject the first <InlineMath math="n/e \approx 37\%" /> candidates, then hire the next one who is better than all seen so far.</p>
        <BlockMath math="P(\text{best}) \to \frac{1}{e} \approx 36.8\% \text{ as } n \to \infty" />

        <h3>Puzzle 16: Conditional Expectation — Waiting for a Bus</h3>
        <p>
          Buses arrive as a Poisson process with rate <InlineMath math="\lambda = 4" /> per hour. You arrive at a random time. What is your expected wait?
        </p>
        <BlockMath math="E[\text{wait}] = \frac{1}{\lambda} = \frac{1}{4} \text{ hour} = 15 \text{ minutes}" />
        <p>
          This is the memoryless property of the exponential distribution — your wait does not depend on when the last bus came.
        </p>

        <h3>Puzzle 17: Simpson&apos;s Paradox</h3>
        <p>
          Hospital A has higher survival rates for both mild and severe cases than Hospital B. Yet Hospital B has a higher <em>overall</em> survival rate. How?
        </p>
        <p>
          <strong>Answer:</strong> Hospital A treats more severe cases (a confounder). When you aggregate, the case mix reverses the comparison. Always condition on confounders.
        </p>

        <h3>Puzzle 18: Binomial — Quality Control</h3>
        <p>A factory has a 2% defect rate. In a batch of 100, what is the probability of 5 or more defects?</p>
        <BlockMath math="P(X \geq 5) = 1 - \sum_{k=0}^{4} \binom{100}{k} (0.02)^k (0.98)^{100-k} \approx 1 - 0.949 = 0.051" />
        <p>Or use the Poisson approximation: <InlineMath math="\lambda = np = 2" />, <InlineMath math="P(X \geq 5) \approx 0.053" />.</p>

        <h3>Puzzle 19: Prisoner&apos;s Hat Problem</h3>
        <p>
          100 prisoners in a line, each wearing a black or white hat. Starting from the back, each must guess their own hat color (can see all hats in front). The group can agree on a strategy beforehand. What is the maximum number they can guarantee to save?
        </p>
        <p>
          <strong>Solution:</strong> 99 guaranteed, 1 at 50/50. Strategy: the last prisoner counts the number of black hats in front. If odd, they say &quot;black&quot;; if even, &quot;white.&quot; Each subsequent prisoner can deduce their own hat color from the parity information and all guesses so far.
        </p>

        <h3>Puzzle 20: Balls and Bins</h3>
        <p>Throw <InlineMath math="m" /> balls into <InlineMath math="n" /> bins uniformly. Expected number of empty bins?</p>
        <BlockMath math="E[\text{empty bins}] = n \cdot \left(1 - \frac{1}{n}\right)^m \approx n \cdot e^{-m/n}" />
        <p>For <InlineMath math="m = n" />: <InlineMath math="E \approx n/e \approx 0.368n" /> bins remain empty.</p>

        <h3>Puzzle 21: St. Petersburg Paradox</h3>
        <p>
          A coin is flipped until heads. If heads on flip <InlineMath math="k" />, you win <InlineMath math="2^k" /> dollars. Expected winnings?
        </p>
        <BlockMath math="E = \sum_{k=1}^{\infty} \frac{1}{2^k} \cdot 2^k = \sum_{k=1}^{\infty} 1 = \infty" />
        <p>
          Yet no one would pay $1000 to play. Resolution: utility of money is concave (logarithmic). <InlineMath math="E[\log(\text{winnings})]" /> is finite.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Simulation Verification for Key Puzzles</h3>
        <CodeBlock
          language="python"
          title="probability_simulations.py"
          code={`import numpy as np
from collections import Counter

np.random.seed(42)
N_SIMS = 100_000

# ---- Monty Hall ----
def simulate_monty_hall(switch: bool, n_sims: int = N_SIMS) -> float:
    wins = 0
    for _ in range(n_sims):
        car = np.random.randint(3)    # Car behind door 0, 1, or 2
        choice = np.random.randint(3) # Player picks a door
        # Host opens a door that is not the car and not the player choice
        if switch:
            wins += (choice != car)   # Switching wins iff initial choice was wrong
        else:
            wins += (choice == car)
    return wins / n_sims

print(f"Monty Hall (stay):   {simulate_monty_hall(False):.4f}")  # ~0.333
print(f"Monty Hall (switch): {simulate_monty_hall(True):.4f}")   # ~0.667

# ---- Birthday Problem ----
def simulate_birthday(n_people: int, n_sims: int = N_SIMS) -> float:
    matches = 0
    for _ in range(n_sims):
        birthdays = np.random.randint(0, 365, size=n_people)
        if len(set(birthdays)) < n_people:
            matches += 1
    return matches / n_sims

print(f"Birthday (23 people): {simulate_birthday(23):.4f}")  # ~0.507

# ---- Coupon Collector (n=50) ----
def simulate_coupon_collector(n: int, n_sims: int = N_SIMS) -> float:
    totals = []
    for _ in range(n_sims):
        seen = set()
        count = 0
        while len(seen) < n:
            seen.add(np.random.randint(n))
            count += 1
        totals.append(count)
    return np.mean(totals)

print(f"Coupon collector (n=50): {simulate_coupon_collector(50):.1f}")  # ~225

# ---- Disease Test (Bayes) ----
def simulate_disease_test(prevalence=0.0001, sensitivity=0.99,
                          specificity=0.99, n_sims=1_000_000):
    has_disease = np.random.random(n_sims) < prevalence
    test_positive = np.where(
        has_disease,
        np.random.random(n_sims) < sensitivity,   # True positive
        np.random.random(n_sims) > specificity,    # False positive
    )
    # P(disease | positive)
    positives = test_positive.sum()
    true_positives = (has_disease & test_positive).sum()
    return true_positives / positives if positives > 0 else 0

print(f"P(disease|positive): {simulate_disease_test():.4f}")  # ~0.0098

# ---- Max of two dice ----
def simulate_max_dice(n_sims: int = N_SIMS) -> float:
    d1 = np.random.randint(1, 7, size=n_sims)
    d2 = np.random.randint(1, 7, size=n_sims)
    return np.mean(np.maximum(d1, d2))

print(f"E[max(D1, D2)]: {simulate_max_dice():.4f}")  # ~4.47

# ---- Secretary Problem ----
def simulate_secretary(n: int = 100, n_sims: int = N_SIMS) -> float:
    cutoff = int(n / np.e)  # Reject first n/e candidates
    wins = 0
    for _ in range(n_sims):
        candidates = np.random.permutation(n)  # Random quality order
        best_in_rejected = max(candidates[:cutoff]) if cutoff > 0 else -1
        for i in range(cutoff, n):
            if candidates[i] > best_in_rejected:
                if candidates[i] == n - 1:  # Is this the best overall?
                    wins += 1
                break
    return wins / n_sims

print(f"Secretary problem P(best): {simulate_secretary():.4f}")  # ~0.368`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always define the sample space first</strong>: Many errors come from an ambiguous or incomplete sample space. Write it down before computing.</li>
          <li><strong>Linearity of expectation is your best friend</strong>: It works even when random variables are dependent. Use it for any &quot;expected number of X&quot; problem (Puzzle 12, 20).</li>
          <li><strong>Bayes&apos; theorem — draw the tree</strong>: For conditional probability problems, a probability tree with branches for prior and likelihoods makes the calculation mechanical.</li>
          <li><strong>When in doubt, simulate</strong>: A 10-line Python simulation can verify (or disprove) your analytical answer in seconds. Interviewers are impressed when you offer to verify.</li>
          <li><strong>Check edge cases</strong>: Does your formula give sensible answers for <InlineMath math="n=1" />, <InlineMath math="p=0" />, or <InlineMath math="p=1" />?</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Confusing P(A|B) with P(B|A)</strong>: The prosecutor&apos;s fallacy. P(positive test | disease) is NOT P(disease | positive test). Always use Bayes&apos; theorem.</li>
          <li><strong>Forgetting base rates</strong>: In Puzzle 7, a 99%-accurate test has under 1% PPV when the condition is rare. Base rate neglect is the most common cognitive bias in probability.</li>
          <li><strong>Overcounting with combinatorics</strong>: If order does not matter, use combinations, not permutations. If objects are indistinguishable, use stars and bars.</li>
          <li><strong>Assuming independence</strong>: &quot;At least one child is a boy&quot; and &quot;the first child is a boy&quot; lead to different answers because they condition on different events.</li>
          <li><strong>Not using complement counting</strong>: P(at least one) = 1 - P(none) is almost always easier than summing P(exactly 1) + P(exactly 2) + ... (Puzzle 2).</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Bonus Question:</strong> You flip a fair coin repeatedly. What is the expected number of flips to get two heads in a row?</p>
        <p><strong>Solution:</strong> Let <InlineMath math="E" /> be the expected flips from the start, and <InlineMath math="E_H" /> be the expected flips given the last flip was H.</p>
        <BlockMath math="E = \frac{1}{2}(1 + E_H) + \frac{1}{2}(1 + E)" />
        <BlockMath math="E_H = \frac{1}{2}(1 + 0) + \frac{1}{2}(1 + E) = 1 + \frac{E}{2}" />
        <p>Substituting:</p>
        <BlockMath math="E = 1 + \frac{1}{2}E_H + \frac{1}{2}E = 1 + \frac{1}{2}\left(1 + \frac{E}{2}\right) + \frac{E}{2}" />
        <BlockMath math="E = 1 + \frac{1}{2} + \frac{E}{4} + \frac{E}{2} = \frac{3}{2} + \frac{3E}{4}" />
        <BlockMath math="\frac{E}{4} = \frac{3}{2} \quad \Rightarrow \quad E = 6" />
        <p>The expected number of flips to see HH is <strong>6</strong>.</p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>&quot;Fifty Challenging Problems in Probability&quot; by Mosteller</strong> — Classic collection of elegant probability puzzles with solutions.</li>
          <li><strong>&quot;Introduction to Probability&quot; by Blitzstein &amp; Hwang (Harvard Stat 110)</strong> — Free textbook and lectures, outstanding for building probability intuition.</li>
          <li><strong>3Blue1Brown &quot;Bayes theorem&quot; video</strong> — Best visual explanation of Bayes&apos; theorem on the internet.</li>
          <li><strong>&quot;Heard on the Street&quot; by Timothy Crack</strong> — Quant interview probability and brainteaser problems.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
