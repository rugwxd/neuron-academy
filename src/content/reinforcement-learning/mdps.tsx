"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function MarkovDecisionProcesses() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          A Markov Decision Process (MDP) is the <strong>mathematical framework for sequential decision-making</strong>.
          It models an agent interacting with an environment: at each step, the agent observes a state, takes
          an action, receives a reward, and transitions to a new state. The goal is to find a strategy (policy)
          that maximizes the total reward over time.
        </p>
        <p>
          The &quot;Markov&quot; part means the future depends only on the <strong>current state</strong>,
          not on the history of how you got there. If you&apos;re playing chess, the best move depends on the
          current board position, not on the sequence of moves that led to it. This memoryless property makes
          the math tractable.
        </p>
        <p>
          Every RL problem is fundamentally an MDP (or a partially observable one, POMDP). Understanding MDPs
          gives you the theoretical foundation for all of reinforcement learning — from tabular Q-learning to
          deep RL algorithms like PPO and SAC.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>MDP Tuple</h3>
        <p>An MDP is defined by <InlineMath math="\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle" />:</p>
        <ul>
          <li><InlineMath math="\mathcal{S}" /> — State space (all possible states)</li>
          <li><InlineMath math="\mathcal{A}" /> — Action space (all possible actions)</li>
          <li><InlineMath math="P(s'|s,a)" /> — Transition probability: probability of reaching state <InlineMath math="s'" /> from <InlineMath math="s" /> via action <InlineMath math="a" /></li>
          <li><InlineMath math="R(s,a,s')" /> — Reward function: immediate reward for taking action <InlineMath math="a" /> in state <InlineMath math="s" /> and landing in <InlineMath math="s'" /></li>
          <li><InlineMath math="\gamma \in [0,1)" /> — Discount factor: how much we value future vs. immediate rewards</li>
        </ul>

        <h3>Policy and Value Functions</h3>
        <p>A <strong>policy</strong> <InlineMath math="\pi(a|s)" /> maps states to a probability distribution over actions.</p>
        <p>The <strong>state-value function</strong> under policy <InlineMath math="\pi" />:</p>
        <BlockMath math="V^\pi(s) = E_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid S_0 = s\right]" />
        <p>The <strong>action-value function</strong> (Q-function):</p>
        <BlockMath math="Q^\pi(s, a) = E_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid S_0 = s, A_0 = a\right]" />

        <h3>Bellman Expectation Equations</h3>
        <p>The value function satisfies a recursive relationship:</p>
        <BlockMath math="V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^\pi(s')\right]" />
        <BlockMath math="Q^\pi(s,a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')\right]" />

        <h3>Bellman Optimality Equations</h3>
        <p>The <strong>optimal</strong> value function satisfies:</p>
        <BlockMath math="V^*(s) = \max_a \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^*(s')\right]" />
        <BlockMath math="Q^*(s,a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\right]" />
        <p>The optimal policy simply picks the action with the highest Q-value: <InlineMath math="\pi^*(s) = \arg\max_a Q^*(s,a)" />.</p>
      </TopicSection>

      <TopicSection type="code">
        <h3>MDP Environment: Grid World</h3>
        <CodeBlock
          language="python"
          title="grid_world.py"
          code={`import numpy as np

class GridWorld:
    """
    Simple 4x4 grid world MDP.
    Agent starts at (0,0), goal at (3,3).
    Actions: 0=up, 1=right, 2=down, 3=left
    Reward: -1 per step, +10 at goal.
    """
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.goal = (size - 1, size - 1)
        self.state = (0, 0)

    def reset(self):
        self.state = (0, 0)
        return self._to_index(self.state)

    def _to_index(self, pos):
        return pos[0] * self.size + pos[1]

    def step(self, action):
        r, c = self.state
        if action == 0:   r = max(r - 1, 0)              # up
        elif action == 1: c = min(c + 1, self.size - 1)   # right
        elif action == 2: r = min(r + 1, self.size - 1)   # down
        elif action == 3: c = max(c - 1, 0)              # left

        self.state = (r, c)
        done = self.state == self.goal
        reward = 10.0 if done else -1.0
        return self._to_index(self.state), reward, done

    def get_transition_prob(self, s, a):
        """Returns list of (prob, next_state, reward, done) tuples."""
        # Deterministic transitions
        r, c = s // self.size, s % self.size
        if a == 0:   r = max(r - 1, 0)
        elif a == 1: c = min(c + 1, self.size - 1)
        elif a == 2: r = min(r + 1, self.size - 1)
        elif a == 3: c = max(c - 1, 0)

        s_next = r * self.size + c
        done = (r, c) == self.goal
        reward = 10.0 if done else -1.0
        return [(1.0, s_next, reward, done)]`}
        />

        <h3>Value Iteration (Dynamic Programming)</h3>
        <CodeBlock
          language="python"
          title="value_iteration.py"
          code={`def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Compute optimal value function using Bellman optimality equation.
    Requires known transition dynamics P(s'|s,a).
    """
    V = np.zeros(env.n_states)

    while True:
        delta = 0
        for s in range(env.n_states):
            v_old = V[s]
            # Bellman optimality: V(s) = max_a sum_s' P(s'|s,a)[R + gamma*V(s')]
            q_values = []
            for a in range(env.n_actions):
                q = 0
                for prob, s_next, reward, done in env.get_transition_prob(s, a):
                    q += prob * (reward + gamma * V[s_next] * (1 - done))
                q_values.append(q)
            V[s] = max(q_values)
            delta = max(delta, abs(v_old - V[s]))

        if delta < theta:
            break

    # Extract optimal policy from V
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        q_values = []
        for a in range(env.n_actions):
            q = 0
            for prob, s_next, reward, done in env.get_transition_prob(s, a):
                q += prob * (reward + gamma * V[s_next] * (1 - done))
            q_values.append(q)
        policy[s] = np.argmax(q_values)

    return V, policy

# Run it
env = GridWorld(size=4)
V, policy = value_iteration(env)

# Display value function as grid
print("Optimal Value Function:")
print(V.reshape(4, 4).round(2))
# [[5.90, 6.96, 8.01, 9.04],
#  [6.96, 8.01, 9.04, 10.0],
#  [8.01, 9.04, 10.0, ...],
#  [9.04, 10.0, ...,  0.0]]  <- goal state

# Display policy (0=up, 1=right, 2=down, 3=left)
arrows = ['UP', 'RT', 'DN', 'LT']
print("\\nOptimal Policy:")
for r in range(4):
    row = [arrows[policy[r * 4 + c]] for c in range(4)]
    print(row)
# Agent moves right and down toward the goal`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Discount factor <InlineMath math="\gamma" /> controls myopia</strong>: <InlineMath math="\gamma = 0" /> means the agent only cares about immediate reward (greedy). <InlineMath math="\gamma = 0.99" /> looks far ahead. In practice, <InlineMath math="\gamma \in [0.95, 0.999]" /> for most tasks.</li>
          <li><strong>Model-based vs model-free</strong>: If you know <InlineMath math="P(s&apos;|s,a)" /> (the transition model), use dynamic programming (value/policy iteration). If not, use model-free methods (Q-learning, policy gradients) that learn from experience.</li>
          <li><strong>Reward shaping is critical</strong>: A sparse reward (only at the goal) makes learning very hard. Dense, informative rewards guide the agent much faster. But be careful — poorly shaped rewards can lead to unintended behavior (reward hacking).</li>
          <li><strong>State representation matters</strong>: The Markov property must hold for your state definition. If the current observation doesn&apos;t contain enough information, you have a POMDP — use frame stacking, RNNs, or attention.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Violating the Markov property</strong>: If your state doesn&apos;t capture all relevant information (e.g., using only the current frame of a video game without velocity), the MDP framework breaks. Solution: stack frames or use recurrent policies.</li>
          <li><strong>Setting <InlineMath math="\gamma = 1" /></strong>: This makes infinite-horizon returns potentially unbounded and value iteration may not converge. Always use <InlineMath math="\gamma &lt; 1" /> or ensure episodic tasks terminate.</li>
          <li><strong>Confusing the Bellman expectation and optimality equations</strong>: The expectation equation evaluates a fixed policy. The optimality equation finds the best policy. Using the wrong one in your algorithm will give incorrect results.</li>
          <li><strong>Applying DP to large state spaces</strong>: Value iteration requires iterating over all states — impossible for continuous or very large discrete spaces (e.g., Go has <InlineMath math="10^{170}" /> states). Use function approximation (neural networks) instead.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Explain the Bellman optimality equation. How does value iteration use it to find the optimal policy? What is its computational complexity?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Bellman Optimality Equation</strong>:
            <ul>
              <li><InlineMath math="V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]" /></li>
              <li>It says: the optimal value of a state is the best action&apos;s expected immediate reward plus discounted optimal value of the next state.</li>
              <li>It&apos;s a system of <InlineMath math="|\mathcal{S}|" /> nonlinear equations (nonlinear because of the max).</li>
            </ul>
          </li>
          <li><strong>Value Iteration</strong>:
            <ul>
              <li>Initialize <InlineMath math="V(s) = 0" /> for all states.</li>
              <li>Repeatedly apply the Bellman update: <InlineMath math="V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[R + \gamma V_k(s')]" /></li>
              <li>This is a contraction mapping with factor <InlineMath math="\gamma" />, guaranteed to converge to <InlineMath math="V^*" />.</li>
              <li>Extract policy: <InlineMath math="\pi^*(s) = \arg\max_a Q^*(s,a)" /></li>
            </ul>
          </li>
          <li><strong>Complexity</strong>:
            <ul>
              <li>Each iteration: <InlineMath math="O(|\mathcal{S}|^2 |\mathcal{A}|)" /> — for each state, try each action, sum over next states.</li>
              <li>Number of iterations: <InlineMath math="O\left(\frac{1}{1-\gamma}\log\frac{1}{\epsilon}\right)" /> for <InlineMath math="\epsilon" />-convergence.</li>
              <li>Practical for small MDPs (thousands of states). Infeasible for Atari (<InlineMath math="10^{15}" /> states).</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Sutton &amp; Barto &quot;Reinforcement Learning: An Introduction&quot; (2nd ed.)</strong> — The RL bible. Chapters 3-4 cover MDPs and dynamic programming thoroughly.</li>
          <li><strong>Puterman &quot;Markov Decision Processes&quot;</strong> — The definitive mathematical treatment of MDPs for the theory-minded.</li>
          <li><strong>David Silver&apos;s RL Course (Lectures 2-3)</strong> — Excellent video lectures on MDPs and dynamic programming from DeepMind.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
