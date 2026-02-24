"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function QLearning() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Q-learning is a <strong>model-free</strong> reinforcement learning algorithm that learns the optimal
          action-value function <InlineMath math="Q^*(s, a)" /> directly from experience, without needing to
          know the environment&apos;s transition probabilities. It&apos;s the simplest and most foundational
          RL algorithm — the one that everything else builds on.
        </p>
        <p>
          The idea is elegant: maintain a table of Q-values (one entry per state-action pair) and update them
          as you interact with the environment. After each step, adjust the Q-value toward the <strong>actual
          reward you received plus the best future value you could get</strong>. Over time, the table converges
          to the optimal Q-function, and the optimal policy is simply: pick the action with the highest Q-value.
        </p>
        <p>
          The key insight that makes Q-learning work is that it&apos;s <strong>off-policy</strong>: the update
          uses the maximum Q-value of the next state regardless of what action the agent actually takes next.
          This means the agent can explore (take random actions) while still learning the optimal policy.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Q-Learning Update Rule</h3>
        <p>After taking action <InlineMath math="a" /> in state <InlineMath math="s" />, receiving reward <InlineMath math="r" />, and arriving in state <InlineMath math="s'" />:</p>
        <BlockMath math="Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]" />
        <p>where:</p>
        <ul>
          <li><InlineMath math="\alpha \in (0, 1]" /> — learning rate (how fast we update)</li>
          <li><InlineMath math="\gamma \in [0, 1)" /> — discount factor (importance of future rewards)</li>
          <li><InlineMath math="r + \gamma \max_{a'} Q(s', a')" /> — the <strong>TD target</strong> (our updated estimate)</li>
          <li><InlineMath math="r + \gamma \max_{a'} Q(s', a') - Q(s, a)" /> — the <strong>TD error</strong> (surprise signal)</li>
        </ul>

        <h3>Epsilon-Greedy Exploration</h3>
        <p>To balance exploration and exploitation:</p>
        <BlockMath math="a = \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon \end{cases}" />
        <p>Typically <InlineMath math="\epsilon" /> starts high (e.g., 1.0) and decays over time (e.g., <InlineMath math="\epsilon_t = \max(0.01, \epsilon_0 \cdot 0.995^t)" />).</p>

        <h3>Convergence Guarantee</h3>
        <p>
          Q-learning converges to <InlineMath math="Q^*" /> under two conditions:
        </p>
        <ol>
          <li>Every state-action pair is visited infinitely often (ensured by exploration).</li>
          <li>The learning rate satisfies the Robbins-Monro conditions: <InlineMath math="\sum_t \alpha_t = \infty" /> and <InlineMath math="\sum_t \alpha_t^2 < \infty" />.</li>
        </ol>

        <h3>SARSA (On-Policy Alternative)</h3>
        <p>SARSA updates using the <em>actual</em> next action <InlineMath math="a'" /> (not the max):</p>
        <BlockMath math="Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]" />
        <p>
          SARSA is more conservative — it accounts for the exploration policy. Q-learning is more aggressive —
          it learns the optimal policy regardless of the behavior policy.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Tabular Q-Learning on Grid World</h3>
        <CodeBlock
          language="python"
          title="q_learning.py"
          code={`import numpy as np

class GridWorld:
    """4x4 grid: start (0,0), goal (3,3). -1 per step, +10 at goal."""
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, right, down, left
        self.goal = size * size - 1

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        r, c = self.state // self.size, self.state % self.size
        if action == 0:   r = max(r - 1, 0)              # up
        elif action == 1: c = min(c + 1, self.size - 1)   # right
        elif action == 2: r = min(r + 1, self.size - 1)   # down
        elif action == 3: c = max(c - 1, 0)              # left
        self.state = r * self.size + c
        done = (self.state == self.goal)
        reward = 10.0 if done else -1.0
        return self.state, reward, done

def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """Tabular Q-Learning with epsilon-greedy exploration."""
    Q = np.zeros((env.n_states, env.n_actions))
    epsilon = epsilon_start
    rewards_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(action)

            # Q-learning update: use max over next actions
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode

# Train
env = GridWorld(size=4)
Q, rewards = q_learning(env)

# Show learned Q-values and policy
policy = np.argmax(Q, axis=1)
arrows = ['UP', 'RT', 'DN', 'LT']
print("Learned Policy:")
for r in range(4):
    row = [arrows[policy[r * 4 + c]] for c in range(4)]
    print(row)

# Show convergence
print(f"\\nAvg reward (first 100 eps):  {np.mean(rewards[:100]):.2f}")
print(f"Avg reward (last 100 eps):   {np.mean(rewards[-100:]):.2f}")`}
        />

        <h3>SARSA Comparison</h3>
        <CodeBlock
          language="python"
          title="sarsa_vs_qlearning.py"
          code={`def sarsa(env, episodes=5000, alpha=0.1, gamma=0.99,
          epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """SARSA: On-policy TD control."""
    Q = np.zeros((env.n_states, env.n_actions))
    epsilon = epsilon_start

    def eps_greedy(state):
        if np.random.random() < epsilon:
            return np.random.randint(env.n_actions)
        return np.argmax(Q[state])

    rewards_per_episode = []
    for ep in range(episodes):
        state = env.reset()
        action = eps_greedy(state)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_action = eps_greedy(next_state)

            # SARSA update: use ACTUAL next action (not max)
            td_target = reward + gamma * Q[next_state, next_action] * (1 - done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            action = next_action
            total_reward += reward

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode

# Compare: Q-learning finds optimal policy, SARSA finds safe policy
# Near cliffs, SARSA avoids edges (accounts for random exploration)
# Q-learning walks right along the cliff (optimal if perfectly greedy)`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Tabular Q-learning only works for small state spaces</strong>: A 4x4 grid has 16 states. Atari has ~<InlineMath math="10^{15}" /> pixel states. For large/continuous spaces, use Deep Q-Networks (DQN) that approximate <InlineMath math="Q(s,a)" /> with a neural network.</li>
          <li><strong>Epsilon decay schedule matters</strong>: Decay too fast and the agent hasn&apos;t explored enough. Decay too slow and it wastes time on random actions. A common approach: linear decay over the first 50% of training, then hold constant.</li>
          <li><strong>Learning rate <InlineMath math="\alpha" /></strong>: Too high causes oscillation, too low causes slow convergence. <InlineMath math="\alpha = 0.1" /> is a reasonable default. For DQN, use Adam with <InlineMath math="10^{-4}" />.</li>
          <li><strong>Experience replay (for DQN)</strong>: Store transitions in a buffer and sample random minibatches for updates. This breaks correlation between consecutive samples and dramatically improves stability.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Not handling terminal states correctly</strong>: At terminal states, there is no future reward. The target should be just <InlineMath math="r" />, not <InlineMath math="r + \gamma \max Q(s&apos;, a&apos;)" />. Forgetting the <InlineMath math="(1 - \text{done})" /> mask is a very common bug.</li>
          <li><strong>Maximization bias</strong>: Q-learning overestimates Q-values because <InlineMath math="E[\max] \geq \max[E]" />. This compounds over time. Fix: use Double Q-learning, which uses one Q-function to select actions and another to evaluate them.</li>
          <li><strong>No exploration</strong>: Setting <InlineMath math="\epsilon = 0" /> from the start means the agent gets stuck in the first semi-reasonable policy it finds. Exploration is not optional — it&apos;s required for convergence.</li>
          <li><strong>Confusing Q-learning and SARSA</strong>: Q-learning uses <InlineMath math="\max_{a'} Q(s', a')" /> (off-policy). SARSA uses <InlineMath math="Q(s', a')" /> where <InlineMath math="a'" /> is the actual next action (on-policy). This distinction matters near dangerous states.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> What is the difference between on-policy and off-policy learning? Explain with SARSA vs. Q-learning. When does the difference matter?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>On-policy (SARSA)</strong>:
            <ul>
              <li>Learns the value of the <em>policy being followed</em> (including exploration).</li>
              <li>Update: <InlineMath math="Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s', a') - Q(s,a)]" /> where <InlineMath math="a'" /> is the action actually taken.</li>
              <li>Converges to the optimal <em>epsilon-greedy</em> policy (not the globally optimal policy).</li>
            </ul>
          </li>
          <li><strong>Off-policy (Q-learning)</strong>:
            <ul>
              <li>Learns the value of the <em>optimal policy</em> regardless of what the agent actually does.</li>
              <li>Update: <InlineMath math="Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s', a') - Q(s,a)]" /></li>
              <li>The behavior policy (epsilon-greedy) differs from the target policy (greedy).</li>
            </ul>
          </li>
          <li><strong>When the difference matters — the cliff walking problem</strong>:
            <ul>
              <li>A grid world with a cliff along one edge. Falling off gives -100 reward.</li>
              <li><strong>Q-learning</strong>: Learns the optimal shortest path right along the cliff edge. But during training with <InlineMath math="\epsilon" />-greedy, the agent occasionally steps off the cliff.</li>
              <li><strong>SARSA</strong>: Learns a safer path away from the cliff, because it accounts for the fact that the policy sometimes takes random actions.</li>
              <li>In safety-critical applications, SARSA&apos;s conservatism is actually preferable.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Sutton &amp; Barto Ch. 6</strong> — Temporal-Difference learning, SARSA, and Q-learning derivations.</li>
          <li><strong>Watkins &amp; Dayan (1992) &quot;Q-learning&quot;</strong> — The original convergence proof for tabular Q-learning.</li>
          <li><strong>Mnih et al. (2015) &quot;Human-level control through deep reinforcement learning&quot;</strong> — DQN: Q-learning with neural networks that mastered Atari games.</li>
          <li><strong>Van Hasselt et al. (2016) &quot;Deep Reinforcement Learning with Double Q-learning&quot;</strong> — Fixes overestimation bias in DQN.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
