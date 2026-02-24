"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function PolicyGradients() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Policy gradient methods take a fundamentally different approach to RL: instead of learning a value
          function and deriving a policy from it (like Q-learning), they <strong>directly optimize the policy
          itself</strong>. The policy is a neural network that takes a state and outputs a probability
          distribution over actions. We adjust the network&apos;s weights to make actions that led to high
          rewards more likely.
        </p>
        <p>
          The simplest version is <strong>REINFORCE</strong>: run an episode, collect all the rewards, then
          increase the probability of actions that led to high total reward and decrease the probability of
          actions that led to low reward. It&apos;s like a student who tries different study strategies for
          an exam — after seeing their score, they do more of what worked and less of what didn&apos;t.
        </p>
        <p>
          The problem with vanilla REINFORCE is <strong>high variance</strong> — rewards are noisy, so the
          gradient estimates are noisy. This is where the <strong>actor-critic</strong> architecture comes in:
          a &quot;critic&quot; network estimates the value function and is used to reduce variance. <strong>PPO
          (Proximal Policy Optimization)</strong> takes this further with a clipped objective that prevents
          destructively large policy updates, making training much more stable. PPO is the workhorse algorithm
          behind ChatGPT&apos;s RLHF training.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Policy Gradient Theorem</h3>
        <p>
          Let <InlineMath math="\pi_\theta(a|s)" /> be a parameterized policy. The objective is to maximize
          expected return:
        </p>
        <BlockMath math="J(\theta) = E_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]" />
        <p>The policy gradient theorem gives us the gradient:</p>
        <BlockMath math="\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]" />
        <p>where <InlineMath math="G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k" /> is the return from time <InlineMath math="t" />.</p>

        <h3>REINFORCE with Baseline</h3>
        <p>Subtracting a baseline <InlineMath math="b(s_t)" /> reduces variance without introducing bias:</p>
        <BlockMath math="\nabla_\theta J(\theta) = E\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]" />
        <p>
          The optimal baseline is approximately <InlineMath math="b(s_t) = V^\pi(s_t)" />, the value function.
          The quantity <InlineMath math="G_t - V(s_t)" /> is called the <strong>advantage</strong>:
        </p>
        <BlockMath math="A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)" />

        <h3>Actor-Critic</h3>
        <p>Instead of waiting until the end of an episode, use a <strong>learned value function</strong> (the critic) to estimate the advantage at each step:</p>
        <BlockMath math="\hat{A}_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)" />
        <p>This is the <strong>TD advantage</strong>. The actor (policy) and critic (value function) are trained simultaneously.</p>

        <h3>Generalized Advantage Estimation (GAE)</h3>
        <p>Interpolate between low-variance (1-step TD) and low-bias (Monte Carlo) estimates:</p>
        <BlockMath math="\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)" />
        <p>where <InlineMath math="\lambda \in [0, 1]" /> controls the bias-variance tradeoff. <InlineMath math="\lambda = 0" /> gives 1-step TD, <InlineMath math="\lambda = 1" /> gives Monte Carlo.</p>

        <h3>PPO Clipped Objective</h3>
        <p>Let <InlineMath math="r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}" /> be the probability ratio. PPO clips this ratio to prevent large updates:</p>
        <BlockMath math="\mathcal{L}^{\text{CLIP}} = E_t\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]" />
        <p>
          If the advantage is positive (good action), the ratio is clipped at <InlineMath math="1+\epsilon" /> — we
          don&apos;t let the policy become <em>too</em> much more likely. If negative (bad action), clipped
          at <InlineMath math="1-\epsilon" />. Typically <InlineMath math="\epsilon = 0.2" />.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>REINFORCE (Vanilla Policy Gradient)</h3>
        <CodeBlock
          language="python"
          title="reinforce.py"
          code={`import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

def reinforce(env_name='CartPole-v1', episodes=1000, gamma=0.99, lr=1e-3):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        # Collect trajectory
        done = False
        while not done:
            state_t = torch.FloatTensor(state)
            probs = policy(state_t)
            dist = Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize

        # Policy gradient update
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G  # negative because we maximize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1:4d} | Total Reward: {sum(rewards):.0f}")

    return policy

policy = reinforce()`}
        />

        <h3>PPO with Actor-Critic</h3>
        <CodeBlock
          language="python"
          title="ppo.py"
          code={`import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, action_dim)   # policy head
        self.critic = nn.Linear(hidden, 1)            # value head

    def forward(self, x):
        features = self.shared(x)
        return torch.softmax(self.actor(features), dim=-1), self.critic(features)

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    # Process in reverse
    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages

def ppo(env_name='CartPole-v1', total_steps=100000, gamma=0.99, lam=0.95,
        clip_eps=0.2, lr=3e-4, epochs_per_update=4, batch_size=64, steps_per_update=2048):

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    state, _ = env.reset()
    ep_reward = 0

    for step in range(0, total_steps, steps_per_update):
        # --- Collect rollout ---
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        for _ in range(steps_per_update):
            state_t = torch.FloatTensor(state)
            probs, value = model(state_t)
            dist = Categorical(probs)
            action = dist.sample()

            states.append(state_t)
            actions.append(action)
            log_probs.append(dist.log_prob(action).detach())
            values.append(value.item())

            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            dones.append(float(terminated or truncated))
            ep_reward += reward

            if terminated or truncated:
                state, _ = env.reset()
                ep_reward = 0

        # --- Compute advantages ---
        advantages = compute_gae(rewards, values, dones, gamma, lam)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = torch.stack(log_probs)
        states_t = torch.stack(states)
        actions_t = torch.stack(actions)

        # --- PPO update ---
        for _ in range(epochs_per_update):
            indices = np.random.permutation(steps_per_update)
            for start in range(0, steps_per_update, batch_size):
                idx = indices[start:start + batch_size]

                probs, vals = model(states_t[idx])
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions_t[idx])

                # Probability ratio
                ratio = torch.exp(new_log_probs - old_log_probs[idx])

                # Clipped surrogate objective
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = nn.functional.mse_loss(vals.squeeze(), returns[idx])

                # Entropy bonus (encourages exploration)
                entropy = dist.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        print(f"Step {step + steps_per_update:6d} | "
              f"Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")

    return model

model = ppo()`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>PPO is the default choice</strong>: It&apos;s used for robotics (OpenAI Five, humanoid control), game playing, and RLHF for LLMs. It&apos;s simpler than TRPO, more stable than vanilla policy gradients, and works well out of the box.</li>
          <li><strong>GAE <InlineMath math="\lambda" /> tuning</strong>: <InlineMath math="\lambda = 0.95" /> works well in most settings. Lower values (0.9) reduce variance but increase bias. Higher values (0.99) are closer to Monte Carlo.</li>
          <li><strong>Shared vs separate networks</strong>: Sharing early layers between actor and critic saves compute and can improve learning, but can also cause interference. Separate networks are safer for complex environments.</li>
          <li><strong>Entropy regularization</strong>: Add an entropy bonus to the loss to prevent premature convergence to a deterministic policy. Coefficient of 0.01 is a common starting point.</li>
          <li><strong>Gradient clipping</strong>: Clip gradient norms to 0.5 for stability. Policy gradient methods are particularly prone to gradient spikes.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Not normalizing advantages</strong>: Raw advantages can have wildly different scales across batches, making learning unstable. Always normalize to zero mean and unit variance within each batch.</li>
          <li><strong>Using old policy data for too many epochs</strong>: PPO assumes the data is from a &quot;close&quot; policy. Too many epochs of reuse make the data stale and the importance weights inaccurate. 3-10 epochs is typical.</li>
          <li><strong>Forgetting the stop-gradient on old log probs</strong>: The ratio <InlineMath math="r_t(\theta)" /> should use <em>detached</em> old log probabilities. Backpropagating through both numerator and denominator creates incorrect gradients.</li>
          <li><strong>Not using GAE</strong>: Vanilla REINFORCE has such high variance that it&apos;s nearly unusable for complex tasks. GAE is essential for practical policy gradient methods.</li>
          <li><strong>Ignoring the value function loss</strong>: If the critic is inaccurate, the advantage estimates are garbage and the policy gradient signal is meaningless. Make sure the critic converges.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Derive the REINFORCE gradient and explain why we need a baseline. Then explain how PPO improves on vanilla policy gradients.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>REINFORCE gradient derivation</strong>:
            <ul>
              <li>Objective: <InlineMath math="J(\theta) = E_{\tau \sim \pi_\theta}[R(\tau)]" /></li>
              <li>Expand: <InlineMath math="J(\theta) = \int \pi_\theta(\tau) R(\tau) d\tau" /></li>
              <li>Take gradient: <InlineMath math="\nabla J = \int \nabla \pi_\theta(\tau) R(\tau) d\tau" /></li>
              <li>Use the log-derivative trick: <InlineMath math="\nabla \pi_\theta = \pi_\theta \nabla \log \pi_\theta" /></li>
              <li>Result: <InlineMath math="\nabla J = E_{\tau}[\nabla \log \pi_\theta(\tau) \cdot R(\tau)]" /></li>
              <li>Since <InlineMath math="\log \pi_\theta(\tau) = \sum_t \log \pi_\theta(a_t|s_t)" />, this decomposes per timestep.</li>
            </ul>
          </li>
          <li><strong>Why we need a baseline</strong>:
            <ul>
              <li>REINFORCE has high variance because it multiplies the log prob by the <em>total</em> return, which is noisy.</li>
              <li>Subtracting a baseline <InlineMath math="b(s)" /> from the return does not change the expected gradient (it&apos;s zero in expectation: <InlineMath math="E[\nabla \log \pi \cdot b] = 0" />).</li>
              <li>But it reduces variance. The optimal baseline is approximately <InlineMath math="V^\pi(s)" />, giving us the advantage <InlineMath math="A = Q - V" />.</li>
            </ul>
          </li>
          <li><strong>How PPO improves this</strong>:
            <ul>
              <li><strong>Sample efficiency</strong>: PPO reuses collected data for multiple gradient steps (vs. REINFORCE which is single-use).</li>
              <li><strong>Stable updates</strong>: The clipped objective prevents catastrophically large policy changes that can collapse performance.</li>
              <li><strong>GAE</strong>: Replaces Monte Carlo returns with a bias-variance controlled advantage estimator.</li>
            </ul>
          </li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Schulman et al. (2017) &quot;Proximal Policy Optimization Algorithms&quot;</strong> — The PPO paper. Concise and practical.</li>
          <li><strong>Schulman et al. (2016) &quot;High-Dimensional Continuous Control Using Generalized Advantage Estimation&quot;</strong> — The GAE paper.</li>
          <li><strong>Sutton &amp; Barto Ch. 13</strong> — Policy gradient methods from first principles.</li>
          <li><strong>Spinning Up in Deep RL (OpenAI)</strong> — Excellent tutorial implementations of REINFORCE, VPG, PPO, and more.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
