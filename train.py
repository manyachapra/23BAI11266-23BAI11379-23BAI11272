from environment import SmartIrrigationEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt

env = SmartIrrigationEnv()

state_size = 5
action_size = 4

agent = DQNAgent(state_size, action_size)

episodes = 300
batch_size = 32

reward_history = []   # ⭐ Store rewards for plotting
water_usage_history = []

# ------------------ TRAINING ------------------
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)

        next_state, reward, done = env.step(action)

        agent.memory.add((state, action, reward, next_state, done))
        agent.learn(batch_size)

        state = next_state
        total_reward += reward

    if episode % 10 == 0:
        agent.update_target()

    reward_history.append(total_reward)   # ⭐ Save episode reward
    water_usage_history.append(env.total_water_used)

    print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

# ⭐ SAVE MODEL HERE
import torch
torch.save(agent.q_network.state_dict(), "dqn_irrigation_model.pth")
print("Model saved successfully!")

# ------------------ PLOT LEARNING CURVE ------------------
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Performance of DQN Irrigation Agent")
plt.show()

plt.figure()
plt.plot(water_usage_history)
plt.xlabel("Episode")
plt.ylabel("Water Used")
plt.title("Water Consumption Over Time")
plt.show()

# ------------------ TESTING ------------------
print("\n--- Testing Trained Agent ---\n")

agent.epsilon = 0.0   # Disable exploration

state = env.reset()
done = False
total_reward = 0
step_number = 1

while not done:
    action = agent.act(state)
    next_state, reward, done = env.step(action)

    print(f"Step {step_number}")
    print(f"Soil Moisture: {next_state[0]:.2f}")
    print(f"Action Taken: {action}")
    print(f"Reward: {reward:.2f}")
    print("------")

    state = next_state
    total_reward += reward
    step_number += 1

print("Final Total Reward:", total_reward)
