import torch
from environment import SmartIrrigationEnv
from dqn_agent import DQNAgent

env = SmartIrrigationEnv()

state_size = 5
action_size = 4

agent = DQNAgent(state_size, action_size)

# Load trained model
agent.q_network.load_state_dict(torch.load("dqn_irrigation_model.pth"))
agent.q_network.eval()

agent.epsilon = 0.0

state = env.reset()
done = False

while not done:
    action = agent.act(state)
    next_state, reward, done = env.step(action)

    print("Soil Moisture:", next_state[0])
    print("Action:", action)
    print("Reward:", reward)
    print("-----")

    state = next_state
