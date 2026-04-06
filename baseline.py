from environment import SmartIrrigationEnv
import numpy as np

def random_policy(env):
    state = env.reset()
    total_reward = 0
    total_water = 0
    done = False

    while not done:
        action = np.random.randint(0, 4)
        state, reward, done = env.step(action)
        total_reward += reward
        total_water += action * 5

    return total_reward, total_water


def rule_based_policy(env):
    state = env.reset()
    total_reward = 0
    total_water = 0
    done = False

    while not done:
        soil_moisture = state[0]

        if soil_moisture < 40:
            action = 3
        elif soil_moisture < 50:
            action = 2
        elif soil_moisture < 60:
            action = 1
        else:
            action = 0

        state, reward, done = env.step(action)
        total_reward += reward
        total_water += action * 5

    return total_reward, total_water


if __name__ == "__main__":
    env = SmartIrrigationEnv()

    r_reward, r_water = random_policy(env)
    rb_reward, rb_water = rule_based_policy(env)

    print("\n--- BASELINE RESULTS ---")
    print(f"Random Policy → Reward: {r_reward:.2f}, Water Used: {r_water}")
    print(f"Rule-Based Policy → Reward: {rb_reward:.2f}, Water Used: {rb_water}")