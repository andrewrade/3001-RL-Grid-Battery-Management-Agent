from gym.envs.registration import register

register(
    id="gym_examples/PowerTrading-v0",
    entry_point="gym_examples.envs:PowerTradingEnv",
)
