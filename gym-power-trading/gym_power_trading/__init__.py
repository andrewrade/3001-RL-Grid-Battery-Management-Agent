from gym.envs.registration import register

register(
    id="gym_power_trading/PowerTrading-v0",
    entry_point="gym_power_trading.envs:PowerTraddiringEnv",
)
