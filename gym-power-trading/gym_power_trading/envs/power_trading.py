import numpy as np
import pandas as pd
from enum import Enum
import gym
from gym_anytrading.envs import TradingEnv
from gym_power_trading.envs.battery import Battery

class Actions(Enum):
    Discharge = 0
    Charge = 1
    Hold = 2

class PowerTradingEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, battery_capacity=80, battery_cont_power=20, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, render_mode)
        self.action_space = gym.spaces.Discrete(len(Actions))
        print(self.action_space) # Adding Hold as a position
        self.trade_fee_ask_percent = 0.005  # unit
        self.battery = Battery(nominal_capacity=battery_capacity, continuous_power=battery_cont_power) # Add battery 
    
    def reset(self, seed=None, options=None):
        '''
        Extend TradingEnv reset method to include battery state re-set 
        '''
        observation, info = super().reset(seed=seed, options=options)
        self.battery.reset()
        return observation, info
    
    def step(self, action):
        '''
         Step forward 1-tick in time
         Parameters:
            action (Enum): Discrete action to take (Hold, Charge, Discharge)
         Returns:
            observation (array): feature array
            step_reward (float): Reward/Penalty agent receives from trading power for the current tick
            done (bool): Whether terminated
            truncated (bool): Whether truncated
            info (dict): Agents total reward, profit and current position
        '''
        self._truncated = False
        if self._current_tick == self._end_tick:
            self._truncated = True

        self._current_tick += 1
        step_reward, step_profit = self._calculate_reward(action)
        self._total_reward += step_reward
        self._total_profit += step_profit

        if action != Actions.Hold.value:
            self._position = Actions.Charge.value if action == Actions.Charge.value else Actions.Discharge.value
            self._last_trade_tick = self._current_tick
        else:
            self._position = Actions.Hold.value

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self._truncated, info

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position, 
            battery_charge=self.battery.current_capacity
        )

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _get_observation(self):
        # Add battery attributes to env observation
        battery_obs = np.array([self.battery.current_capacity, self.battery.avg_energy_price])
        base_obs = np.array([super()._get_observation()])
        augmented_obs = np.append(base_obs, battery_obs)
        return augmented_obs

    def _calculate_reward(self, action):
        '''
        Calculate reward
        Parameters:
            action (Enum): Discrete action to tkae (Hold, Charge, Discharge)
        Returns:
            reward (float): Reward/Penalty from action taken
            profit (float): $ in profit (different from reward & calculated after trade fees)
        '''
        trade = False
        reward = 0
        penalty = 0
        profit = 0

        if action != Actions.Hold.value:
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            if action == Actions.Charge.value:
                # Return overcharge=True if can't charge for full 1-hr tick 
                duration, overcharge = self.battery.charge(current_price, duration=1) 
                reward = (self.battery.avg_energy_price - current_price) * duration

                if overcharge:
                    if duration > 0.9:
                        duration = 0.9 # Clip duration to prevent excessively large penalties
                    # Scale penalty based on extent of overcharging (shorter charge durations indicates more overcharging)
                    penalty = -1 / (1-duration) 
                reward -= penalty 
            else:
                duration = self.battery.discharge(current_price, duration=1)
                # Positive reward for profit, negative for loss
                reward = (self.battery.continuous_power * duration) * (current_price - self.battery.avg_energy_price)
                profit += reward * (1-self.trade_fee_ask_percent)
        
        return reward, profit


