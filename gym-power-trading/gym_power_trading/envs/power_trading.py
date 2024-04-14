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
            done (bool): Whether terminated (always False because environment is not episodic)
            truncated (bool): Whether truncated 
            info (dict): Agents total reward, profit and current position
        '''
        trade = action != Actions.Hold.value # Trade = True if action is not hold
        self._truncated = (self._current_tick == self._end_tick) # Truncated = True if last tick in time series
        self._current_tick += 1

        # Calculate reward & profit, update totals
        step_reward, power_traded = self._calculate_reward(action)
        self._total_reward += step_reward
        self._total_profit += self._update_profit(power_traded, action)

        # Update position if agent makes trade
        if trade:
            self._position = Actions.Charge.value if action == Actions.Charge.value else Actions.Discharge.value
            self._last_trade_tick = self._current_tick
        else:
            self._position = Actions.Hold.value

        # Record latest observation + environment info
        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self._truncated, info

    def _get_info(self):
        '''
        Store info about agent after each tick in a dictionary including:
            total_reward: Cumlative reward over Episode
            total_profit: Cumulative profit over Episode
            position: Current agent position (Hold, Charge, Discharge)
            battery_charge: Battery current state of charge (Mwh)
        Returns:
            info (dict): Dictionary containing above summarized info
        '''
        info = super()._get_info()
        info['battery_charge'] = self.battery.current_capacity # Extend info to include battery charge state
        return info

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()
        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _get_observation(self):
        '''
        Extend agent observations to include battery state
        '''
        base_obs = np.array([super()._get_observation()])
        battery_obs = np.array([self.battery.current_capacity, self.battery.avg_energy_price])
        augmented_obs = np.append(base_obs, battery_obs)
        return augmented_obs

    def _calculate_reward(self, action):
        '''
        Calculate reward over last tick based on action taken
        Parameters:
            action (Enum): Discrete action to take (Hold, Charge, Discharge)
        Returns:
            reward (float): Reward/Penalty from action taken
            profit (float): $ in profit (different from reward & calculated after trade fees)
        '''
        trade = action != Actions.Hold.value # Trade = True if action isn't hold
        reward = 0
        penalty = 0
        power_traded = 0

        if trade:
            current_price = self.prices[self._current_tick]
            
            if action == Actions.Charge.value:
                # Charge battery and calculate reward 
                # (Positive reward for reducing avg power price, penalty for increasing avg power price & overcharging)
                duration_actual, overcharge = self.battery.charge(current_price, duration=1) # Overcharge=True if battery has insufficient capacity to charge full 1-hr tick 
                reward = (self.battery.avg_energy_price - current_price) * duration_actual
                
                if overcharge:
                    if duration_actual > 0.9:
                        duration_actual = 0.9 # Clip duration to prevent excessively large penalties
                    penalty = -1 / (1-duration_actual) # Scale penalty by amt of overcharging (shorter charge duration = longer overcharging)
                reward += penalty 
            
            else:
                # Discharge battery and calculate reward (Positive reward for profit, negative for loss)
                duration_actual = self.battery.discharge(current_price, duration=1)
                reward = (self.battery.continuous_power * duration_actual) * (current_price - self.battery.avg_energy_price) 
            
            power_traded = (duration_actual * self.battery.continuous_power)
        return reward, power_traded
    
    def _update_profit(self, power_traded, action):
        '''
        Calculate agent profit over last tick based on action taken
        Parameters:
            power_traded (float): Quantity of power agent traded
        Returns:
            profit (float): Profit in $ generated when the agent sells power
        '''
        profit = 0

        if action == Actions.Discharge.value:
            current_price = self.prices[self._current_tick]
            profit += (current_price - self.battery.avg_energy_price) * power_traded
        
        return profit



