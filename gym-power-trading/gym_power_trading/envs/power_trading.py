import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

import gymnasium as gym
from gym_power_trading.envs.battery import Battery

import logging
logging.basicConfig(level=logging.DEBUG)

class Actions(Enum):
    Discharge = 0
    Charge = 1
    Hold = 2

class PowerTradingEnv(gym.Env):
    '''
    Based on TradingEnv: https://github.com/AminHP/gym-anytrading/tree/master
    Modified to include holding as an action, incorporated power trading 
    via battery charging, discharging, reward and profit functions. 
    '''

    def __init__(self, df, window_size, frame_bound, battery_capacity=80, battery_cont_power=20, charging_efficiency=0.95):
        assert len(frame_bound) == 2

        # Process Inputs
        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.prices, self.signal_features = self._process_data() 
        # Observations (Window Size, Signal Features + Battery Observations)
        self.shape = (window_size * (self.signal_features.shape[1] + 2), )
        self.trade_fee_ask_percent = 0.005  # unit
        
        self.battery = Battery(
                battery_capacity,
                battery_cont_power,
                charging_efficiency,
                self.window_size
        )
        
        BOUND = 1e3
        # Initialize Spaces 
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-BOUND, high=BOUND, shape=self.shape, dtype=np.float32
        )
        
        # episode attributes
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None
        
    
    def reset(self, seed=None, options=None):
        '''
        Reset environment to random state and re-initialize the battery
        '''
        super().reset(seed=seed, options=options)
        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Actions.Hold
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 0.  # unit
        self._first_rendering = True
        self.history = {}
        self.battery.reset()
        observation = self._get_observation()
        info = self._get_info()
        return observation.astype(np.float32), info
    
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
        self._current_tick += 1
        self._truncated = (self._current_tick == self._end_tick) # Truncated = True if last tick in time series
        trade = action != Actions.Hold.value # Trade = True if action is not hold
        
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

        return observation, step_reward, False, self._truncated, info
    
    def render_all(self, title=None):
        '''
        Render Agent actions
        '''
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Actions.Discharge:
                short_ticks.append(tick)
            elif self._position_history[i] == Actions.Charge:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def _get_info(self):
        '''
        Store info about agent after each tick in a dictionary including:
        Parameters:
            total_reward: Cumlative reward over Episode
            total_profit: Cumulative profit over Episode
            position: Current agent position (Hold, Charge, Discharge)
            battery_charge: Battery current state of charge (Mwh)
        Returns:
            info (dict): Dictionary containing above summarized info
        '''
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position,
            battery_charge=self.battery.current_capacity
        )

    def _process_data(self): # Scale features between -1 / 1
        prices = self.df.loc[:, 'Close'].to_numpy()
        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _get_observation(self):
        '''
        Produce Observations for agent
        '''
        # Array with dimensions (window_size x num_features)
        env_obs = self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick+1]
        
        # Battery observations are array with dimensions (window_size x 1)
        # Make column vectors & flip to ensure chronological order with env_obs
        battery_capacity = np.flip(np.array(self.battery.capacity_observation).reshape(-1, 1)) 
        battery_price = np.flip(np.array(self.battery.avg_price_observation).reshape(-1, 1))
        
        # Flatten matrix to window_size * features size array with observations in chronological order
        observation = np.column_stack((env_obs, battery_capacity, battery_price)). \
            reshape(-1, (self.shape[0])).squeeze()
        
        return observation.astype(np.float32)
    
    def _update_history(self, info):
        '''
        Add info from latest state to history
        Parameters:
            info (dict): Contains summary of reward, profit, battery charge 
            and position from latest observation
        '''
        if not self.history:
            self.history = {key: [] for key in info.keys()}
        else:
            for key, value in info.items():
                self.history[key].append(value)

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
        duration_actual = 0
        current_price = self.prices[self._current_tick]

        if trade:
            if action == Actions.Charge.value:
                # Charge battery and calculate reward 
                # (Positive reward for reducing avg power price, penalty for increasing avg power price & overcharging)
                duration_actual, overcharge = self.battery.charge(current_price, duration=1) # Overcharge=True if battery has insufficient capacity to charge full 1-hr tick 
                reward = (self.battery.avg_energy_price - current_price) * duration_actual
                
                if overcharge:
                    if duration_actual < 0.1:
                        duration_actual = 0.1 # Clip duration to prevent excessively large penalties
                    penalty = -1 / (duration_actual) # Scale penalty by amt of overcharging (shorter charge duration = longer overcharging)
                reward += penalty 
            else:
                # Discharge battery and calculate reward (Positive reward for profit, negative for loss)
                duration_actual = self.battery.discharge(duration=1)
                reward = (self.battery.continuous_power * duration_actual) * (current_price - self.battery.avg_energy_price) 
        else:
            self.battery.hold() # Call hold method to capture state observation in battery deque 
            
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
            profit += (current_price - self.battery.avg_energy_price) * power_traded * (1 - self.trade_fee_ask_percent)
        
        return profit



