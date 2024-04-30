from enum import Enum

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from gym_power_trading.envs.battery import Battery

class Actions(Enum):
    Discharge = 0
    Charge = 1
    Hold = 2

class PowerTradingEnv(gym.Env):
    """
    Based on TradingEnv: https://github.com/AminHP/gym-anytrading/tree/master
    Modified to include holding as an action, incorporated power trading 
    via battery charging, discharging, reward and profit functions. 
    """
    def __init__(self, df, window_size, frame_bound, battery_capacity=80, battery_cont_power=20, charging_efficiency=0.95):
        """
        Parameters:
            df (dataframe): Dataframe containing raw historical power prices for agent training / testing
            window_size (int): Number of time ticks to include in each observation
            frame_bound (tuple): Start and end index from the dataframe to train / test the agent on
            battery_capacity (float): Capacity of battery agent manages. Same units as cost basis from df
            battery_cont_power (float): Battery continious power output. Same scale as capacity (ie kWh -> kW, Mwh -> Mw)
            charging_efficiency (float): Percentage conversion between purchased an stored energy. 
        """
        assert len(frame_bound) == 2

        # Process Inputs
        self.df = df
        self.frame_bound = frame_bound
        self.window_size = window_size
        self._start_tick = self.frame_bound[0] + (self.window_size - 1)
        self._end_tick = self.frame_bound[1] - 1
        self.prices, self.signal_features = self._process_data() 
        self.trade_fee_ask_percent = 0.005  # unit

        # Observations (Window Size, Signal Features + Battery Observations)
        BATTERY_OBSERVATIONS = 2
        self.shape = (window_size * (self.signal_features.shape[1] + BATTERY_OBSERVATIONS),)
        
        self.battery = Battery(
                battery_capacity,
                battery_cont_power,
                charging_efficiency,
                self.window_size
        )

        BOUND = 1e2
        # Initialize Spaces 
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-BOUND, high=BOUND, shape=self.shape, dtype=np.float32
        )
        
        # episode attributes
        self._truncated = None
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None
    
    def set_frame_bound(self, start, end):
        """
        Used to increment frame indices for training episodes
        Parameters:
            start (int): Episode starting index
            end (int): Index for final tick in episode  
        """
        self.frame_bound = (start, end)
        self._start_tick = start + (self.window_size - 1)
        self._end_tick = end - 1
        self.reset()
        
    def reset(self, seed=None, options=None):
        """
        Reset environment and battery state
        """
        super().reset(seed=seed, options=options)
        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Actions.Hold
        self._position_history = ((self.window_size - 1) * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 0.  # unit
        self.history = {}
        self.battery.reset()
        observation = self._get_observation()
        info = self._get_info()
        return observation.astype(np.float32), info
    
    def step(self, action):
        """
         Step forward 1-tick in time
         Parameters:
            action (Enum): Discrete action to take (Hold, Charge, Discharge)
         Returns:
            observation (array): feature array
            step_reward (float): Reward/Penalty agent receives from trading power for the current tick
            done (bool): Whether terminated (always False because environment is not episodic)
            truncated (bool): Whether truncated 
            info (dict): Agents total reward, profit and current position
        """
        self._current_tick += 1
        self._truncated = (self._current_tick == self._end_tick) # Truncated = True if last tick in time series
        self._done = self._truncated
        trade = action != Actions.Hold.value # Trade = True if action is not hold
        
        # Calculate reward & profit, update totals
        step_reward, power_traded = self._calculate_reward(action)
        self._total_reward += step_reward
        self._total_profit += self._update_profit(power_traded, action)

        # Update position if agent makes trade
        if trade:
            self._position = Actions.Charge if action == Actions.Charge.value else Actions.Discharge
            self._last_trade_tick = self._current_tick
        else:
            self._position = Actions.Hold

            # Record latest observation + environment info
        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        return observation, step_reward, self._done, self._truncated, info
    
    def render_all(self, title=None, xlim=None, fig_size=(10, 5)):
        """
        Render plot of Agent actions (Charge / Discharge) throughout Episode
        Parameters:
            title (Str): Title of chart
            xlim (tup): Start and end indices for x-axis
            fig_size (tup): Size of rendered plot
        """
        # Plot prices over agent frame bound (start --> end)
        eval_window_prices = self.prices[self._start_tick:self._end_tick]
        window_ticks = np.arange(len(eval_window_prices))
    
        discharge_ticks = []
        charge_ticks = []

        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Actions.Charge:
                charge_ticks.append(tick)
            elif self._position_history[i] == Actions.Discharge:
                discharge_ticks.append(tick)
        
        fig, ax = plt.subplots(figsize=fig_size)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.plot(eval_window_prices)
        ax.plot(discharge_ticks, eval_window_prices[discharge_ticks], 'ro', label="Discharge")
        ax.plot(charge_ticks, eval_window_prices[charge_ticks], 'go', label="Charge")
        plt.legend()

        if xlim is not None:
            plt.xlim(xlim)
        if title:
            plt.title(title)

        plt.suptitle(
            f"Total Reward: {self._total_reward:,.2f} ~ Total Profit: ${self._total_profit:,.2f}" 
        )


    def _get_info(self):
        """
        Store info about agent after each tick in a dictionary including:
        Parameters:
            total_reward: Cumlative reward over Episode
            total_profit: Cumulative profit over Episode
            position: Current agent position (Hold, Charge, Discharge)
            battery_charge: Battery current state of charge (Mwh)
        Returns:
            info (dict): Dictionary containing above summarized info
        """
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position,
            battery_charge=self.battery.current_capacity
        )

    def _process_data(self): 
        # Shit 10 ticks forward to match DA dims
        prices = self.df.loc[:, 'RT_LMP'].to_numpy()[:-10] 
        # See the Day Ahead price"forecast" for 10 hours ahead 
        # (DA LMPs are released at 2pm and apply to the 24 hours of the next day, 
        #   so 10 future hours are always available)
        da_prices = self.df.loc[:, 'DA_LMP'].to_numpy()[10:] 

        diff = np.diff(prices)
        pct_diff = np.insert(np.where(prices[:-1] != 0, diff / prices[:-1], 0), 0, 0) # Change from price 1-tick ago
        prices_signal = prices / da_prices # Take ratio of current price to DA price
        signal_features = np.column_stack((prices_signal, pct_diff))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _get_observation(self):
        """
        Produce Observations vector for agent. Each vector has 
        window_size obervations in chronological order that include
        signal_features + battery_capacity and battery_avg_price.
        Returns:
            observation (ndarray): Array with dims window_size  x (signal_features + battery_observations) 
        """
        # Array with dimensions (window_size x num_features)
        env_obs = self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick + 1]
        
        # Battery observations are arrays with dimensions (window_size x 1)
        battery_capacity = np.array(self.battery.capacity_observation).reshape(-1, 1)
        battery_avg_charge_price = np.array(self.battery.avg_price_observation).reshape(-1, 1)

        # Normalize battery avg_charge price based on rolling avg over window 
        # (focus on local price dynamics + don't peek into future)
        rolling_avg = self.prices[(self._current_tick - self.window_size + 1):self._current_tick].mean() # Window ma
        battery_avg_charge_price_norm =  battery_avg_charge_price / rolling_avg 
        
        # Flatten matrix to (1, window_size * features size) array with observations in chronological order
        observation = np.column_stack((env_obs, battery_capacity, battery_avg_charge_price_norm)). \
            reshape(-1, (self.shape[0])).squeeze()
        
        return observation.astype(np.float32)
    
    def _update_history(self, info):
        """
        Add info from latest state to history
        Parameters:
            info (dict): Contains summary of reward, profit, battery charge 
            and position from latest observation
        """
        if not self.history:
            self.history = {key: [] for key in info.keys()}
        else:
            for key, value in info.items():
                self.history[key].append(value)

    def _calculate_reward(self, action):
        """
        Calculate reward over last tick based on action taken
        Parameters:
            action (Enum): Discrete action to take (Hold, Charge, Discharge)
        Returns:
            reward (float): Reward/Penalty from action taken
            power traded (float): Mwh of power traded
        """
        BATTERY_PENALTY = 2 # Penalty for mismanaging battery (ie discharge empty battery, charge full battery)
        NEGATIVE_REVENUE_PENALTY = 10 # Penalty for trading when power prices are negative
        
        reward = 0
        power_traded = 0
        duration_actual = 0
        current_price = self.prices[self._current_tick]

        def symlog(x):
            return np.sign(x)*np.log(np.abs(x)+1)
                
        match action:

            case Actions.Charge.value:
                '''
                Charge battery and calculate penalty
                Overcharge=True if battery has insufficient capacity to charge full 1-hr tick 
                '''
                duration_actual, overcharge = self.battery.charge(current_price, duration=1) 
                
                if current_price < 0:
                    reward += symlog(duration_actual * self.battery.continuous_power * current_price)
                if overcharge:
                    reward -= BATTERY_PENALTY # Overcharging Penalty

                '''
                Discharge battery as calculate reward/penalty
                Penalize agent for battery mismanagement (charging full / discharging empty) or trading at a loss
                Reward agent for trading at profit (log returns of reward)
                '''
            case Actions.Discharge.value:

                duration_actual = self.battery.discharge(duration=1)

                if duration_actual == 0:
                    reward -= BATTERY_PENALTY # Discharging when empty Penalty
                elif current_price < 0:
                    reward -= NEGATIVE_REVENUE_PENALTY
                else:
                    revenue = (self.battery.continuous_power * duration_actual) * (current_price)
                    cost_basis = (self.battery.continuous_power * duration_actual) * (self.battery.avg_energy_price)
                    pnl = revenue - cost_basis
                    reward = symlog(pnl)
            
            case Actions.Hold.value:
                '''
                Do nothing 
                (call hold method to capture current state observation in battery observation queue)
                '''
                self.battery.hold()

        power_traded = (duration_actual * self.battery.continuous_power)
        return reward, power_traded
    
    def _update_profit(self, power_traded, action):
        """
        Calculate agent profit over last tick based on action taken
        Parameters:
            power_traded (float): Quantity of power agent traded
        Returns:
            profit (float): Profit in $ generated when the agent sells power
        """
        profit = 0
        current_price = self.prices[self._current_tick]
        
        match action:

            case Actions.Charge.value:
                # Charging costs money (revenue when power price is negative)
                profit -= current_price * power_traded 
            
            case Actions.Discharge.value:
                # Discharging produces revenue (cost when power price is negative)
                profit += current_price * power_traded * (1 - self.trade_fee_ask_percent)
        
        return profit
    



