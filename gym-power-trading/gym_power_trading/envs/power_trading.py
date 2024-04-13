import numpy as np
import pandas as pd
from gym_anytrading import TradingEnv, Actions, Positions
from battery import Battery

class PowerTradingEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, battery_capacity=80, battery_cont_power=20, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, render_mode)

        self.trade_fee_ask_percent = 0.005  # unit
        self.battery = Battery(nominal_capacity=battery_capacity, continuous_power=battery_cont_power)
    
    def reset(self, seed=None, options=None):
        '''
        Extends TradingEnv reset method to re-set battery state
        '''
        observation, info = super().reset(seed=seed, options=options)
        self.battery.reset()
        return observation, info

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _get_observation(self):
        # Add battery attributes to env observation
        battery_obs = np.array(self.battery.current_charge, self.battery.avg_energy_price)
        augmented_obs = np.append(super()._get_observation(), battery_obs)
        return augmented_obs

    def _calculate_reward(self, action):

        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        reward = 0
        
        if trade:
            current_price = self.prices[self._current_tick]
            
            if action == Actions.Buy.value:
                reward = self.battery.charge(current_price, duration=1) # Charges for full 1-hr tick 
            else:
                reward = self.battery.discharge(current_price, duration=1)
                self._total_profit += reward * (1-self.trade_fee_ask_percent)

        return reward



    


