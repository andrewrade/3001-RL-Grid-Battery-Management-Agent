from collections import deque

class Battery():

    def __init__(self, nominal_capacity, continuous_power, charging_efficiency, observation_window_size) -> None:
        """
        Based on San Diego BESS System 
            nominal_capacity = 80 Mwh
            continuous_power = 20 kW (charging and discharging)
            charging_efficiency = 95%
        """
        self.nominal_capacity = nominal_capacity
        self.continuous_power = continuous_power
        self.charging_efficiency = charging_efficiency
        self.observation_window_size = observation_window_size
        self.current_capacity = 0 
        self.avg_energy_price = 0
        
        # Store window size observations of battery state  to append to envirnoment observations
        self.capacity_observation = deque([0] * self.observation_window_size, maxlen=self.observation_window_size) 
        self.avg_price_observation = deque([0] * self.observation_window_size, maxlen=self.observation_window_size)
    
    def charge(self, energy_price, duration=1):
        """
        Parameters:
            Duration (float): Charging duration in hours 
            Energy_Price (float): Energy price in $/kWh
        Returns:
            duration (float): Total duration charged (for calculating reward penalty for overcharging)
            overcharge (bool): Flag indicating whether capacity would've been exceeded
        """
        charge_0 = self.current_capacity
        charge_1 = charge_0 + (duration * self.continuous_power)

        overcharge = False
        # Check for sufficient capacity to charge for full duration
        if charge_1 > self.nominal_capacity: 
            actual_charge = self.nominal_capacity - charge_0
            duration = actual_charge / self.continuous_power # Correct duration if capacity would be exceeded
            overcharge = True

        # Correct energy price for efficiency losses
        effective_energy_price = energy_price / self.charging_efficiency 

        # Pool energy costs; treat all energy as fungible once it enters battery
        self.avg_energy_price = round((self.avg_energy_price * self.current_capacity + duration * effective_energy_price * self.continuous_power) \
            / (self.current_capacity + duration * self.continuous_power), 2) 
     
        self.current_capacity += duration * self.continuous_power

        # Append State to Observation Window
        # Normalize state of charge
        pct_capacity = self.current_capacity / self.nominal_capacity 
        
        self.capacity_observation.append(pct_capacity)
        self.avg_price_observation.append(self.avg_energy_price) 

        return (duration, overcharge)
    
    def discharge(self, duration=1):
        """
        Parameters:
            Duration (float): Charging duration in hours 
            Energy_Price (float): Energy price in $/kWh
        Returns:
            energy sold (float): Total amount of energy actually sold (discharged) 
        """
        charge_0 = self.current_capacity
        charge_1 = charge_0 - duration * self.continuous_power
        
        # Check for sufficient capacity to discharge for full duration
        if charge_1 < 0: 
            actual_discharge = self.current_capacity
            duration = (actual_discharge / self.continuous_power) # Correct duration if capacity would be exceeded

        energy_sold = duration * self.continuous_power
        self.current_capacity -= energy_sold

        # Append State to Observation Window
        self.capacity_observation.append(self.current_capacity)
        self.avg_price_observation.append(self.avg_energy_price)
        
        return  (duration)
    
    def hold(self):
        """
        Append Battery state to observation window for ticks 
        when Agent decides to hold
        """
        self.capacity_observation.append(self.current_capacity)
        self.avg_price_observation.append(self.avg_energy_price)
    
    def reset(self):
        """
        Reset battery internal state (between episodes)
        """
        self.current_capacity = 0
        self.avg_energy_price = 0
