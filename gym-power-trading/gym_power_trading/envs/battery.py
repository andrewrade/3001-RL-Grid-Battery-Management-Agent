class Battery():
    def __init__(self, nominal_capacity=80, continuous_power=20, charging_efficiency=0.95) -> None:
        '''
        Based on San Diego BESS System 
            nominal_capacity = 80 Mwh
            continuous_power = 20 kW (charging and discharging)
            charging_efficiency = 95%
        '''
        self.nominal_capacity = nominal_capacity
        self.continuous_power = continuous_power
        self.charging_efficiency = charging_efficiency
        self.current_capacity = 0
        self.avg_energy_price = 0
    
    def charge(self, energy_price, duration=1):
        '''
        Parameters:
            Duration (float): Charging duration in hours 
            Energy_Price (float): Energy price in $/kWh
        Returns:
            duration (float): Total duration charged (for calculating reward penalty for overcharging)
            overcharge (bool): Flag indicating whether capacity would've been exceeded
        '''
        charge_0 = self.current_capacity
        charge_1 = charge_0 + duration * self.continuous_power

        overcharge = False
        # Check for sufficient capacity to charge for full duration
        if charge_1 > self.nominal_capacity: 
            actual_charge = self.nominal_capacity - charge_0
            duration = actual_charge / self.continuous_power # Correct duration if capacity would be exceeded
            overcharge = True
        
        effective_energy_price = energy_price / self.charging_efficiency # Correct energy price for efficiency losses

        # Pool energy costs; treat all energy as fungible once it enters battery
        self.avg_energy_price = \
            (self.avg_energy_price * self.current_capacity + duration * effective_energy_price * self.continuous_power) \
            / (self.current_capacity + duration * self.continuous_power)
        
        self.current_capacity += duration * self.continuous_power

        return (duration, overcharge)
    
    def discharge(self, energy_price, duration=1):
        '''
        Parameters:
            Duration (float): Charging duration in hours 
            Energy_Price (float): Energy price in $/kWh
        Returns:
            energy sold (float): Total amount of energy actually sold (discharged) 
        '''
        charge_0 = self.current_capacity
        charge_1 = charge_0 - duration * self.continuous_power
        
        # Check for sufficient capacity to discharge for full duration
        if charge_1 < 0: 
            actual_discharge = self.current_capacity
            duration = actual_discharge / self.continuous_power # Correct duration if capacity would be exceeded

        energy_sold = duration * self.continuous_power
        self.current_capacity -= energy_sold
        profit = (energy_price - self.avg_energy_price) * energy_sold

        return  (duration)
    
    def reset(self):
        '''
        Reset battery internal state (between episodes)
        '''
        self.current_capacity = 0
        self.avg_energy_price = 0
