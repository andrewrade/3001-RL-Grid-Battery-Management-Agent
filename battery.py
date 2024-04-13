class Battery():
    def __init__(self, nominal_capacity=13.5, continuous_power=5, charging_efficiency=0.95) -> None:
        '''
        Battery parameters based on Tesla Powerwall 2 spec sheet:
        (https://digitalassets.tesla.com/tesla-contents/image/upload/powerwall-2-ac-datasheet-en-na_001)
            nominal_capacity = 13.5 kWh
            continuous_power = 5 kW (charging and discharging)
            charging_efficiency = 95%
        '''
        self.nominal_capacity = nominal_capacity
        self.continuous_power = continuous_power
        self.charging_efficiency = charging_efficiency
        self.current_capacity = 0
        self.avg_energy_price = 0
    
    def charge(self, energy_price, duration=1):
        '''
            Duration: Charging duration in hours 
            Energy_Price: Energy price in $/kWh
        '''
        if self.current_capacity == self.nominal_capacity:
            return (-1) # penalize agent for trying to charge full battery

        charge_0 = self.current_capacity
        charge_1 = charge_0 + duration * self.continuous_power
        
        # Check for sufficient capacity to charge for full duration
        if charge_1 > self.nominal_capacity: 
            actual_charge = self.nominal_capacity - charge_0
            duration = actual_charge / self.continuous_power # Correct duration if capacity would be exceeded
        
        effective_energy_price = energy_price / self.charging_efficiency # Correct energy price for efficiency losses

        # Pool energy costs; treat all energy as fungible once it enters battery
        self.avg_energy_price = \
            (self.avg_energy_price * self.current_capacity + duration * effective_energy_price * self.continuous_power) \
            / (self.current_capacity + duration * self.continuous_power)
        
        self.current_capacity += duration * self.continuous_power

        return (0)
    
    def discharge(self, energy_price, duration=1):
        '''
        Parameters:
            Duration: Charging duration in hours 
            Energy_Price: Energy price in $/kWh
        Returns:
            current_capacity: 
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

        return  (profit)
            

