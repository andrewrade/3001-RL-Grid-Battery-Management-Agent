class Battery():
    def __init__(self, nominal_capacity=13.5, continuous_power_rate=5, charging_efficiency=0.975) -> None:
        '''
        Battery parameters based on Tesla Powerwall 2 spec sheet:
        (https://digitalassets.tesla.com/tesla-contents/image/upload/powerwall-2-ac-datasheet-en-na_001)
            nominal_capacity = 13.5 kWh
            continuous_power_rate = 5 kW (charging and discharging)
            charging_efficiency = 90%
        '''
        self.nominal_capacity = nominal_capacity
        self.continuous_power_rate = continuous_power_rate
        self.charging_efficiency = charging_efficiency
        self.current_capacity = 0
        self.avg_power_price = 0
    
    def charge(self, duration, power_price):
        '''
            Duration: Charging duration in hours 
            Power_Price: Power price in $/kWh
        '''
        if self.current_capacity == self.nominal_capacity:
            return (self.current_capacity, self.avg_power_price, -1) # penalize agent for trying to charge full battery

        charge_0 = self.current_capacity
        charge_1 = charge_0 + duration * self.continuous_power_rate
        
        # Check for sufficient capacity to charge for full duration
        if charge_1 > self.nominal_capacity: 
            actual_charge = self.nominal_capacity - charge_0
            duration = actual_charge / self.continuous_power_rate # Correct duration if capacity would be exceeded
        
        effective_power_price = power_price / self.charging_efficiency # Correct power price for efficiency losses

        # Pool power costs; treat all power as fungible once it enters battery
        self.avg_power_price = \
            (self.avg_power_price * self.current_capacity + duration * effective_power_price * self.continuous_power_rate) \
            / (self.current_capacity + duration * self.continuous_power_rate)
        
        self.current_capacity += duration * self.continuous_power_rate

        return (self.current_capacity, self.avg_power_price, 0) 
    
    def discharge(self, duration, power_price):
        '''
            Duration: Charging duration in hours 
            Power_Price: Power price in $/kWh
        '''
        charge_0 = self.current_capacity
        charge_1 = charge_0 - duration * self.continuous_power_rate
        
        # Check for sufficient capacity to discharge for full duration
        if charge_1 < 0: 
            actual_discharge = self.current_capacity
            duration = actual_discharge / self.continuous_power_rate # Correct duration if capacity would be exceeded

        power_sold = duration * self.continuous_power_rate
        self.current_capacity -= power_sold
        arbitrage_profit = (power_price - self.avg_power_price) * power_sold

        return (self.current_capacity, self.avg_power_price, arbitrage_profit)

            

