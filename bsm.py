# GRAVITY SUCKS
# Valuation of European options in Black-Scholes-Merton model
# and their relative greeks, and implied volatility estimator
# in object-oriented programmed language

from math import log, sqrt, exp
from scipy.stats import norm

class BSM():
    def __init__(self, S, K, r, T, option_type, sigma=0, option_price = 0):
        ''' Black-Scholes Used for pricing European options on stocks without dividends.
        Attributes
        ==========
        S0: float
            initial stock/index level
        K: float
            strike price
        T: float
            maturity (in year fractions)
        r: float
            constant risk-free short rate
        option_type: string
            'put' or 'call'
        sigma: float
            volatility factor in percentage term
        option_price: float
            price of option     
        
        
        e.g.
        
        Given implied volatility return the option_price
        BSM(S, K, r, T, option_type, sigma =sigma).price()
        
        Other methods:
        - d1
        - d2
        - delta
        - gamma
        - theta
        - vega
        
        
        Given an option price return the implied volatility
        BSM(S, K, r, T, option_type, option_price = option_price).implied_volatility()
        
        
        
        '''
        
        
    
        if sigma == 0 and option_price == 0:
            raise ValueError('Sigma or option_price must be used to initialize the class')
    
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.option_type = option_type
        self.sigma = sigma
        self.option_price = option_price

    def d1(self):
        d1 = (log(self.S/self.K) + (self.r + self.sigma**2/2)*self.T) / (self.sigma * sqrt(self.T))
        return d1

    def d2(self):
        d1 = (log(self.S/self.K) + (self.r + self.sigma**2/2)*self.T) / (self.sigma* sqrt(self.T))
        d2 = d1 - self.sigma * sqrt(self.T)
        return d2

    def price(self):
        ''' Returns price of option given implied volatility.
        '''
        if self.option_type == 'call':
            price = self.S * norm.cdf(self.d1()) - self.K * exp(-self.r * self.T) * norm.cdf(self.d2())
        elif self.option_type == 'put':
            price = self.K * exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S * norm.cdf(-self.d1())
        else:
            raise ValueError('Invalid option type')
        return price
    
    def delta(self):
        if self.option_type == 'call':
            delta = norm.cdf(self.d1())
        elif self.option_type == 'put':
            delta = -norm.cdf(-self.d1())
        else:
            raise ValueError('Invalid option type')
        return delta
        
    def gamma(self):
        gamma = norm.pdf(self.d1()) / (self.S*self.sigma* sqrt(self.T))
        return gamma

    def theta(self):
        if self.option_type == 'call':
            theta_calc = -self.S*norm.pdf(self.d1())*self.sigma/(2* sqrt(self.T)) - self.r*self.K* exp(-self.r*self.T)*norm.cdf(self.d2())
        elif self.option_type == 'put':
            theta_calc = -self.S*norm.pdf(self.d1())*self.sigma/(2* sqrt(self.T)) + self.r*self.K* exp(-self.r*self.T)*norm.cdf(-self.d2())
        else:
            raise ValueError('Invalid option type')
        
        return theta_calc / 365

    def vega(self):
        vega = self.S * norm.pdf(self.d1()) * sqrt(self.T)
        return vega * 0.01
    
    def implied_volatility(self, sigma_est=0.5, it=100):
        ''' Returns implied volatility given option price.
        '''
        option = BSM(self.S, self.K, self.r, self.T, self.option_type, sigma_est)
        for i in range(it):
            option.sigma -= (option.price() - self.option_price) / (option.vega() * 100)
        return option.sigma