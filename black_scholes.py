from math import log, sqrt, exp
from scipy.stats import norm

class BlackScholes():
    def __init__(self, S, K, r, T, option_type, sigma=0, option_price=0):
        ''' Black-Scholes Used for pricing European options on stocks without dividends.
        Attributes
        ==========
        S: spot/underlying price
        K: strike price
        T: time to expiration (in year)
        r: risk-free rate
        option_type: 'put'/'call'
        sigma: implied volatility percentage
        option_price: price of option     

        User can alternatevely uses option_price or sigma as inputs.
        
        '''
        if sigma == 0 and option_price == 0:
            raise ValueError('Sigma or option_price must be used to initialize the class')
    
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.option_type = option_type.lower()
        self.sigma = sigma
        self.option_price = option_price
    
    def _ensure_sigma_initialized(self):
        if self.sigma == 0:
            self.sigma = self.implied_volatility()
    
    def _ensure_price_initialized(self):
        if self.option_price == 0:
            self.option_price = self.price()

    def d1(self):
        self._ensure_sigma_initialized()
        d1 = (log(self.S/self.K) + (self.r + self.sigma**2/2)*self.T) / (self.sigma * sqrt(self.T))
        return d1

    def d2(self):
        self._ensure_sigma_initialized()
        d1 = self.d1()
        d2 = d1 - self.sigma * sqrt(self.T)
        return d2

    def price(self):
        self._ensure_sigma_initialized()
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
    
    def rho(self):
        self._ensure_sigma_initialized()
        if self.option_type == 'call':
            return self.K * self.T * exp(-self.r * self.T) * norm.cdf(self.d2()) /100
        elif self.option_type == 'put':
            return -self.K * self.T * exp(-self.r * self.T) * norm.cdf(-self.d2()) /100
        else:
            raise ValueError('Invalid option type')
    
    def implied_volatility(self, sigma_est=0.5, it=100):
        self._ensure_price_initialized()
        option = BSM(self.S, self.K, self.r, self.T, self.option_type, sigma_est)
        #Newton-Raphson
        for i in range(it):
            option.sigma -= (option.price() - self.option_price) / (option.vega() * 100)
        self.sigma = option.sigma
        return option.sigma