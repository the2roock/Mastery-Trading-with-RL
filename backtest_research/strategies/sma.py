from backtesting import Strategy

from indicators import sma


class SMAStrategy(Strategy):
    lower_bound: int = 2
    upper_bound: int = 8
    
    def init(self):
        self.entry = self.I(sma, self.data.df, self.lower_bound, self.upper_bound)
    
    def next(self):
        if self.entry == 1:
            self.position.close()
            self.buy(size=1)
        elif self.entry == 0:
            pass
        elif self.entry == -1:
            self.position.close()
            self.sell(size=1)
