import numpy as np
import matplotlib as plt

initial = 1000
annual_income = lambda t: 1000  # disposable only
n_years = 10
n_samples = 1000
market_data = np.random.randn(n_years, n_samples) * 0.2 + 0.1  # first axis is time, second is randomness
market_data[market_data < -1] = -1


def stock_market(market_data, fraction_cash):
    s1 = market_data.shape
    s2 = fraction_cash.shape
    money = np.empty(s1 + s2)
    fraction_cash = fraction_cash.reshape((1,) * len(s1) + s2)
    market_data = market_data.reshape(s1 + (1,) * len(s2))
    money[-1] = initial
    for t in range(len(market_data)):
        money[t] = money[t - 1] + annual_income(t) + (1 - fraction_cash) * money[t - 1] * market_data[t]
    return money


f = np.linspace(0, 1, 11)
money = stock_market(market_data, f)
plt.plot(f, np.mean(money[-1], axis=0))
for q in [.05, .25, .75, .95]:
    plt.plot(f, np.quantile(money[-1], q, axis=0), 'k--')
