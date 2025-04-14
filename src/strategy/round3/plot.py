import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

S = 10000
K = 10000
r = 0.0
sigma = 0.3

Ts = np.linspace(0.001, 2.0, 100)
prices = [bs_call_price(S, K, T, r, sigma) for T in Ts]
intrinsic = [max(S - K, 0) for _ in Ts]
time_values = [p - iv for p, iv in zip(prices, intrinsic)]

plt.plot(Ts, prices, label='Total Option Price (BS)')
plt.plot(Ts, intrinsic, label='Intrinsic Value')
plt.plot(Ts, time_values, label='Time Value')
plt.xlabel('T (Time to Maturity)')
plt.ylabel('Value')
plt.title('Decomposition of Option Price')
plt.legend()
plt.grid()
plt.show()