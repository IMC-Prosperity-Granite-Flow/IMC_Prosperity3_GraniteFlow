#%%
import math
import numpy as np
from scipy.stats import norm

def bs_call_price(S, K, T, sigma):
    """
    Black-Scholes 欧式看涨期权定价（r=0）

    参数:
    S : 标的资产当前价格
    K : 行权价
    T : 距离到期时间（以年为单位）
    sigma : 年化波动率（如 0.2 表示 20%）

    返回:
    欧式看涨期权的理论价格
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)  # 到期或波动率为0时的payoff

    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * norm.cdf(d1) - K * norm.cdf(d2)
    return call_price

def implied_volatility_call_bisect(market_price, S, K, T, 
                                   sigma_low=0.001, sigma_high=1.0, 
                                   tol=1e-6, max_iter=50):
    """
    用二分法估计欧式看涨期权的隐含波动率（r=0）

    参数:
    market_price : 市场价格
    S, K, T : 当前价格、行权价、到期时间
    sigma_low, sigma_high : 搜索区间
    tol : 收敛容差
    max_iter : 最大迭代次数

    返回:
    隐含波动率 或 None（如果未收敛）
    """
    def objective(sigma):
        return bs_call_price(S, K, T, sigma) - market_price

    low = sigma_low
    high = sigma_high

    f_low = objective(low)
    f_high = objective(high)

    if f_low * f_high > 0:
        return None  # 无根或有多个根，不满足二分法条件

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = objective(mid)

        if abs(f_mid) < tol:
            return mid

        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)  # 达到最大迭代次数，返回中值

#%%
import pandas as pd

midprice = pd.read_csv(r"E:\LGU\quant\IMC\IMC_Prosperity3_GraniteFlow\data\round3\days\mid_price_day0.csv")

#%%
def compute_iv_dataframe(df, maturity_days=7):
    T = maturity_days / 365  # 年化到期时间
    underlying_col = 'VOLCANIC_ROCK'

    result_iv = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col == 'timestamp' or col == underlying_col:
            continue

        # 从列名提取行权价
        try:
            strike = float(col.split('_')[-1])
        except ValueError:
            continue

        # 用 lambda 向量化处理：对每行解隐含波动率
        ivs = []
        for idx, row in df.iterrows():
            S = row[underlying_col]
            market_price = row[col]
            iv = implied_volatility_call_bisect(market_price, S, strike, T)
            ivs.append(iv)

        result_iv[col.replace('VOLCANIC_ROCK_VOUCHER_', 'IV_')] = ivs

    return result_iv

#%%
iv_dataframe = compute_iv_dataframe(midprice.iloc[:50])
# midprice_iv_all = pd.concat([midprice[['timestamp']], iv], axis=1)

#%%
def fit_vol_surface(K, sigma_imp):
    K = np.array(K)/10000
    sigma_imp = np.array(sigma_imp)
    
    # 构造设计矩阵 X: [1, K, K^2]
    X = np.column_stack([
        np.ones_like(K),    # 常数项 β0
        K,                  # β1
        K**2,               # β2
    ])
    
    # 最小二乘解 β = (X^T X)^(-1) X^T y
    XtX = X.T @ X
    Xty = X.T @ sigma_imp
    beta = np.linalg.inv(XtX) @ Xty
    
    return beta

beta = fit_vol_surface([9500, 9750, 10000, 10250, 10500], iv_dataframe.iloc[0])

#%%
K = np.array([9500, 9750, 10000, 10250, 10500])/10000
X = np.column_stack([
    np.ones_like(K),    # 常数项 β0
    K,                  # β1
    K**2,               # β2
])
iv = beta @ X.T