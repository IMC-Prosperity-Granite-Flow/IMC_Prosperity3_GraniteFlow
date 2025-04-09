from factor_pipeline import FactorPipeline
import pandas as pd
import numpy as np

#提取数据
df1 = pd.read_csv('./data/round-1-island-data-bottle/prices_round_1_day_-2.csv', delimiter = ";")
df2 = pd.read_csv('./data/round-1-island-data-bottle/prices_round_1_day_-1.csv', delimiter = ";")
df3 = pd.read_csv('./data/round-1-island-data-bottle/prices_round_1_day_0.csv', delimiter = ";")

df = pd.concat([df1, df2, df3])

def extract_product_df(df, symbol: str):
    df_product = df[df['product'] == symbol]
    return df_product

#按product分割数据
df_kelp = extract_product_df(df, 'KELP')
df_resin = extract_product_df(df, 'RAINFOREST_RESIN')
df_ink = extract_product_df(df, 'SQUID_INK')

feature_cols = [
    'spread', 
    'vwap',
    'ask_volume',
    'bid_volume',
    'ask_bid_ratio',
    'vol_10',
    'kurt_50',
    'mid_mean_20',
    'mid_reversion_gap',
    'orderbook_imbalance',
    'depth_ratio',
    'relative_spread',
    'imbalance_diff',
    'vol_diff',
    'spread_change',
    'return_vol_ratio',
    'alpha1',
    'alpha2',
    'alpha3',
    'alpha4',
    'alpha5'
]

def train_and_predict(symbol:str):
    #训练模型
    df_train = extract_product_df(df, symbol)
    pipeline = FactorPipeline(df_train, 10, feature_cols=feature_cols)
    pipeline_df, trained_model= pipeline.run()


    new_data = df_train.iloc[10000:]
    new_df = pipeline.predict(new_data, trained_model)

    new_df.to_csv('./src/analysis/round1/' + symbol + '_prediction.csv', index=False)



if __name__ == '__main__':
    products = ['KELP', 'RAINFOREST_RESIN', 'SQUID_INK']
    for product in products:
        train_and_predict(product)
