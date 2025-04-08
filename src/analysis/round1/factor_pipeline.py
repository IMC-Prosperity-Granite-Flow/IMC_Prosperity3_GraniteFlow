import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from scipy.special import gamma
from sklearn.model_selection import GridSearchCV

class FactorPipeline:
    def __init__(self, df, horizon=10, feature_cols=None):
        self.df = df
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.scaler = None
        self.pca = None

    def run(self):
        """
        Runs the factor pipeline.
        All procedure.
        """
        print("Running factor pipeline...")
        print("Preprocessing data...")
        self.df = self.orderbook_preprocess(self.df)
        print("Generating features...")
        self.df = self.generate_features(self.df)
        print("Adding fractional features...")
        self.df = self.add_fractional_features(self.df, self.feature_cols)

        # 更新feature_cols包含新添加的分数阶特征
        new_frac_cols = [col for col in self.df.columns if '_fracdiff_' in col]
        self.feature_cols.extend(new_frac_cols)

        print("Cleaning and standardizing features...")
        self.df , self.scaler = self.clean_and_standardize(self.df, self.feature_cols)
        print("Generating label...")
        self.df = self.generate_label(self.df, self.horizon)
        print("Selecting features...")
        importance, model, X_train, y_train = self.select_features(self.df, self.feature_cols)
        #print("Plotting feature importance...")
        #self.plot_feature_importance(importance)
        print("Running SHAP analysis...")
        self.shap_analysis(model, X_train)
        print("Running cross-validation...")
        scores = self.cross_validation_score(model, self.df[self.feature_cols], self.df['target'])
        print("Cross-validation scores:", scores)
        print("Mean CV accuracy:", np.mean(scores))
        
        X_pca, self.pca = self.reduce_dimensionality(self.df[self.feature_cols])
        
        print("Running auto-tuning...")
        model = self.auto_tune_model(X_pca, self.df['target'])
        return self.df, model
    
    def predict(self, df, trained_model):
        """
        make prediction 
        """
        feature_cols = self.feature_cols.copy()
        df = df.copy()
        print("Preprocessing data...")
        df = self.orderbook_preprocess(df)
        print("Generating features...")
        df = self.generate_features(df)

        # 🚫 避免对已经是 fracdiff 的特征重复处理
        base_feature_cols = [col for col in feature_cols if 'fracdiff' not in col]
        print("Adding fractional features...")
        df = self.add_fractional_features(df, base_feature_cols)

        # 更新 feature_cols：基础 + 新生成的 fracdiff 特征
        new_frac_cols = [col for col in df.columns if '_fracdiff_' in col]
        feature_cols = base_feature_cols + new_frac_cols

        print("Cleaning and standardizing features...")
        df , _ = self.clean_and_standardize(df, feature_cols)
        print("Generating label...")
        df = self.generate_label(df, self.horizon)

        # 使用PCA转换新数据
        new_X_pca = self.pca.transform(df[feature_cols])

        # 预测
        predictions = trained_model.predict(new_X_pca)

        #计算误差
        errors = abs(predictions - df['target'])
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
        df['predictions'] = predictions

        return df
    @staticmethod
    def orderbook_preprocess(df):
        """
        对订单簿数据进行预处理
        1.填充NAN为0
        2.计算best_bid, best_ask
        """

        #填充orderbook数据,nan填为0
        columns = ['ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3']
        df = df.copy()
        for column in columns:
            df[column] = df[column].fillna(0)
        
        #计算best_bid, best_ask, spread, mid_price, vwap
        df['best_bid'] = df[['bid_price_1', 'bid_price_2', 'bid_price_3']].min(axis=1)
        df['best_ask'] = df[['ask_price_1', 'ask_price_2', 'ask_price_3']].max(axis=1)

        return df
    
    @staticmethod
    def generate_features(df):
        df = df.copy()
        df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2
        df['spread'] = df['ask_price_1'] - df['bid_price_1']
        df['vwap'] = (
            df['ask_price_1'] * df['ask_volume_1'] +
            df['ask_price_2'] * df['ask_volume_2'] +
            df['ask_price_3'] * df['ask_volume_3'] +
            df['bid_price_1'] * df['bid_volume_1'] +
            df['bid_price_2'] * df['bid_volume_2'] +
            df['bid_price_3'] * df['bid_volume_3']
        ) / (
            df['ask_volume_1'] + df['ask_volume_2'] + df['ask_volume_3'] +
            df['bid_volume_1'] + df['bid_volume_2'] + df['bid_volume_3']
        )

        df['ask_volume'] = df['ask_volume_1'] + df['ask_volume_2'] + df['ask_volume_3']
        df['bid_volume'] = df['bid_volume_1'] + df['bid_volume_2'] + df['bid_volume_3']
        df['ask_bid_ratio'] = (df['ask_volume'] - df['bid_volume']) / (df['ask_volume'] + df['bid_volume'])
        df['log_return_5'] = np.log(df['mid_price']).diff(5)
        df['log_return_1'] = np.log(df['mid_price']).diff(1)
        df['vol_10'] = df['mid_price'].rolling(10).std()
        df['kurt_50'] = df['log_return_5'].rolling(50).apply(lambda x: kurtosis(x, fisher=False), raw=True)
        df['mid_mean_20'] = df['mid_price'].rolling(20).mean()
        df['mid_reversion_gap'] = df['mid_price'] - df['mid_mean_20']
        df['orderbook_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'] + 1e-6)
        df['depth_ratio'] = df['bid_volume_1'] / (df['bid_volume'] + 1e-6)
        df['relative_spread'] = df['spread'] / (df['mid_price'] + 1e-6)
        df['imbalance_diff'] = df['orderbook_imbalance'].diff()
        df['vol_diff'] = df['vol_10'].diff()
        df['spread_change'] = df['spread'].diff()
        df['return_vol_ratio'] = df['log_return_1'].abs() / (df['vol_10'] + 1e-6)

        #人工构造alpha
        df['alpha1'] = df['log_return_1'] - df['log_return_5']
        df['alpha2'] = df['mid_reversion_gap'] * df['orderbook_imbalance']
        df['alpha3'] = df['spread'] * df['depth_ratio']
        df['alpha4'] = df['vwap'] - df['mid_price']
        df['alpha5'] = df['relative_spread'] * df['return_vol_ratio']
        return df
    
    @staticmethod
    def add_fractional_features(df, base_cols, orders=[0.5, 1.0], window=5):
        df = df.copy()
        for col in base_cols:
            for order in orders:
                name = f"{col}_fracdiff_{order}"
                df[name] = FactorPipeline.fractional_derivative(df[col].astype(np.float64).fillna(0), order, window)
        return df

    #@staticmethod
    def clean_and_standardize(self, df, feature_cols):
        df = df.copy()
        # 获取所有特征列（包括分数阶特征）
        all_features = [col for col in df.columns if col in feature_cols or col.startswith(tuple(feature_cols)) and 'fracdiff' in col]
        df = df.dropna(subset=all_features + ['mid_price'])
        scaler = StandardScaler()
        df[all_features] = scaler.fit_transform(df[all_features])
        return df, scaler

    @staticmethod
    def generate_label(df, horizon=10):
        df = df.copy()
        df['future_mid'] = df['mid_price'].shift(-horizon)
        # 生成原始标签 (-1, 0, 1)
        raw_label = np.sign(df['future_mid'] - df['mid_price'])
        
        # 将标签转换为 0,1,2 格式
        df['target'] = raw_label.replace({
            -1: 0,  # 下跌 -> 0
            0: 1,   # 平 -> 1
            1: 2    # 上涨 -> 2
        })
        
        # 删除最后horizon行的nan值
        df = df.dropna(subset=['future_mid', 'target'])
        return df

    @staticmethod
    def select_features(df, feature_cols):
        all_features = [col for col in df.columns if col in feature_cols or col.startswith(tuple(feature_cols)) and 'fracdiff' in col]
        X = df[all_features]
        y = df['target']
        
        # 确保y中没有nan
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
        model.fit(X_train, y_train)
        importance = pd.Series(model.feature_importances_, index=feature_cols)
        importance = importance.sort_values(ascending=False)
        return importance, model, X_train, y_train
    
    @staticmethod
    def plot_feature_importance(importance):
        top_k = importance.head(15)
        top_k.plot(kind='barh', figsize=(10, 6))
        plt.gca().invert_yaxis()
        plt.title("Top 15 Feature Importances from XGBoost")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def shap_analysis(model, X_train):
        explainer = shap.Explainer(model)
        shap_values = explainer(X_train)
        shap.summary_plot(shap_values, X_train, plot_type='bar')

    @staticmethod
    def cross_validation_score(model, X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        print("Cross-validation scores:", scores)
        print("Mean CV accuracy:", np.mean(scores))
        return scores

    @staticmethod
    def reduce_dimensionality(X, n_components=5):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        print("Explained variance ratio:", pca.explained_variance_ratio_)
        return X_pca, pca

    @staticmethod
    def fractional_derivative(series, order, window):
        weights = [(-1)**k * gamma(order + 1) / (gamma(k + 1) * gamma(order - k + 1)) for k in range(window)]
        result = np.zeros_like(series, dtype=np.float64)
        for i in range(window, len(series)):
            result[i] = np.dot(weights, series[i-window:i][::-1])
        result[:window] = np.nan
        return result
    
    @staticmethod
    def auto_tune_model(X, y):
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 150]
        }
        model = XGBClassifier()
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X, y)
        print("Best parameters found:", grid_search.best_params_)
        return grid_search.best_estimator_
