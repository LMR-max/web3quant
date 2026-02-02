# factor_research.py
"""
因子研究与测试框架
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

class FactorResearch:
    """
    因子研究框架
    用于因子挖掘、测试、组合
    """
    
    def __init__(self):
        self.factors = {}  # 存储因子数据
        self.factor_performance = {}  # 因子绩效
        self.combined_factors = {}  # 组合因子
        
    def create_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        创建常见因子
        """
        factors = {}
        
        # 1. 价格动量因子
        for window in [5, 10, 20, 60]:
            factors[f'momentum_{window}'] = df['close'].pct_change(window)
            factors[f'roc_{window}'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
        
        # 2. 均线距离因子
        for short in [5, 10, 20]:
            for long in [50, 100, 200]:
                if short < long:
                    ma_short = df['close'].rolling(short).mean()
                    ma_long = df['close'].rolling(long).mean()
                    factors[f'ma_distance_{short}_{long}'] = (ma_short - ma_long) / ma_long
        
        # 3. 波动率因子
        for window in [5, 10, 20]:
            returns = df['close'].pct_change()
            factors[f'volatility_{window}'] = returns.rolling(window).std()
            factors[f'volatility_ratio_{window}'] = returns.rolling(window).std() / returns.rolling(window*2).std()
        
        # 4. 成交量因子
        factors['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        factors['volume_price_corr'] = df['close'].rolling(20).corr(df['volume'])
        
        # 5. 技术指标因子
        # RSI (Standard Wilder's RSI)
        delta = df['close'].diff()
        # 使用 Wilder's Smoothing (alpha=1/14) 替代简单移动平均
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        factors['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        factors['macd'] = exp1 - exp2
        # 新增 Signal 线 (9日 EMA) 和 Histogram
        factors['macd_signal'] = factors['macd'].ewm(span=9, adjust=False).mean()
        factors['macd_hist'] = factors['macd'] - factors['macd_signal']
        
        # 布林带位置
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        factors['bb_position'] = (df['close'] - bb_middle) / (2 * bb_std)
        
        # 6. 反转因子
        for window in [5, 10, 20]:
            factors[f'reversal_{window}'] = -df['close'].pct_change(window)
        
        # 7. 偏度因子
        for window in [20, 50, 100]:
            returns = df['close'].pct_change()
            factors[f'skewness_{window}'] = returns.rolling(window).skew()
            factors[f'kurtosis_{window}'] = returns.rolling(window).kurt()
        
        # 8. 波动/流动性因子
        # Range Amplitude (原 spread, 重命名以避免歧义): (High - Low) / Close
        factors['range_amplitude'] = (df['high'] - df['low']) / df['close']
        
        # Parkinson Volatility (基于 High/Low 的波动率估算)
        # Formula: sqrt( 1/(4*ln(2)) * mean( ln(H/L)^2 ) )
        const_parkinson = 1.0 / (4.0 * np.log(2.0))
        log_hl_sq = np.log(df['high'] / df['low']) ** 2
        # 计算 20 周期 Parkinson 波动率
        factors['volatility_parkinson_20'] = np.sqrt(const_parkinson * log_hl_sq.rolling(20).mean())
        
        # 9. 时间序列因子
        factors['day_of_week'] = df.index.dayofweek
        factors['hour_of_day'] = df.index.hour
        factors['month'] = df.index.month
        
        # 10. 价格位置因子
        window = 20
        rolling_min = df['low'].rolling(window).min()
        rolling_max = df['high'].rolling(window).max()
        factors['price_position'] = (df['close'] - rolling_min) / (rolling_max - rolling_min)
        
        self.factors = factors
        return factors
    
    def evaluate_factor(self, 
                       factor: pd.Series, 
                       forward_returns: pd.Series,
                       evaluation_periods: List[int] = [1, 5, 10, 20]) -> Dict:
        """
        评估因子表现
        """
        results = {}
        
        # 对齐数据
        data = pd.concat([factor, forward_returns], axis=1).dropna()
        factor_values = data.iloc[:, 0]
        returns = data.iloc[:, 1]
        
        # 基础统计
        results['mean'] = factor_values.mean()
        results['std'] = factor_values.std()
        results['sharpe'] = results['mean'] / results['std'] if results['std'] > 0 else 0
        results['skew'] = factor_values.skew()
        results['kurtosis'] = factor_values.kurtosis()
        results['autocorr'] = factor_values.autocorr()
        
        # 计算不同预测周期的IC
        for period in evaluation_periods:
            # 计算未来period期收益率
            future_returns = forward_returns.rolling(period).apply(
                lambda x: (1 + x).prod() - 1, raw=False
            ).shift(-period)
            
            # 对齐数据
            aligned = pd.concat([factor, future_returns], axis=1).dropna()
            if len(aligned) > 10:
                # 计算Rank IC
                ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method='spearman')
                results[f'ic_{period}'] = ic
            else:
                results[f'ic_{period}'] = np.nan
        
        # 分组收益分析
        groups = self._factor_group_analysis(factor_values, returns, n_groups=5)
        results['group_returns'] = groups['mean_return'].tolist()
        results['top_minus_bottom'] = groups['mean_return'].iloc[-1] - groups['mean_return'].iloc[0]
        
        return results
    
    def _factor_group_analysis(self, 
                              factor: pd.Series, 
                              forward_returns: pd.Series,
                              n_groups: int = 5) -> pd.DataFrame:
        """因子分组分析"""
        # 对齐数据
        data = pd.concat([factor, forward_returns], axis=1).dropna()
        factor_values = data.iloc[:, 0]
        returns = data.iloc[:, 1]
        
        # 创建分组
        labels = range(1, n_groups + 1)
        factor_groups = pd.qcut(factor_values, q=n_groups, labels=labels)
        
        # 计算每组收益
        group_returns = returns.groupby(factor_groups).mean()
        group_counts = factor_groups.value_counts().sort_index()
        group_std = returns.groupby(factor_groups).std()
        
        return pd.DataFrame({
            'group': labels,
            'mean_return': group_returns.values,
            'std': group_std.values,
            'count': group_counts.values,
            'sharpe': group_returns.values / group_std.values
        })
    
    def factor_correlation_analysis(self, factors: Dict[str, pd.Series]) -> pd.DataFrame:
        """因子相关性分析"""
        # 创建因子DataFrame
        factor_df = pd.DataFrame(factors)
        
        # 计算相关性矩阵
        correlation_matrix = factor_df.corr()
        
        # 计算特征值和特征向量（用于检验多重共线性）
        corr_matrix = factor_df.corr().values
        eigenvalues = np.linalg.eigvals(corr_matrix)
        
        print(f"特征值范围: {eigenvalues.min():.4f} - {eigenvalues.max():.4f}")
        print(f"条件数: {eigenvalues.max() / eigenvalues.min():.4f}" if eigenvalues.min() > 0 else "条件数: ∞")
        
        return correlation_matrix
    
    def create_factor_composite(self, 
                               factors: Dict[str, pd.Series],
                               method: str = 'equal_weight',
                               weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        创建复合因子
        
        method: 
            'equal_weight' - 等权
            'ic_weight' - IC加权
            'pca' - 主成分分析
            'custom' - 自定义权重
        """
        # 创建因子DataFrame
        factor_df = pd.DataFrame(factors)
        factor_df = factor_df.dropna()
        
        if method == 'equal_weight':
            # 等权复合
            composite = factor_df.mean(axis=1)
            
        elif method == 'ic_weight':
            # 需要计算每个因子的IC
            # 这里简化处理，假设已计算IC
            if weights is None:
                # 如果没有提供权重，使用等权
                composite = factor_df.mean(axis=1)
            else:
                # 使用IC加权
                weighted_factors = []
                for factor_name, weight in weights.items():
                    if factor_name in factor_df.columns:
                        weighted_factors.append(factor_df[factor_name] * weight)
                
                if weighted_factors:
                    composite = pd.concat(weighted_factors, axis=1).mean(axis=1)
                else:
                    composite = factor_df.mean(axis=1)
        
        elif method == 'pca':
            # 主成分分析
            from sklearn.decomposition import PCA
            
            # 标准化
            scaler = StandardScaler()
            scaled_factors = scaler.fit_transform(factor_df)
            
            # PCA
            pca = PCA(n_components=1)
            composite_values = pca.fit_transform(scaled_factors)
            composite = pd.Series(composite_values.flatten(), index=factor_df.index)
            
            # 解释方差比
            print(f"第一主成分解释方差: {pca.explained_variance_ratio_[0]:.2%}")
        
        elif method == 'custom' and weights:
            # 自定义权重
            weighted_factors = []
            for factor_name, weight in weights.items():
                if factor_name in factor_df.columns:
                    weighted_factors.append(factor_df[factor_name] * weight)
            
            if weighted_factors:
                composite = pd.concat(weighted_factors, axis=1).sum(axis=1)
            else:
                composite = factor_df.mean(axis=1)
        
        else:
            raise ValueError(f"不支持的复合方法: {method}")
        
        return composite
    
    def factor_stability_test(self, 
                             factor: pd.Series, 
                             forward_returns: pd.Series,
                             window: int = 60) -> Dict:
        """
        因子稳定性测试
        
        计算滚动IC，检验因子稳定性
        """
        # 对齐数据
        data = pd.concat([factor, forward_returns], axis=1).dropna()
        factor_values = data.iloc[:, 0]
        returns = data.iloc[:, 1]
        
        # 计算滚动IC
        rolling_ic = []
        dates = []
        
        for i in range(len(factor_values) - window):
            start_idx = i
            end_idx = i + window
            
            factor_window = factor_values.iloc[start_idx:end_idx]
            returns_window = returns.iloc[start_idx:end_idx]
            
            ic = factor_window.corr(returns_window, method='spearman')
            rolling_ic.append(ic)
            dates.append(factor_values.index[end_idx])
        
        if rolling_ic:
            rolling_ic_series = pd.Series(rolling_ic, index=dates)
            
            results = {
                'rolling_ic_mean': np.mean(rolling_ic),
                'rolling_ic_std': np.std(rolling_ic),
                'rolling_ic_sharpe': np.mean(rolling_ic) / np.std(rolling_ic) if np.std(rolling_ic) > 0 else 0,
                'ic_positive_ratio': np.mean([1 if ic > 0 else 0 for ic in rolling_ic]),
                'rolling_ic_series': rolling_ic_series
            }
        else:
            results = {
                'rolling_ic_mean': 0,
                'rolling_ic_std': 0,
                'rolling_ic_sharpe': 0,
                'ic_positive_ratio': 0,
                'rolling_ic_series': pd.Series()
            }
        
        return results
    
    def plot_factor_analysis(self, 
                            factor_name: str,
                            factor_series: pd.Series,
                            forward_returns: pd.Series,
                            save_path: Optional[str] = None):
        """
        绘制因子分析图表
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 因子值时间序列
        ax1 = axes[0, 0]
        ax1.plot(factor_series.index, factor_series.values, linewidth=1)
        ax1.set_title(f'{factor_name} - 因子值', fontsize=12)
        ax1.set_ylabel('因子值')
        ax1.grid(True, alpha=0.3)
        
        # 2. 因子分布直方图
        ax2 = axes[0, 1]
        ax2.hist(factor_series.dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(factor_series.mean(), color='red', linestyle='--', 
                   label=f'均值: {factor_series.mean():.4f}')
        ax2.set_title(f'{factor_name} - 分布', fontsize=12)
        ax2.set_xlabel('因子值')
        ax2.set_ylabel('频率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 因子自相关
        ax3 = axes[0, 2]
        lags = min(50, len(factor_series) // 2)
        if lags > 1:
            from pandas.plotting import autocorrelation_plot
            autocorrelation_plot(factor_series, ax=ax3)
            ax3.set_title(f'{factor_name} - 自相关', fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        # 4. 分组收益
        ax4 = axes[1, 0]
        groups = self._factor_group_analysis(factor_series, forward_returns, n_groups=5)
        ax4.bar(groups['group'], groups['mean_return'], alpha=0.7)
        ax4.set_title(f'{factor_name} - 分组收益', fontsize=12)
        ax4.set_xlabel('分组')
        ax4.set_ylabel('平均收益率')
        ax4.grid(True, alpha=0.3)
        
        # 5. 滚动IC
        ax5 = axes[1, 1]
        stability_results = self.factor_stability_test(factor_series, forward_returns)
        rolling_ic = stability_results.get('rolling_ic_series', pd.Series())
        if len(rolling_ic) > 0:
            ax5.plot(rolling_ic.index, rolling_ic.values, linewidth=1, color='green')
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax5.set_title(f'滚动IC (均值: {stability_results["rolling_ic_mean"]:.4f})', fontsize=12)
            ax5.set_ylabel('IC')
            ax5.grid(True, alpha=0.3)
        
        # 6. IC衰减
        ax6 = axes[1, 2]
        max_lag = 20
        decay_results = []
        for lag in range(1, max_lag + 1):
            factor_lagged = factor_series.shift(lag)
            aligned = pd.concat([factor_lagged, forward_returns], axis=1).dropna()
            if len(aligned) > 10:
                ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method='spearman')
                decay_results.append(ic)
            else:
                decay_results.append(np.nan)
        
        ax6.plot(range(1, max_lag + 1), decay_results, marker='o', linewidth=1)
        ax6.set_title('IC衰减', fontsize=12)
        ax6.set_xlabel('滞后周期')
        ax6.set_ylabel('IC')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def generate_factor_report(self, 
                              factor_results: Dict[str, Dict],
                              save_path: Optional[str] = None) -> pd.DataFrame:
        """
        生成因子分析报告
        """
        # 转换为DataFrame
        report_data = []
        
        for factor_name, results in factor_results.items():
            row = {'factor': factor_name}
            row.update(results)
            report_data.append(row)
        
        df_report = pd.DataFrame(report_data)
        
        # 排序（按IC均值）
        if 'ic_1' in df_report.columns:
            df_report = df_report.sort_values('ic_1', ascending=False)
        
        print("\n" + "=" * 80)
        print("因子分析报告")
        print("=" * 80)
        print(df_report.to_string(max_rows=20))
        
        if save_path:
            df_report.to_csv(save_path, index=False)
            print(f"\n报告已保存到: {save_path}")
        
        return df_report


# 使用示例
def run_factor_research_example():
    """运行因子研究示例"""
    print("=" * 60)
    print("因子研究框架 - 示例")
    print("=" * 60)
    
    # 创建因子研究实例
    researcher = FactorResearch()
    
    # 这里需要加载你的数据
    # 假设我们有一个DataFrame df，包含价格数据
    
    # 示例：创建因子
    print("创建因子...")
    # factors = researcher.create_factors(df)
    
    # 示例：评估因子
    print("评估因子...")
    # for factor_name, factor_series in factors.items():
    #     results = researcher.evaluate_factor(factor_series, df['returns'])
    #     researcher.factor_performance[factor_name] = results
    
    # 生成报告
    # report = researcher.generate_factor_report(researcher.factor_performance)
    
    print("示例完成！")


if __name__ == "__main__":
    run_factor_research_example()