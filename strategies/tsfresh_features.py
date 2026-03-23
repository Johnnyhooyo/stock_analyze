"""
tsfresh 特征提取模块
------------------------------------------------------
提供基于 tsfresh 的自动时间序列特征提取功能。

主要功能:
  - 从 OHLCV 数据中自动提取数百个时间序列特征
  - 结合传统技术指标与 tsfresh 特征
  - 特征选择以减少维度并提升模型效果

依赖:
  - tsfresh
  - pyts (用于额外的时间序列特征)

用法:
  from strategies.tsfresh_features import extract_tsfresh_features, TSFreshFeatureExtractor

  extractor = TSFreshFeatureExtractor()
  features = extractor.extract(df, window_size=20)
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from functools import lru_cache
import warnings

# 忽略 tsfresh 内部的 FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning, module='tsfresh')

# tsfresh 相关
try:
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_selection import select_features
    from tsfresh.feature_extraction.settings import (
        MinimalFCParameters,
        EfficientFCParameters,
        ComprehensiveFCParameters,
    )
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False
    extract_features = None
    impute = None
    select_features = None
    MinimalFCParameters = None
    EfficientFCParameters = None
    ComprehensiveFCParameters = None

# 备用: 使用 pyts 的特征 (如果可用)
try:
    from pyts.feature_extraction import GASF, TAOO
    PYTS_AVAILABLE = True
except ImportError:
    PYTS_AVAILABLE = False
    GASF = None
    TAOO = None


class TSFreshFeatureExtractor:
    """
    基于 tsfresh 的特征提取器

    将原始 OHLCV 数据转换为 tsfresh 所需格式，
    提取时间序列特征，并进行特征选择。
    """

    def __init__(
        self,
        extraction_level: str = "efficient",
        feature_selection: bool = True,
        max_features: int = 200,
    ):
        """
        Args:
            extraction_level: 特征提取级别
                - "minimal": 少量基础特征
                - "efficient": 平衡效率和覆盖率 (默认)
                - "comprehensive": 最大特征集
            feature_selection: 是否进行特征选择
            max_features: 最大特征数量 (用于特征选择后)
        """
        self.extraction_level = extraction_level
        self.feature_selection = feature_selection
        self.max_features = max_features
        self.feature_names_: Optional[List[str]] = None
        self.selected_features_: Optional[List[str]] = None
        self._fc_settings = self._get_fc_settings()

    def _get_fc_settings(self):
        """获取 tsfresh 特征提取配置"""
        if not TSFRESH_AVAILABLE:
            return None

        if self.extraction_level == "minimal":
            return MinimalFCParameters()
        elif self.extraction_level == "comprehensive":
            return ComprehensiveFCParameters()
        else:  # efficient
            return EfficientFCParameters()

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将 OHLCV 数据转换为 tsfresh 所需格式

        tsfresh 需要:
        - id: 标识符列
        - time: 时间索引列
        - 其他: 待提取特征的数值列
        """
        df = df.copy()

        # 确保索引是日期时间
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        # 重置索引，生成 tsfresh 格式
        df = df.reset_index()
        # 处理索引列名（可能是 'index', 'date' 或其他）
        if 'index' in df.columns:
            df.rename(columns={'index': 'datetime'}, inplace=True)
        elif df.columns[0] not in ['Open', 'High', 'Low', 'Close', 'Volume']:
            # 第一列是日期时间但不是标准名称
            df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
        # 否则假设第一列就是日期时间

        # 创建 id 列 (每个交易日一个唯一的 id)
        # 对于滚动窗口场景，我们使用日期作为 id
        df['id'] = df['datetime'].dt.strftime('%Y-%m-%d')

        # 确保数值列
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 只保留需要的列
        cols_to_keep = ['id', 'datetime'] + [c for c in numeric_cols if c in df.columns]
        df = df[cols_to_keep].dropna()

        return df

    def _extract_rolling_features(
        self,
        df: pd.DataFrame,
        window_size: int = 20,
    ) -> pd.DataFrame:
        """
        使用滚动窗口提取 tsfresh 特征（向量化高效版本）

        对于每个时间点，使用过去 window_size 天的数据提取特征。
        使用向量化计算替代逐窗口调用，极大提升性能。
        """
        if not TSFRESH_AVAILABLE:
            return pd.DataFrame()

        df = df.copy()

        # 确保按日期排序
        df = df.sort_values('datetime').reset_index(drop=True)

        # ===== 核心优化：使用向量化滚动计算替代逐窗口调用 =====
        # tsfresh EfficientFCParameters 中的大部分特征都可以用 pandas rolling 向量化计算
        features = pd.DataFrame(index=df.index)

        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        available_cols = [c for c in numeric_cols if c in df.columns]

        for col in available_cols:
            series = df[col]

            # 1. 滚动统计特征（均值、标准差、最值、偏度、峰度）
            for w in [window_size]:
                # 滚动均值
                features[f'{col}_rolling_mean_{w}'] = series.rolling(w).mean()
                # 滚动标准差
                features[f'{col}_rolling_std_{w}'] = series.rolling(w).std()
                # 滚动最小值
                features[f'{col}_rolling_min_{w}'] = series.rolling(w).min()
                # 滚动最大值
                features[f'{col}_rolling_max_{w}'] = series.rolling(w).max()
                # 滚动中位数
                features[f'{col}_rolling_median_{w}'] = series.rolling(w).median()
                # 滚动偏度
                features[f'{col}_rolling_skew_{w}'] = series.rolling(w).skew()
                # 滚动峰度
                features[f'{col}_rolling_kurt_{w}'] = series.rolling(w).kurt()

            # 2. 收益率相关特征
            ret = series.pct_change()
            for w in [window_size]:
                # 滚动收益率标准差（波动率）
                features[f'{col}_volatility_{w}'] = ret.rolling(w).std()
                # 滚动最大收益
                features[f'{col}_max_return_{w}'] = ret.rolling(w).max()
                # 滚动最小收益
                features[f'{col}_min_return_{w}'] = ret.rolling(w).min()
                # 滚动收益率均值
                features[f'{col}_mean_return_{w}'] = ret.rolling(w).mean()

            # 3. 位置特征
            for w in [window_size]:
                rolling_mean = series.rolling(w).mean()
                rolling_std = series.rolling(w).std().replace(0, np.nan)
                features[f'{col}_zscore_{w}'] = (series - rolling_mean) / rolling_std

            # 4. 变化率特征
            for w in [window_size]:
                # 价格变化率
                features[f'{col}_pct_change_{w}'] = series.pct_change(w)
                # 与滚动均值的偏离
                rolling_mean = series.rolling(w).mean()
                features[f'{col}_diff_ma_{w}'] = series - rolling_mean

        # 5. 跨列特征（价格范围、成交量比率等）
        if 'High' in df.columns and 'Low' in df.columns:
            hl_range = df['High'] - df['Low']
            for w in [window_size]:
                features[f'hl_range_rolling_mean_{w}'] = hl_range.rolling(w).mean()
                features[f'hl_range_rolling_std_{w}'] = hl_range.rolling(w).std()

        if 'Close' in df.columns and 'Open' in df.columns:
            co_change = df['Close'] - df['Open']
            for w in [window_size]:
                features[f'co_change_rolling_mean_{w}'] = co_change.rolling(w).mean()

        if 'Volume' in df.columns and 'Close' in df.columns:
            # 成交量与价格变化的关系
            vol_ret_corr = df['Volume'].rolling(window_size).corr(df['Close'].pct_change())
            features[f'volume_close_corr_{window_size}'] = vol_ret_corr

        # 去除全 NaN 列
        features = features.dropna(axis=1, how='all')

        # 设置索引为 datetime 列的值（确保是 datetime 类型以便与 df.index 对齐）
        features.index = pd.to_datetime(df['datetime']).values
        features.index.name = 'datetime'

        return features

    def extract(
        self,
        df: pd.DataFrame,
        window_size: int = 20,
    ) -> pd.DataFrame:
        """
        从时间序列数据中提取 tsfresh 特征

        Args:
            df: 包含 OHLCV 数据的 DataFrame，索引为日期
            window_size: 滚动窗口大小

        Returns:
            特征 DataFrame，索引为日期
        """
        if not TSFRESH_AVAILABLE:
            raise ImportError(
                "tsfresh not installed. Install with: pip install tsfresh"
            )

        # 准备数据格式
        df_tsfresh = self._prepare_dataframe(df)

        if len(df_tsfresh) < window_size:
            return pd.DataFrame()

        # 提取滚动窗口特征
        features = self._extract_rolling_features(df_tsfresh, window_size)

        self.feature_names_ = list(features.columns)

        return features

    def extract_with_selection(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        window_size: int = 20,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        提取特征并进行相关性选择

        Args:
            df: 原始 OHLCV 数据
            y: 标签序列 (与 df 的日期索引对齐)
            window_size: 滚动窗口大小

        Returns:
            (selected_features, selected_feature_names)
        """
        if not TSFRESH_AVAILABLE or select_features is None:
            raise ImportError("tsfresh not installed")

        # 提取特征
        features = self.extract(df, window_size)

        if features.empty:
            return pd.DataFrame(), []

        # 对齐 y
        common_idx = features.index.intersection(y.index)
        if len(common_idx) == 0:
            return features, list(features.columns)

        features_aligned = features.loc[common_idx]
        y_aligned = y.loc[common_idx].values  # convert to numpy array for select_features

        # 使用 select_features 进行特征选择
        try:
            # tsfresh select_features 要求无 NaN，先用 impute 处理
            features_aligned = impute(features_aligned)

            # P3-C: 限制 n_jobs，避免 Optuna 多进程下与 tsfresh 并行叠加导致进程爆炸
            import os as _os
            _tsfresh_jobs = max(1, (_os.cpu_count() or 2) // 2)
            selected = select_features(
                features_aligned,
                y_aligned,
                fdr_level=0.05,
                show_warnings=False,
                n_jobs=_tsfresh_jobs,
            )

            self.selected_features_ = list(selected.columns)
            return selected, list(selected.columns)

        except Exception as e:
            # 如果特征选择失败，返回所有特征
            print(f"  [tsfresh] 特征选择失败: {e}")
            return features_aligned, list(features_aligned.columns)


def extract_tsfresh_features(
    df: pd.DataFrame,
    window_sizes: List[int] = [10, 20, 30],
    extraction_level: str = "efficient",
    with_selection: bool = True,
    y: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    便捷函数: 从 OHLCV 数据中提取多窗口 tsfresh 特征

    Args:
        df: OHLCV 数据
        window_sizes: 滚动窗口大小列表
        extraction_level: 特征提取级别
        with_selection: 是否进行特征选择
        y: 标签 (用于特征选择)

    Returns:
        (features_df, feature_names)
    """
    if not TSFRESH_AVAILABLE:
        print("  [tsfresh] tsfresh 未安装，跳过特征提取")
        return pd.DataFrame(), []

    all_features = []
    feature_names = []

    for window in window_sizes:
        print(f"  [tsfresh] 提取窗口={window} 的特征...")
        extractor = TSFreshFeatureExtractor(
            extraction_level=extraction_level,
            feature_selection=False,  # 先不选择，合并后再选
            max_features=200,
        )

        features = extractor.extract(df, window_size=window)

        if not features.empty:
            # 添加窗口后缀避免列名冲突
            features.columns = [f"{c}_w{window}" for c in features.columns]
            all_features.append(features)

    if not all_features:
        return pd.DataFrame(), []

    # 合并所有窗口特征
    combined = pd.concat(all_features, axis=1)

    # 去除全 NaN 列
    combined = combined.dropna(axis=1, how='all')

    print(f"  [tsfresh] 共提取 {combined.shape[1]} 个原始特征")

    # 如果有标签且需要进行特征选择
    if with_selection and y is not None and not combined.empty:
        try:
            common_idx = combined.index.intersection(y.index)
            if len(common_idx) > 50:  # 需要足够的数据进行特征选择
                from tsfresh.feature_selection import select_features
                from tsfresh.utilities.dataframe_functions import impute

                features_aligned = combined.loc[common_idx]
                # ⚠️ 修复：保持 pd.Series 类型并重置索引，避免 tsfresh 内部
                # 对 numpy array 做布尔运算时出现 "Series ambiguous" 错误
                y_aligned = y.loc[common_idx]
                if not isinstance(y_aligned, pd.Series):
                    y_aligned = pd.Series(y_aligned, index=common_idx)
                y_aligned = y_aligned.reset_index(drop=True)
                features_aligned = impute(features_aligned)
                features_for_sel  = features_aligned.reset_index(drop=True)

                selected = select_features(
                    features_for_sel,
                    y_aligned,
                    fdr_level=0.05,
                    show_warnings=False,
                    n_jobs=0,
                )

                if not selected.empty and len(selected.columns) > 0:
                    if len(selected.columns) > 200:
                        variances = features_for_sel.var().sort_values(ascending=False)
                        top_cols = variances.head(200).index.tolist()
                        selected = features_for_sel[top_cols]

                    # 恢复原始索引
                    selected.index = common_idx
                    combined = features_aligned[selected.columns]
                    print(f"  [tsfresh] 特征选择后保留 {len(selected.columns)} 个特征")
                else:
                    print(f"  [tsfresh] 无显著特征，选择前 100 个")
                    combined = features_aligned.iloc[:, :100]
                    # fix #13: 确保空选择分支索引与 common_idx 对齐
                    if not combined.index.equals(common_idx):
                        combined.index = common_idx
        except Exception as e:
            print(f"  [tsfresh] 特征选择异常: {e}")

    return combined, list(combined.columns)


# ============================================================
# 备用: 不依赖 tsfresh 的简化版特征提取
# ============================================================

def extract_simple_ts_features(
    df: pd.DataFrame,
    windows: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    简化版时间序列特征提取 (不依赖 tsfresh)

    提取基本统计特征:
    - 移动均值
    - 移动标准差
    - 最小值/最大值
    - 偏度/峰度
    - 收益率特征
    """
    df = df.copy()
    features = pd.DataFrame(index=df.index)

    # 基本收益率
    for col in ['Close', 'Volume']:
        if col not in df.columns:
            continue

        # 日收益率
        if col == 'Close':
            ret = df[col].pct_change()
            features[f'{col}_return'] = ret

            # 不同窗口的收益
            for w in windows:
                features[f'{col}_ret_{w}d'] = df[col].pct_change(w)

        # 成交量变化
        if col == 'Volume':
            features[f'{col}_change'] = df[col].pct_change()

        # 滚动统计特征
        for w in windows:
            # 均值
            features[f'{col}_ma_{w}'] = df[col].rolling(w).mean()
            # 标准差
            features[f'{col}_std_{w}'] = df[col].rolling(w).std()
            # 最小/最大值
            features[f'{col}_min_{w}'] = df[col].rolling(w).min()
            features[f'{col}_max_{w}'] = df[col].rolling(w).max()
            # 偏度和峰度
            features[f'{col}_skew_{w}'] = df[col].rolling(w).skew()
            features[f'{col}_kurt_{w}'] = df[col].rolling(w).kurt()

            # 位置特征 (当前价格相对均值的百分比)
            features[f'{col}_pct_ma_{w}'] = (df[col] - df[col].rolling(w).mean()) / df[col].rolling(w).std()

            # 成交量的滚动均值比
            if col == 'Volume':
                features[f'{col}_ratio_{w}'] = df[col] / df[col].rolling(w).mean()

    # 价格范围特征
    if all(c in df.columns for c in ['High', 'Low', 'Close']):
        for w in windows:
            # 日内范围
            features[f'hl_range_{w}'] = (df['High'] - df['Low']).rolling(w).mean()
            # 收盘相对开盘的变化
            if 'Open' in df.columns:
                features[f'co_change_{w}'] = (df['Close'] - df['Open']).rolling(w).mean()

    # 价格动量
    for w in windows:
        features[f'momentum_{w}'] = df['Close'] / df['Close'].shift(w) - 1

    # 波动率
    ret = df['Close'].pct_change()
    for w in windows:
        features[f'volatility_{w}'] = ret.rolling(w).std()

    # 去除全 NaN 列
    features = features.dropna(axis=1, how='all')

    return features
