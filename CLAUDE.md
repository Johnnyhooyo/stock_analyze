# Claude Code 记忆文件

## 项目概述
腾讯控股 (0700.hk) 股票智能分析系统

## 项目结构
```
stock_analyze/
├── main.py                 # 主入口
├── config.yaml             # 主配置文件
├── keys.yaml               # 密钥配置（不上传）
├── analyze_factor.py       # 因子分析与回测
├── validate_strategy.py    # 策略验证
├── backtest_vectorbt.py    # Vectorbt回测引擎
├── position_manager.py     # 持仓管理
├── sentiment_analysis.py   # 情感分析
├── google_trends.py        # Google Trends热度数据
├── feishu_notify.py       # 飞书通知
├── fetch_data.py           # 数据获取
├── visualize.py           # 可视化
├── strategies/             # 策略模块
│   ├── macd_rsi_trend.py
│   ├── bollinger_rsi_trend.py
│   └── ...
└── data/                  # 数据目录
```

## 配置文件

### config.yaml
```yaml
ticker: 0700.hk           # 股票代码
backtest_engine: vectorbt  # 回测引擎: native / vectorbt
strategies:                # 运行的策略列表
  - macd_rsi_trend
  - bollinger_rsi_trend
min_return: 0.03         # 验证集最低收益阈值
max_tries: 300            # 参数搜索次数
```

### keys.yaml
```yaml
alpha_vantage_key: null
feishu_webhook: https://open.feishu.cn/...
```

## 关键逻辑

### 1. 回测引擎选择
- `config.yaml` 中 `backtest_engine: native` 或 `vectorbt`
- Vectorbt 与 Native 逻辑略有差异，结果不完全相同
- Native 更适合当前策略逻辑

### 2. 数据更新逻辑 (fetch_data.py)
- 检查本地数据最后交易日
- 超过2天自动更新（考虑周末）
- 港股：周一周二按上周五计算

### 3. 策略列表
- 在 `config.yaml` 的 `strategies` 中配置
- 当前策略: macd_rsi_trend, bollinger_rsi_trend

### 4. 数据持久化
- 热度数据: data/trends/tencent_trends.csv
- 情感数据: data/sentiment/sentiment_cache.csv
- 都已添加到 .gitignore

### 5. 飞书通知
- Webhook 在 keys.yaml 中配置
- 自动发送分析报告和验证结果

## 运行命令
```bash
python3 main.py                    # 完整分析
python3 main.py --skip-search     # 跳过搜索，使用已有因子
python3 main.py --n-days 5        # 预测天数
```

## 注意事项
1. 所有 API 密钥放在 keys.yaml，不提交到 git
2. Vectorbt 回测与 Native 有差异，当前项目用 Native 效果更好
3. 情感分析和热度数据都有本地缓存，避免频繁调用 API
4. 报告自动保存到 data/reports/

## Git 状态
- 已初始化 git 仓库
- 已推送到 github.com/Johnnyhooyo/stock_analyze.git
