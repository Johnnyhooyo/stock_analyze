# 腾讯股票智能分析系统

自动化股票分析工具，支持多策略回测、超参数优化、信号预测和持仓管理。

## 功能特性

- **多策略支持**: 布林带、MACD、RSI、移动平均线等多种技术指标策略
- **超参数优化**: 自动搜索最优策略参数组合
- **信号预测**: 基于历史数据和策略信号预测未来走势
- **持仓管理**: 根据持仓情况生成交易建议
- **验证分析**: 样本外测试、Walk-Forward 分析
- **热度因子**: 集成 Google Trends 搜索热度数据

## 策略列表

| 策略名称 | 描述 |
|---------|------|
| macd_rsi_trend | MACD + RSI + 热度因子组合 |
| bollinger_rsi_trend | 布林带 + RSI + 热度因子组合 |
| macd_rsi_combo | MACD + RSI 组合 |
| bollinger_breakout | 布林带突破 |
| rsi_reversion | RSI 均值回归 |
| kdj_obv | KDJ + OBV 组合 |
| rsi_obv | RSI + OBV 组合 |

## 项目结构

```
stock_analyze/
├── main.py                 # 主入口
├── config.yaml             # 主配置文件
├── keys.yaml               # 密钥配置（不上传）
├── analyze_factor.py       # 因子分析与回测
├── validate_strategy.py    # 策略验证
├── position_manager.py     # 持仓管理
├── google_trends.py       # Google Trends 热度数据
├── fetch_data.py          # 数据获取
├── visualize.py           # 可视化
├── strategies/            # 策略模块
│   ├── macd_rsi_trend.py
│   ├── bollinger_rsi_trend.py
│   └── ...
└── data/                  # 数据目录
    ├── historical/        # 历史数据
    ├── factors/          # 因子文件
    ├── reports/          # 分析报告
    ├── trends/           # 热度数据
    └── plots/            # 图表
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.yaml`:

```yaml
# 股票代码（港股）
ticker: 0700.hk

# 持仓配置
position_shares: 200      # 持股数量，0表示空仓
position_avg_cost: 530.0  # 平均成本

# 密钥（如需要）
# 编辑 keys.yaml 文件
```

### 3. 运行分析

```bash
python3 main.py
```

### 4. 查看报告

报告保存在 `data/reports/` 目录：
- `report_*.md` - 预测分析报告
- `validation_*.md` - 策略验证报告

## 配置说明

### config.yaml

| 参数 | 说明 | 默认值 |
|------|------|--------|
| ticker | 股票代码 | 0700.hk |
| period | 回测周期 | 5y |
| train_years | 训练年数 | 5 |
| max_tries | 最大参数组合数 | 300 |
| min_return | 验证集最低收益 | 0.03 |
| position_shares | 持股数量 | 0 |
| position_avg_cost | 平均成本 | 0 |

### keys.yaml

API 密钥配置（可选）：

```yaml
# Alpha Vantage API Key
alpha_vantage_key: your_key_here
```

## 输出报告示例

### 预测报告

- 基本信息（股票代码、价格）
- 策略信息（策略名称、参数、信号）
- 收益指标（累计收益、夏普比率等）
- 风险指标（最大回撤、波动率）
- 交易信号（BS点）
- 预测结果（未来N天价格预测）
- 持仓状态与建议

### 验证报告

- 样本外测试（超额收益、夏普比率）
- Walk-Forward 分析（窗口胜率、交易胜率）

## 交易建议逻辑

| 当前持仓 | 信号 | 预测涨幅 | 建议 |
|---------|------|---------|------|
| 有 | 1 | 上涨 | 持有 |
| 有 | 0 | 下跌 | 卖出 |
| 无 | 1 | 上涨 | 买入 |
| 无 | 0 | 下跌 | 观望 |

## 注意事项

- 预测结果仅供参考，不构成投资建议
- 策略历史表现不代表未来收益
- 请根据自身风险承受能力决策
- Google Trends API 有频率限制，会自动降级使用备用方案

## 技术栈

- Python 3.10+
- pandas, numpy - 数据处理
- scikit-learn - 机器学习
- matplotlib - 可视化
- pytrends - Google Trends
- yfinance - 行情数据

## License

MIT
