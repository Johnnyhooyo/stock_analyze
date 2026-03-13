# 腾讯股票分析技能

## 功能
1. 下载腾讯股票历史数据
2. 开发并测试交易因子
3. 回测验证收益率
4. 可视化交易信号

## 使用说明
1. 安装依赖: `pip install -r requirements.txt`
2. 配置股票代码和时间范围: 修改 `config.yaml`
3. 运行完整流程: `python main.py`

## 文件结构
- `fetch_data.py`: 数据获取模块
- `analyze_factor.py`: 因子分析与回测
- `visualize.py`: 交易信号可视化
- `main.py`: 主执行脚本
- `data/`: 本地数据存储
  - `historical/`: 原始历史数据
  - `factors/`: 有效因子存储