macOS 安装说明（针对常见依赖安装失败问题）

先决条件
- macOS (Apple Silicon 或 Intel) 兼容 Python 3.8+。
- 推荐使用 venv 或 conda/miniforge 来避免系统 Python 冲突。

步骤（推荐）
1) 建议使用 Miniforge (尤其是 Apple Silicon M1/M2).
   - Miniforge 下载: https://github.com/conda-forge/miniforge
   - 安装并创建环境示例:
     ```bash
     # 安装后
     conda create -n stock_analyze python=3.10 -y
     conda activate stock_analyze
     ```

2) 如果使用系统 pip:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     python -m pip install --upgrade pip setuptools wheel
     ```

3) 尝试安装松耦合依赖（遇到问题优先试这个文件）:
     ```bash
     pip install -r requirements-relaxed.txt
     ```

4) 如果仍需精确版本或报错，使用以下建议：
   - numpy/pandas/scikit-learn 常因二进制 wheel 问题失败，请优先用 conda 安装：
     ```bash
     conda install numpy pandas scikit-learn matplotlib -c conda-forge
     pip install yfinance PyYAML joblib
     ```
   - 对于 Apple Silicon (M1/M2):
     - 确保使用的 Python 与架构一致（arm64 vs x86_64）。
     - 如果遇到编译错误，优先尝试 conda-forge 的预编译包。

5) 常见单包问题参考：
   - scikit-learn: 推荐使用 conda install scikit-learn 或 pip 安装 wheel（确保 pip 升级到最新）。
   - numpy: pip install numpy 有时需要先安装 wheel 或 Xcode 命令行工具；conda 更可靠。

6) 最后，运行项目 smoke test:
   ```bash
   python3 smoke_test.py
   ```

如果安装时报错，请把完整的 pip/conda 错误输出贴给我，我会给出具体解决方案（比如使用不同版本或 conda 安装命令）。

