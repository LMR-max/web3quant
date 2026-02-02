# Machine Learning Research Pipeline

基线研究流水线：

- 从 parquet 读取行情
- 构造 K 线因子 + 滚动特征
- 预测 t+1h 或 t+1d 收益方向（涨/跌/区间）与收益回归
- Walk-Forward 防止 data leakage
- 产出评分 → 分层 → 简单回测

## 默认配置

见 config.py，默认读取：

- data_manager_storage/spot/binance/BTC_USDT/1h/ohlcv_merged.parquet

## 运行

在项目根目录执行：

- python -m machine_learning.run_pipeline

## 实验矩阵

在 config.py 的 experiments 中配置：

- ML 基线（OLS/Ridge/Lasso/RF/XGBoost/LightGBM）
- PCA 版本
- IC_weight 版本
- LSTM / Transformer（需要 torch）

## 输出

- machine_learning/outputs/<exp_name>/metrics.csv
- machine_learning/outputs/<exp_name>/metrics.json
- machine_learning/outputs/<exp_name>/preds_*.parquet
- machine_learning/outputs/<exp_name>/backtest_summary.json
- machine_learning/outputs/summary.json（前端可直接读取）

## 数据与产物管理（重要）

- `machine_learning/outputs/` 属于实验产物，默认 **不入库**（已在 .gitignore 中忽略）。
- 上游数据路径 `data_manager_storage/` 同样为运行产物，避免提交到 Git。
- 如需分享结果，建议导出到 `exports/` 并进行脱敏处理后再提交。

## 可选依赖

若需要 XGBoost/LightGBM/深度模型，请安装：

- xgboost
- lightgbm
- torch
