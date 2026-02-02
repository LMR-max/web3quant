# Web3Quant - 加密货币数据系统与量化研究框架

面向加密市场的数据获取、存储与可视化平台，并内置因子研究与机器学习模块，支持从数据 → 特征 → 模型 → 回测的研究闭环。

---

## 你可以用它做什么

- 一键拉取多市场行情数据（Spot/Swap/Future/Option/Margin/OnChain/Social）
- Web UI 可视化与管理数据、导出 CSV
- Python API 便捷接入：Fetcher + DataManager
- 量化研究与因子验证（alphagen_style、factor_research、machine_learning）
- 回测与策略原型验证

---

## 快速开始（3 步）

### 1) 配置本地密钥（必做）

复制模板并填写密钥（仅保存在本机，勿提交到 Git）：

- 复制 crypto_data_system/local_secrets.example.json 为 crypto_data_system/local_secrets.json
- 填入你的 API Key（如 Dune、TheGraph、Etherscan、X 等）

程序启动时会自动读取本地私有文件或环境变量（优先本地私有文件）。

### 2) 安装依赖

项目基于 Python，建议使用虚拟环境：

- 依赖核心：flask、flask-cors
- 如需扩展研究能力，可按需安装数据与模型相关依赖

### 3) 启动 Web 应用

- 运行：python run_web.py
- 浏览器访问：<http://localhost:5000>

---

## Web UI 功能概览

- 数据获取：交易所/市场选择、交易对搜索、时间范围与粒度配置、批量拉取
- 数据可视化：K 线、成交量、对比分析、技术指标
- 数据管理：自动保存、导出 CSV、存储监控
- 系统监控：运行状态、缓存与日志

---

## Python API 简要示例

> 以下为示意用法，详见文档和模块注释。

- Fetcher：获取行情、K 线、订单簿、快照等
- DataManager：批量拉取、缓存管理、持久化保存

---

## 目录结构（精简）

- crypto_data_system/：核心数据系统（Fetcher、存储、工具）
- web_app.py / run_web.py：Web API 与启动脚本
- web_static/、web_templates/：前端资源
- alphagen_style/：因子表达式与自动化实验
- factor_research/：因子研究
- machine_learning/：特征、标签、模型与训练流程
- quant_backtest/：回测组件

---

## 数据与缓存说明（重要）

- 运行数据与缓存默认不入库：data/、data_manager_storage/、logs/、models/ 等
- 本地私钥文件已被 .gitignore 忽略

---

## 常用命令

- 启动 Web：python run_web.py
- 系统检查：python health_check.py

---

## 适用人群

- 需要系统化获取加密市场数据的研究者
- 构建量化因子、模型和回测验证流程的团队
- 需要快速可视化与数据管理的个人/团队

---

## 进一步阅读

- 详尽文档：PROJECT.md
- 研究框架：FRAMEWORK.md

---

## 免责声明

本项目仅用于研究与教育目的，不构成任何投资建议。使用者需自行承担风险。
