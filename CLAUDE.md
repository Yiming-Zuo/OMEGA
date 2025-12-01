# OMEGA AI 协作指南

## 项目概述
OMEGA（OpenMM Enhanced-sampling General Architecture）是一个面向分子模拟的通用增强采样框架。

## 开发阶段
1. **阶段一**：丙氨酸二肽（Alanine Dipeptide）的测试采样 ← 当前阶段
2. **阶段二**：溶剂分支采样实现
3. **阶段三**：复合物分支采样实现

## 目录结构
- `data/` - 输入数据（按体系类型组织）
- `docs/` - 项目文档
- `notebooks/` - Jupyter notebooks
- `outputs/` - 运行输出（系统文件、轨迹等）
- `results/` - 分析结果（figures/, reports/）
- `scripts/` - Python 脚本
- `tests/` - 测试代码

## 工作流程
```
01_prepare_system.py → 02_run_rest2_hremd.py → 03_analyze_results.py → 04_mbar_reweighting.py
```

## 编码规范
- 使用 UTF-8 编码，直接使用中文注释
- 遵循 PEP 8 风格指南
- 使用 conventional commits 规范提交代码

## Emoji 规则
- 禁止在代码和文档中使用 emoji 或特殊图标符号
- 使用纯文本标记替代：[OK], [WARN], [FAIL] 等
- Markdown 文档中避免装饰性 emoji，保持简洁专业
