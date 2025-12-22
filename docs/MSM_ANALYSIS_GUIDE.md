# MSM 马尔科夫状态模型分析指南

本文档介绍如何使用 Deeptime 进行自动化亚稳态识别和动力学分析。

## 概述

MSM（Markov State Model）分析是一种数据驱动的方法，用于从分子动力学轨迹中识别亚稳态和计算动力学性质。与手工定义的构象划分相比，MSM 具有以下优势：

- **数据驱动**：自动从轨迹数据中学习状态划分
- **动力学信息**：提供隐含时间尺度、平均首次通过时间等
- **自由能估计**：基于平稳分布精确计算
- **可验证性**：通过 Chapman-Kolmogorov 测试验证模型质量

## 安装依赖

MSM 分析需要安装 Deeptime：

```bash
conda install -c conda-forge deeptime
```

Deeptime 是 PyEMMA 的继任者，提供了更现代的 API 设计（类似 scikit-learn）。

文档：https://deeptime-ml.github.io/latest/index.html

## 快速开始

### 基础用法

```bash
# 运行基础分析 + MSM
python 03_analyze_results.py --solvent vacuum --msm

# 或者 GBSA 模式
python 03_analyze_results.py --solvent gbsa --msm
```

### 自定义参数

```bash
python 03_analyze_results.py --solvent vacuum --msm \
    --tica-lag 100 \
    --n-clusters 150 \
    --msm-lag 50 \
    --n-metastable 5
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--msm` | - | 启用 MSM 分析（开关） |
| `--tica-lag` | 50 | TICA 降维的 lag time（帧数） |
| `--n-clusters` | 100 | k-means 聚类的微观状态数目 |
| `--msm-lag` | 自动 | MSM 构建的 lag time（帧数），默认通过 ITS 分析自动选择 |
| `--n-metastable` | 4 | PCCA+ 识别的亚稳态数目 |

## 工作流程

MSM 分析包含以下步骤：

```
1. 特征提取
   └─ 计算 φ/ψ 二面角，使用 sin/cos 编码消除周期性

2. TICA 降维
   └─ 时间独立成分分析，提取慢模式

3. k-means 聚类
   └─ 生成微观状态（离散轨迹）

4. Lag Time 选择
   └─ 通过隐含时间尺度（ITS）分析自动选择

5. MSM 构建
   └─ 估计转移概率矩阵
   └─ Chapman-Kolmogorov 测试验证

6. PCCA+ 亚稳态识别
   └─ 将微观状态聚合为亚稳态

7. 动力学分析
   └─ 计算隐含时间尺度、MFPT、自由能
```

## 输出文件

MSM 分析结果保存在 `results/.../msm/` 目录下：

### 图表文件

| 文件 | 说明 |
|------|------|
| `tica_projection.png` | TICA 投影散点图，按亚稳态和时间着色 |
| `implied_timescales.png` | 隐含时间尺度 vs lag time，用于验证 MSM 收敛性 |
| `cktest.png` | Chapman-Kolmogorov 测试图，验证马尔科夫性 |
| `free_energy_surface.png` | φ/ψ 和 TICA 空间的自由能面 |
| `metastable_states.png` | 各亚稳态在 Ramachandran 图中的分布 |
| `transition_network.png` | 亚稳态间的转换网络，显示 MFPT |
| `metastable_timeline.png` | 亚稳态随时间演化图 |

### 数据文件

| 文件 | 说明 |
|------|------|
| `kinetics_summary.csv` | 动力学量汇总（时间尺度、自由能、MFPT） |
| `msm_model.pkl` | 完整的 MSM 分析结果（可用于后续分析） |

## 参数选择指南

### TICA Lag Time (`--tica-lag`)

TICA 的 lag time 决定了降维时关注的时间尺度。

- **建议**：设置为预期最慢运动时间尺度的 1/10 到 1/5
- **丙氨酸二肽**：50-100 帧（1-2 ns @ 20 ps/帧）
- **太小**：可能捕获快速波动而非慢模式
- **太大**：统计精度下降

### 聚类数目 (`--n-clusters`)

微观状态的数目影响 MSM 的精度和稳定性。

- **建议起步值**：100
- **验证方法**：比较不同聚类数的隐含时间尺度
- **太少 (<30)**：丢失精细动力学信息
- **太多 (>500)**：统计噪声增大

### MSM Lag Time (`--msm-lag`)

MSM 的 lag time 是最关键的参数，决定了模型的马尔科夫性。

- **自动选择**：默认通过 ITS 分析选择时间尺度收敛的最小 lag
- **手动指定**：根据 `implied_timescales.png` 图选择
- **选择原则**：ITS 曲线开始进入平台区的最小 lag

### 亚稳态数目 (`--n-metastable`)

PCCA+ 识别的亚稳态数目。

- **丙氨酸二肽典型值**：3-5 个
- **判断方法**：
  - 查看隐含时间尺度的 gap（第 k 和 k+1 之间有显著间隔 → k+1 个亚稳态）
  - 结合物理直觉（C7eq, C7ax, αR, PII, β 等）

## 结果解读

### 隐含时间尺度图 (implied_timescales.png)

- X 轴：lag time（帧数）
- Y 轴：隐含时间尺度（ps）
- **好的 MSM**：各条曲线在某个 lag 后趋于平稳
- **红色虚线**：选定的 MSM lag time

### Chapman-Kolmogorov 测试 (cktest.png)

- 比较 MSM 预测（蓝线）和直接估计（红点）
- **好的 MSM**：两者吻合良好
- **偏差大**：需要增大 MSM lag time

### 自由能面 (free_energy_surface.png)

- 颜色表示自由能（kcal/mol）
- 深蓝色区域为稳定构象
- 左图：φ/ψ 空间
- 右图：TICA 空间

### 转换网络 (transition_network.png)

- 节点大小：亚稳态占比
- 箭头粗细：与 MFPT 成反比（快转换 = 粗箭头）
- 标注：平均首次通过时间

## Python API 使用

除了命令行，你也可以在 Python 中直接使用 MSM 模块：

```python
from msm_analysis import MSMAnalyzer, MSMAnalysisResult
from msm_visualization import MSMVisualizer
from pathlib import Path

# 初始化分析器
analyzer = MSMAnalyzer(
    trajectory_path="outputs/.../md/trajectory.dcd",
    topology_path="outputs/.../system.pdb",
    timestep_ps=20.0,
)

# 运行分析
result = analyzer.run_full_analysis(
    tica_lag=50,
    n_clusters=100,
    msm_lag=None,  # 自动选择
    n_metastable=4,
)

# 可视化
output_dir = Path("results/.../msm")
output_dir.mkdir(parents=True, exist_ok=True)

visualizer = MSMVisualizer(result, output_dir)
visualizer.plot_all()

# 保存和加载
result.save(output_dir / "msm_model.pkl")
loaded = MSMAnalysisResult.load(output_dir / "msm_model.pkl")

# 访问结果
print(f"隐含时间尺度: {result.kinetics['timescales_ps'][:3]} ps")
print(f"亚稳态占比: {result.metastable_probs}")
print(f"自由能: {result.kinetics['free_energies_kcal']} kcal/mol")

# 获取每帧的亚稳态标签
meta_labels = result.get_metastable_labels()
```

## Deeptime vs PyEMMA

本模块使用 Deeptime 替代了已停止维护的 PyEMMA。主要 API 差异：

| 功能 | PyEMMA | Deeptime |
|------|--------|----------|
| TICA | `pyemma.coordinates.tica()` | `deeptime.decomposition.TICA` |
| k-means | `pyemma.coordinates.cluster_kmeans()` | `deeptime.clustering.KMeans` |
| MSM | `pyemma.msm.estimate_markov_model()` | `deeptime.markov.msm.MaximumLikelihoodMSM` |
| PCCA+ | `msm.pcca()` | `msm.pcca()` 返回 `PCCAModel` |

Deeptime 采用 scikit-learn 风格的 Estimator/Model 分离设计：

```python
# Deeptime 风格
estimator = TICA(lagtime=10)
model = estimator.fit(data).fetch_model()
output = model.transform(data)
```

## 常见问题

### Q: MSM 分析失败，提示 "需要安装 Deeptime"

Deeptime 需要通过 conda 安装：

```bash
conda install -c conda-forge deeptime
```

也可以使用 pip：

```bash
pip install deeptime
```

### Q: 隐含时间尺度不收敛

可能原因：
1. 轨迹太短，采样不足
2. 聚类数目太少或太多
3. 存在未连通的状态空间区域

解决方法：
1. 延长模拟时间
2. 尝试不同的聚类数
3. 检查轨迹是否覆盖了所有重要构象

### Q: CK 测试偏差大

可能原因：
1. MSM lag time 太小，马尔科夫性不满足
2. 采样不足

解决方法：
1. 增大 `--msm-lag` 参数
2. 延长模拟时间

### Q: 亚稳态数目如何确定？

1. 查看 `implied_timescales.png`，寻找时间尺度的 gap
2. 结合物理先验（丙氨酸二肽通常 3-5 个亚稳态）
3. 尝试不同数目，检查 `metastable_states.png` 是否有意义

## 参考文献

1. Prinz, J.-H. et al. (2011). Markov models of molecular kinetics. J. Chem. Phys. 134, 174105.
2. Perez-Hernandez, G. et al. (2013). Identification of slow molecular order parameters for Markov model construction. J. Chem. Phys. 139, 015102.
3. Hoffmann, M. et al. (2021). Deeptime: a Python library for machine learning dynamical models from time series data. Machine Learning: Science and Technology. DOI: 10.1088/2632-2153/ac3de0
