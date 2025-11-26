# OMEGA

OMEGA（OpenMM Enhanced-sampling General Architecture）是一个面向分子模拟的通用增强采样框架，基于 OpenMM 构建，计划支持多种增强采样策略（如 REST2、SST2和生成式方法 等）在不同体系条件下的灵活组合与扩展。OMEGA 专注于为热力学循环中的溶剂分支与复合物分支生成高质量的构象采样数据，可作为自由能计算、构象再加权与生成路径建模的统一前端。

结合 FReD 数据构建工具，OMEGA 生成的增强采样轨迹可通过 MBAR 映射到无偏平衡系综，为基于概率流的相对结合自由能估计提供标准化、可扩展、可复现的数据基础。

> **当前状态**：初步开发阶段，正在进行丙氨酸二肽（Alanine Dipeptide）的测试采样。

## 开发阶段

1. **阶段一**：丙氨酸二肽（Alanine Dipeptide）的测试采样 ← 当前阶段
2. **阶段二**：溶剂分支采样实现
3. **阶段三**：复合物分支采样实现

## 数据流动
```
    输入结构                系统准备                  HREMD 采样               结果分析       
   (溶质 PDB)       (溶剂化+参数化+REST2缩放)         (多副本交换)           (接受率/能量/构象) 
        ↓                    ↓                          ↓                       ↓          
┌──────────────┐    ┌──────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│  ligand.pdb  │    │ system.xml(构型)  │     │ samples.arrow(能量) │     │ 接受率矩阵     │ 
│              │ →  │ system.pdb(拓扑)  │  →  │    r*.dcd(轨迹)     │  →  │ 能量收敛       │
│              │    │ topology.pkl(力场)│     │    checkpoint      │     │ Ramachandran  │
└──────────────┘    └──────────────────┘     └─────────────────────┘     └──────────────┘
                     01_prepare_system          02_run_rest2_hremd      03_analyze_results 
1. 溶剂化（Solvation）
	- 水模型：TIP3P
	- 盒子：cube、10Å padding
2. 参数化（Parameterization）
	- 力场：Amber14
	- 键、角、二面角
	- 非键合：电荷、LJ参数
3. REST2缩放（Scaling）
	- 选择溶质分子
	- 缩放二面角
	- 缩放非键合
	- 添加全局参数 bm_b0(二面角/非键合 × bm_b0)
```

## 应用场景

本项目服务于基于概率流的相对结合自由能计算框架：

```
                        热力学循环

    蛋白-配体A复合物  ─────────────────→  蛋白-配体B复合物
          │                                      │
          │ ΔG_unbind_A                          │ ΔG_bind_B
          ↓                                      ↓
      溶剂中配体A  ──────────────────→  溶剂中配体B
                      ΔG_solvent
```

**REST2 采样任务**：
- 溶剂分支：配体在溶剂中的构象采样
- 复合物分支：配体在蛋白口袋中的构象采样

## 项目结构

```
OMEGA/
├── CLAUDE.md                     # AI 协作指南
├── LICENSE                       # MIT 许可证
├── README.md                     # 项目说明
├── requirements.txt              # 依赖列表
│
├── data/                         # 原始输入数据
│   ├── alanine_dipeptide/        # 阶段一：测试体系
│   ├── ligands/                  # 阶段二：溶剂分支配体
│   └── complexes/                # 阶段三：复合物结构
│
├── docs/                         # 文档
│   ├── MBAR_REWEIGHTING_PLAN.md
│   └── alanine_dipeptide_workflow.md
│
├── notebooks/                    # Jupyter notebooks
│   └── Running_a_REST_simulation.ipynb
│
├── outputs/                      # 运行输出
│   ├── alanine_dipeptide/        # 阶段一输出
│   │   ├── system.xml
│   │   ├── system.pdb
│   │   ├── topology.pkl
│   │   └── hremd/                # HREMD 采样输出
│   ├── solvent/                  # 阶段二：溶剂分支输出
│   └── complex/                  # 阶段三：复合物分支输出
│
├── results/                      # 分析结果
│   ├── figures/                  # 图片
│   └── reports/                  # 报告
│
├── scripts/                      # Python 脚本
│   ├── 00_optimize_ladder.py     # 温度阶梯优化
│   ├── 01_prepare_system.py      # 系统准备
│   ├── 02_run_rest2_hremd.py     # 运行 HREMD (CPU)
│   ├── 02_run_rest2_hremd_gpu.py # 运行 HREMD (GPU)
│   ├── 03_analyze_results.py     # 分析结果
│   ├── 04_mbar_reweighting.py    # MBAR 重加权
│   └── utils/                    # 工具函数
│
└── tests/                        # 测试代码
```

## 快速开始

### 1. 安装依赖

```bash
conda create -n rest2 python=3.10 -y
conda activate rest2
conda install -c conda-forge openmm openmmtools pymbar mdtraj matplotlib pyarrow -y
```

### 2. 运行测试案例

```bash
cd scripts

# 步骤 1: 准备系统（溶剂化 + REST2 缩放）
python 01_prepare_system.py

# 步骤 2: 运行 HREMD 采样
python 02_run_rest2_hremd.py      # CPU 版本
# 或
python 02_run_rest2_hremd_gpu.py  # GPU 版本

# 步骤 3: 分析结果
python 03_analyze_results.py

# 步骤 4: MBAR 重加权
python 04_mbar_reweighting.py
```

## 数据格式

### samples.arrow
```python
# 采样统计（PyArrow 格式）
u_kn: 能量矩阵
replica_states: 副本状态映射
acceptance_matrix: 交换接受率
```

### 轨迹文件 (r*.dcd)
```python
# 各副本的 DCD 轨迹文件
r0.dcd, r1.dcd, ...  # 可用 MDTraj 读取
```

## 参考文献

- **REST2 原始论文**: Wang, L. et al. (2011). *Replica Exchange with Solute Scaling: A More Efficient Version of Replica Exchange with Solute Tempering (REST2)*. J. Phys. Chem. B, 115(30), 9431-9438. DOI: [10.1021/jp204407d](https://doi.org/10.1021/jp204407d)

- **OpenMM**: Eastman, P. et al. (2017). *OpenMM 7: Rapid development of high performance algorithms for molecular dynamics*. PLoS Comput. Biol., 13(7), e1005659.

- **MBAR**: Shirts, M.R. & Chodera, J.D. (2008). *Statistically optimal analysis of samples from multiple equilibrium states*. J. Chem. Phys., 129, 124105.

## 许可

MIT License - Use this however you want. Make it your own.

随意使用，让他成为你自己的。

