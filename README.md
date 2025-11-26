# REST2

REST2 是一个基于 OpenMM 的多副本增强采样（REST2-HREMD）工具，用于热力学循环中溶剂分支和蛋白-配体复合物分支两端状态的分子动力学采样。搭配 FReD 数据构建工具，可将多副本采样数据通过 MBAR 重加权为无偏平衡系综，为基于概率流的相对结合自由能计算提供标准化的数据基础。

> **当前状态**：初步开发阶段，正在进行丙氨酸二肽（Alanine Dipeptide）的测试采样。

## 数据流动

```
    输入结构               系统准备                  HREMD 采样              结果分析              MBAR 重加权
   (溶质 PDB)        (溶剂化+REST2缩放)           (多副本交换)          (接受率/能量/构象)        (无偏系综)
        ↓                   ↓                        ↓                      ↓                     ↓
┌──────────────┐    ┌──────────────┐        ┌──────────────┐       ┌──────────────┐      ┌──────────────┐
│  ligand.pdb  │    │ system.xml   │        │ samples.arrow│       │ 接受率矩阵    │      │ MBAR 权重    │
│              │ →  │ system.pdb   │   →    │ r*.dcd 轨迹  │  →    │ 能量收敛     │  →   │ 重加权构象   │
│              │    │ topology.pkl │        │ checkpoint   │       │ Ramachandran │      │ 自由能估计   │
└──────────────┘    └──────────────┘        └──────────────┘       └──────────────┘      └──────────────┘
                    01_prepare_system       02_run_rest2_hremd     03_analyze_results   04_mbar_reweighting
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
REST2/
├── README.md
├── Running_a_REST_simulation.ipynb   # OpenMM REST2 教程
└── test_alanine_dipeptide/           # 丙氨酸二肽测试案例
    ├── alanine-dipeptide.pdb         # 输入结构
    ├── 00_optimize_ladder.py         # 温度阶梯优化
    ├── 01_prepare_system.py          # 系统准备（溶剂化 + REST2 缩放）
    ├── 02_run_rest2_hremd.py         # 运行 HREMD 采样
    ├── 03_analyze_results.py         # 结果分析
    ├── 04_mbar_reweighting.py        # MBAR 重加权
    ├── system.xml                    # OpenMM System（运行后生成）
    ├── system.pdb                    # 溶剂化后的结构（运行后生成）
    ├── topology.pkl                  # 拓扑对象（运行后生成）
    └── outputs/                      # 采样输出（运行后生成）
        ├── samples.arrow             # 采样统计
        └── r*.dcd                    # 各副本轨迹
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
cd test_alanine_dipeptide

# 步骤 1: 准备系统（溶剂化 + REST2 缩放）
python 01_prepare_system.py

# 步骤 2: 运行 HREMD 采样
python 02_run_rest2_hremd.py

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

