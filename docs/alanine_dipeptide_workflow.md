# REST2 测试：Alanine Dipeptide（显式溶剂，使用 femto 标准 API）

## 测试目标

验证 femto 的 REST2 实现在显式溶剂中的性能，并展示 **femto 标准 API 的正确使用方式**：

### REST2 特性
- 只缩放扭转和非键合项（不缩放键和角）
- 数值稳定性好（水分子不受影响）
- 计算效率高（减少不必要的 CustomForce）
- 采样增强有效（构象转换加速）

### femto 标准 API（v0.3.0+）
- 使用 `mdtop.Topology` 而不是 OpenMM topology
- 使用 `femto.md.prepare.prepare_system()` 进行溶剂化
- 使用 `femto.md.config.Prepare` 配置类
- 使用 `topology.select()` PyMol 语法选择原子
- 完全符合 [femto 官方文档](https://psivant.github.io/femto/latest/guide-md/)

## 文件结构

```
test_alanine_dipeptide/
├── README.md                      # 本文件
├── alanine-dipeptide.pdb         # 输入结构（真空）
├── 01_prepare_system.py          #  使用 femto API 准备系统
├── 02_run_rest2_hremd.py         #  运行 REST2 HREMD
├── 03_analyze_results.py         # 分析结果
├── system.xml                    # OpenMM System（运行后生成）
├── system.pdb                    # 溶剂化后的 PDB（运行后生成）
├── topology.pkl                  #  mdtop.Topology 对象（运行后生成）
└── outputs/                      # 模拟输出（运行后生成）
    ├── samples.arrow             # 采样统计
    ├── trajectories/r*.dcd       # 各副本轨迹
    └── checkpoint.pkl            # 检查点
```

** = 使用 femto 标准 API 实现**

## 快速开始

### 1. 准备系统（~1 分钟）

```bash
cd /Users/yiming/projects/rest2/test_alanine_dipeptide
python 01_prepare_system.py
```

**使用的 femto API**:
```python
# [OK] 使用 mdtop.Topology
topology = mdtop.Topology.from_file('alanine-dipeptide.pdb')

# [OK] 使用 femto.md.config.Prepare 配置类
config = femto.md.config.Prepare(
    water_model='tip3p',
    box_padding=10.0 * openmm.unit.angstrom,
    box_shape='cube'
)

# [OK] 使用 femto.md.prepare.prepare_system() 标准 API
topology, system = femto.md.prepare.prepare_system(
    receptor=None,
    ligand_1=topology,  # alanine dipeptide 作为 ligand
    ligand_2=None,
    config=config
)

# [OK] 使用 PyMol 选择语法
solute_idxs = topology.select(f"resn {femto.md.constants.LIGAND_1_RESIDUE_NAME}")

# [OK] 应用 REST2
femto.md.rest.apply_rest(system, solute_idxs, rest_config)
```

**输出**:
- `system.xml` - 包含 REST2 缩放的 OpenMM 系统
- `system.pdb` - 溶剂化后的坐标（~3000 原子，~1000 水分子）
- `topology.pkl` - **mdtop.Topology 对象（含坐标）**

**检查点**:
- 应该看到 "系统总原子数: ~3000"
- 应该看到 "溶质原子: 22 (L01)"
- 应该看到 "REST 全局参数已添加: {'bm_b0', 'sqrt<bm_b0>'}"
- 应该看到 "使用的 API: mdtop.Topology, prepare_system(), ..."

### 2. 运行 HREMD（~15-20 分钟 CPU / ~3-5 分钟 GPU）

```bash
# CPU 版本
python 02_run_rest2_hremd.py

# 如果有 GPU，修改脚本中的 platform='CPU' 为 platform='CUDA'
```

**过程**:
1. 平衡化阶段（~2 分钟）
   - 最小化
   - NVT 升温（50K → 300K）
   - NPT 平衡（300K, 1 bar）

2. HREMD 采样（~15 分钟）
   - 6 个副本（300K - 500K）
   - 500 cycles × 0.5 ps = 250 ps 总采样时间
   - 进度条显示实时进度

**输出**:
- `outputs/samples.arrow` - u_kn 矩阵和交换统计
- `outputs/trajectories/r{0-5}.dcd` - 各副本轨迹
- `outputs/checkpoint.pkl` - 检查点（可用于续算）

**检查点**:
- 应该看到进度条从 0% 到 100%
- 应该看到 "HREMD 完成！"
- 不应该有 NaN 错误（REST2 的鲁棒性）

### 3. 分析结果（~1 分钟）

```bash
python 03_analyze_results.py
```

**输出**:
- `acceptance_rates.png` - 交换接受率矩阵和柱状图
- `energy_convergence.png` - 能量时间序列和移动平均
- `ramachandran.png` - φ/ψ 扭转角分布（如果安装了 mdtraj）

**关键指标**:
- 相邻态接受率：15-35%（理想范围）
- 能量收敛：移动平均趋于稳定
- 构象转换：观察到 C7eq ↔ C7ax 转换

## 预期结果

### 1. 交换接受率

**理想范围**: 15-35%（相邻态）

```
相邻态接受率:
  State 0 ↔ 1: 25.3% [OK]
  State 1 ↔ 2: 22.8% [OK]
  State 2 ↔ 3: 19.5% [OK]
  ...
```

### 2. 能量收敛

- 前 50-100 cycles：能量快速下降（平衡阶段）
- 之后：能量围绕平均值涨落（已收敛）

### 3. 扭转角分布

- **C7eq** (αR): φ ~ -80°, ψ ~ 80°
- **C7ax** (αL): φ ~ 60°, ψ ~ -60°
- REST2 应该能观察到这两种构象的转换

## 改进优势（相比旧版代码）

### 1. **符合 femto 0.3.0+ 标准**
```python
# [FAIL] 旧版（不推荐）
pdb = openmm.app.PDBFile('file.pdb')
modeller = openmm.app.Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, ...)
system = forcefield.createSystem(modeller.topology, ...)
solute_idxs = set(range(22))  # 硬编码

# [OK] 新版（推荐）
topology = mdtop.Topology.from_file('file.pdb')
topology, system = femto.md.prepare.prepare_system(..., config=config)
solute_idxs = topology.select("not water")  # PyMol 语法
```

### 2. **更强大的原子选择**
```python
# PyMol 语法示例
topology.select("not water")           # 非水分子
topology.select("resn L01")            # 按残基名
topology.select("protein")             # 蛋白质
topology.select("name CA")             # α碳原子
topology.select("within 5 of resn L01") # 5Å范围内
```

### 3. **配置化管理**
- 所有参数通过配置类管理，易于修改和复用
- 符合 femto 文档规范
- 支持序列化和持久化

### 4. **与 femto 生态无缝集成**
- `mdtop.Topology` 可用于 `femto.fe.*` 模块（FEP、ATM、SepTop）
- 支持 `.to_openmm()` / `.from_openmm()` 双向转换
- 统一的文件 I/O 接口

## 故障排除

### 问题 1: ImportError: No module named 'femto' 或 'mdtop'

**解决**:
```bash
# 确保使用 femto_test 环境
conda activate femto_test

# 从 conda-forge 安装（推荐）
conda install -c conda-forge femto

# 或从源码安装
cd /Users/yiming/projects/rest2/femto
pip install -e .
```

### 问题 2: Python 版本太低

**错误**: `ERROR: Package 'femto' requires a different Python: 3.8.x not in '>=3.10'`

**解决**: 创建 Python 3.10+ 环境
```bash
conda create -n femto_test python=3.10 -y
conda activate femto_test
conda install -c conda-forge femto
```

### 问题 3: CUDA 相关错误

**解决**: 改用 CPU
```python
# 在脚本中修改
platform='CPU'  # 而不是 'CUDA'
```

### 问题 4: 内存不足

**解决**: 减少副本数或缩短采样时间
```python
# 02_run_rest2_hremd.py 中修改
n_replicas = 4      # 原来是 6
n_cycles = 250      # 原来是 500
```

## 参考文献

**REST2 原始论文**:
- Wang, L. et al. (2011). *Replica Exchange with Solute Scaling: A More Efficient Version of Replica Exchange with Solute Tempering (REST2)*. J. Phys. Chem. B, 115(30), 9431-9438.
- DOI: 10.1021/jp204407d
- 关键发现："只缩放扭转角，不缩放键和角"

**femto 文档**:
- 官方文档：https://psivant.github.io/femto/latest/
- MD 指南：https://psivant.github.io/femto/latest/guide-md/
- 迁移指南：https://psivant.github.io/femto/latest/migration/

## 时间估算

| 步骤 | CPU | GPU |
|-----|-----|-----|
| Step 1 (准备) | ~1 分钟 | ~1 分钟 |
| Step 2 (HREMD) | ~15-20 分钟 | ~3-5 分钟 |
| Step 3 (分析) | ~1 分钟 | ~1 分钟 |
| **总计** | **~17-22 分钟** | **~5-7 分钟** |

## 成功标准

- [x] 系统包含 ~3000 原子（~1000 水分子）
- [x] REST2 只缩放溶质 22 原子
- [x] 使用 femto 标准 API（mdtop.Topology, prepare_system）
- [x] HREMD 运行完成无 NaN
- [x] 相邻态接受率 15-35%
- [x] 观察到构象转换
- [x] 生成完整的分析图表

## 🎓 学习要点

1. **femto 0.3.0 API 变化**: 从 `parmed.Structure` 迁移到 `mdtop.Topology`，使用 `prepare_system()` 而不是手动溶剂化。

2. **REST2 设计哲学**: 只缩放影响构象转换的自由度（扭转），不浪费计算在快速振动模式（键/角）上。

3. **solute tempering**: 只对溶质"加温"，溶剂保持常温 → 减少计算量。

4. **显式溶剂的重要性**: 比隐式溶剂更真实，但计算量更大 → REST2 的效率优势更明显。

---

**祝测试顺利！** 

如有问题，请检查：
1. Python 版本是否 >= 3.10？
2. conda 环境是否激活？
3. 依赖是否齐全？（femto, mdtop, pyarrow, matplotlib, mdtraj）
4. 脚本是否报错？（查看终端输出）
