# xyz 转 PDB/MOL2 工具

## 简介

`xyz_to_pdb.py` 是一个将量子化学计算输出的 xyz 文件转换为分子动力学模拟输入格式（PDB/MOL2）的工具。

### 主要功能

- 读取多种格式的 xyz 文件
- 智能推断化学键连接
- 生成 PDB 文件（REST2 输入）
- 生成 MOL2 文件（ACPYPE 参数化）
- 验证分子结构合理性
- 输出详细的分子信息

---

## 基本用法

```bash
# 最简单的转换
python examples/xyz_to_pdb.py ethanol.xyz

# 自定义残基名
python examples/xyz_to_pdb.py ethanol.xyz -r ETH

# 包含几何优化
python examples/xyz_to_pdb.py ethanol.xyz --optimize

# 生成所有格式 + 详细信息
python examples/xyz_to_pdb.py ethanol.xyz --all-formats -v
```

---

## 支持的 xyz 格式

### 格式 1：标准 xyz

```
9
ethanol molecule
C   0.000   0.555   0.000
C   1.169  -0.404   0.000
O  -1.190  -0.216   0.000
H  -1.949   0.368   0.000
H   0.039   1.198   0.885
H   0.039   1.198  -0.885
H   2.112   0.142   0.000
H   1.133  -1.040  -0.883
H   1.133  -1.040   0.883
```

- 第一行：原子总数
- 第二行：注释（可选）
- 后续行：元素符号 + xyz 坐标（埃）

### 格式 2：带电荷信息（量化软件输出）

```
0 1
6   0.000   0.555   0.000
6   1.169  -0.404   0.000
8  -1.190  -0.216   0.000
1  -1.949   0.368   0.000
1   0.039   1.198   0.885
1   0.039   1.198  -0.885
1   2.112   0.142   0.000
1   1.133  -1.040  -0.883
1   1.133  -1.040   0.883
```

- 第一行：净电荷 + 自旋多重度
- 后续行：原子序号 + xyz 坐标（埃）

---

## 参数说明

### 必需参数

- `xyz_file`：输入的 xyz 文件路径

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-o, --output` | 输出文件名前缀 | 使用输入文件名 |
| `-r, --residue-name` | PDB 残基名称 | MOL |
| `--chain` | PDB 链 ID | A |
| `--optimize` | 使用 UFF 力场优化几何 | False |
| `--all-formats` | 生成所有格式（PDB/MOL2/SDF） | False |
| `-v, --verbose` | 显示详细分子信息 | False |

---

## 输出文件

### 默认输出

- `molecule.pdb`：用于 REST2 模拟输入
- `molecule.mol2`：用于 ACPYPE 力场参数化

### 使用 `--all-formats` 时额外输出

- `molecule.sdf`：用于其他化学信息工具

---

## 完整工作流示例

### 示例 1：乙醇溶剂化 REST2 模拟

```bash
# 步骤 1：转换 xyz 格式
python examples/xyz_to_pdb.py ethanol.xyz -r ETH

# 输出：
#   ethanol.pdb（REST2 输入）
#   ethanol.mol2（ACPYPE 输入）

# 步骤 2：使用 ACPYPE 生成力场参数
acpype -i ethanol.mol2 -c bcc -n 0

# 输出：
#   ethanol.acpype/ethanol_GMX.xml（OpenMM 力场）

# 步骤 3：运行 REST2 溶剂化采样
python bin/launch_REST2_small_molecule.py \
    -pdb ethanol.pdb \
    -n ethanol_rest2 \
    -o output \
    --extra-ff ethanol.acpype/ethanol_GMX.xml \
    --solute-residues ETH \
    --time 50 \
    --minimize
```

### 示例 2：带电分子（如离子）

假设你的 xyz 文件是 +1 电荷的分子：

```bash
# xyz 文件第一行应该是：
# 1 1  （+1 电荷，单重态）

# 步骤 1：转换
python examples/xyz_to_pdb.py cation.xyz -r CAT

# 步骤 2：ACPYPE 参数化（指定电荷）
acpype -i cation.mol2 -c bcc -n 1

# 步骤 3：REST2（使用 Reaction Field）
python bin/launch_REST2_small_molecule.py \
    -pdb cation.pdb \
    -n cation_rest2 \
    -o output \
    --extra-ff cation.acpype/cation_GMX.xml \
    --solute-residues CAT \
    --nonbonded-method cutoff-periodic \
    --time 50
```

### 示例 3：优化气相 QM 结构到溶剂相

如果你的 xyz 来自气相 DFT 优化（如 M06-2X/MG3S），建议：

```bash
# 使用 --optimize 选项轻微调整几何
python examples/xyz_to_pdb.py molecule.xyz -r MOL --optimize

# 然后在 REST2 中启用能量最小化
python bin/launch_REST2_small_molecule.py \
    -pdb molecule.pdb \
    -n mol_rest2 \
    -o output \
    --extra-ff molecule.acpype/molecule_GMX.xml \
    --minimize \
    --time 100
```

---

## 结构验证

脚本会自动检查以下内容：

### 1. 键长合理性
- 正常键长范围：0.5-3.0 Å
- 异常键长会发出警告

### 2. 分子片段
- 检测不连接的原子团
- 提示可能的问题

### 3. 电荷状态
- 显示分子形式电荷
- 与输入的净电荷对比

---

## 常见问题

### Q1: 脚本报错 "键连接推断失败"

**可能原因：**
- 原子间距离过近或过远
- xyz 文件格式不正确

**解决方法：**
```bash
# 尝试几何优化
python examples/xyz_to_pdb.py molecule.xyz --optimize

# 检查 xyz 文件坐标单位（应为埃 Å）
# 检查第一行格式是否正确
```

### Q2: 生成的 MOL2 文件电荷不对

**说明：**
RDKit 默认不计算部分电荷，MOL2 文件中的电荷是形式电荷。

**正确流程：**
1. 使用本脚本生成 MOL2
2. 用 ACPYPE 计算 AM1-BCC 电荷：
   ```bash
   acpype -i molecule.mol2 -c bcc -n <净电荷>
   ```

### Q3: 如何处理多构象分子？

**方法 1：使用量化软件生成多个构象**
```bash
# 对每个构象分别转换
python examples/xyz_to_pdb.py conf1.xyz -o mol_conf1
python examples/xyz_to_pdb.py conf2.xyz -o mol_conf2
```

**方法 2：使用 REST2 采样构象**
```bash
# REST2 会自动探索不同构象
python bin/launch_REST2_small_molecule.py \
    -pdb molecule.pdb \
    --time 100  # 更长的采样时间
```

### Q4: 残基名有什么要求？

- **长度**：必须是 3 个字符（PDB 格式限制）
- **字符**：建议使用大写字母和数字
- **唯一性**：不要与水（HOH）、离子（NA、CL）重名

**推荐命名：**
- 乙醇：ETH
- 乙酸：ACE
- 苯：BEN
- 甲醇：MET
- 自定义：MOL、LIG、DRG

### Q5: 为什么需要 MOL2 格式？

- **PDB 格式**：只有坐标和原子名，缺少键信息和电荷
- **MOL2 格式**：包含键连接、键类型、部分电荷
- **ACPYPE 需求**：需要 MOL2 的键信息来正确分配力场参数

### Q6: 脚本支持哪些元素？

支持所有 RDKit 识别的元素（元素周期表前 118 个）。

**常见元素：**
- 有机元素：C, H, O, N, S, P
- 卤素：F, Cl, Br, I
- 金属（需要特殊处理）：Na, K, Ca, Mg, Zn

---

## 高级用法

### 批量转换脚本

创建 `batch_convert.sh`：

```bash
#!/bin/bash
# 批量转换 xyz 文件

for xyz_file in *.xyz; do
    base=$(basename "$xyz_file" .xyz)
    python examples/xyz_to_pdb.py "$xyz_file" -o "$base" -r MOL
    echo "已转换：$xyz_file"
done
```

### 与 Avogadro 配合使用

```bash
# 转换后在 Avogadro 中可视化检查
python examples/xyz_to_pdb.py molecule.xyz
avogadro molecule.pdb
```

### 与 PyMOL 配合使用

```bash
python examples/xyz_to_pdb.py molecule.xyz
pymol molecule.pdb

# 在 PyMOL 中：
# show sticks
# show spheres
# label all, "%s%s" % (resn, resi)
```

---

## 参考资源

### RDKit 文档
- 官网：https://www.rdkit.org/
- 教程：https://www.rdkit.org/docs/GettingStartedInPython.html

### ACPYPE 文档
- GitHub：https://github.com/alanwilter/acpype
- 论文：https://bmcresnotes.biomedcentral.com/articles/10.1186/1756-0500-5-367

---

## 故障排除

### 问题 1：ImportError: No module named 'rdkit'

```bash
# 安装 RDKit
conda install -c conda-forge rdkit

# 或在特定环境中安装
conda activate my_env
conda install -c conda-forge rdkit
```

### 问题 2：键连接推断结果不合理

**可能原因：**
- QM 计算的几何结构不合理
- xyz 文件坐标单位错误

**解决方法：**
1. 检查 xyz 文件（坐标应为埃 Å）
2. 使用 `--optimize` 选项
3. 在量化软件中重新优化几何

### 问题 3：生成的 PDB 无法在其他软件中打开

**可能原因：**
- 残基名不符合规范

**解决方法：**
```bash
# 使用标准的 3 字符残基名
python examples/xyz_to_pdb.py molecule.xyz -r LIG
```
