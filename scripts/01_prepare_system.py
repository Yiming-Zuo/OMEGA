#!/usr/bin/env python
"""
步骤 1: 准备 Alanine Dipeptide 的显式溶剂系统（使用 femto 标准 API）

改进要点：
- 使用 mdtop.Topology 而不是 OpenMM topology
- 使用 femto.md.prepare.prepare_system() 标准 API
- 使用 femto.md.config.Prepare 配置类
- 灵活指定溶质残基（支持多残基小分子/肽段）
- 使用 PyMol 选择语法（topology.select()）
- 完全符合 femto 文档规范
"""

import pickle
import pathlib
import openmm
import openmm.unit
import mdtop
import femto.md.prepare
import femto.md.config
import femto.md.rest
import femto.md.constants

# =====================================================================
# 配置：指定哪些残基是溶质（可根据需要修改）
# =====================================================================
# 对于 alanine dipeptide，所有三个残基都是溶质
SOLUTE_RESIDUES = ["ACE", "ALA", "NME"]

# 其他示例：
# SOLUTE_RESIDUES = ["LIG"]              # 单残基配体
# SOLUTE_RESIDUES = ["GLY", "ALA"]       # 双残基肽
# SOLUTE_RESIDUES = None                 # 自动：所有非水分子
# =====================================================================

print("="*60)
print("Step 1: 准备显式溶剂系统 (REST2)")
print("使用 femto 标准 API + 灵活指定溶质残基")
print("="*60)

# =====================================================================
# 1. 加载 alanine dipeptide 为 mdtop.Topology
# =====================================================================
print("\n[1/6] 加载 alanine dipeptide...")
topology = mdtop.Topology.from_file('alanine-dipeptide.pdb')

# 获取所有残基名
original_residues = [r.name for r in topology.residues]
print(f"✓ 加载完成: {len(topology.atoms)} 原子（真空）")
print(f"  - 残基: {original_residues}")
print(f"  - 残基数: {len(topology.residues)}")

# =====================================================================
# 2. 配置溶剂化和参数化（使用 femto.md.config.Prepare）
# =====================================================================
print("\n[2/6] 配置系统准备参数...")
prep_config = femto.md.config.Prepare(
    # 溶剂化设置
    ionic_strength=0.0 * openmm.unit.molar,    # 不添加额外的盐
    neutralize=False,                           # alanine dipeptide 是中性的
    water_model='tip3p',                        # TIP3P 水模型
    box_padding=10.0 * openmm.unit.angstrom,   # 1.0 nm 边界
    box_shape='cube',                           # 立方盒子

    # 力场设置
    default_protein_ff=[
        'amber14-all.xml',       # Amber14 力场
        'amber14/tip3pfb.xml'    # TIP3P 水模型参数
    ],
    default_ligand_ff=None  # 不用 OpenFF，直接用 Amber14
)

print("✓ 配置完成:")
print(f"  - 水模型: {prep_config.water_model}")
print(f"  - 盒子边界: {prep_config.box_padding.value_in_unit(openmm.unit.nanometers)} nm")
print(f"  - 盒子形状: {prep_config.box_shape}")
print(f"  - 离子浓度: {prep_config.ionic_strength}")
print(f"  - 力场: Amber14")

# =====================================================================
# 3. 溶剂化和参数化（使用 femto 标准 API）
# =====================================================================
print("\n[3/6] 使用 femto.md.prepare.prepare_system()...")
print("（这会自动完成溶剂化、参数化、添加压强控制等）")

# 将 alanine dipeptide 作为 ligand_1 传入
# 注意：prepare_system() 不会修改原始残基名
topology, system = femto.md.prepare.prepare_system(
    receptor=None,       # 没有蛋白质受体
    ligand_1=topology,   # alanine dipeptide 作为 ligand
    ligand_2=None,       # 没有第二个配体
    cofactors=None,      # 没有辅因子
    config=prep_config
)

# 统计系统组成（按残基类型）
residue_counts = {}
for residue in topology.residues:
    residue_counts[residue.name] = residue_counts.get(residue.name, 0) + 1

n_water_molecules = residue_counts.get('HOH', 0)
solute_residue_info = ', '.join(f'{name}({count})' for name, count in residue_counts.items() if name != 'HOH')

print(f"✓ 系统准备完成:")
print(f"  - 总原子数: {len(topology.atoms)}")
print(f"  - 溶质残基: {solute_residue_info}")
print(f"  - 水分子数: {n_water_molecules}")
print(f"  - 盒子中总残基: {len(topology.residues)}")

# 检查系统中的力
print(f"\n  系统中的力:")
for i, force in enumerate(system.getForces()):
    print(f"    [{i}] {type(force).__name__}")

# =====================================================================
# 4. 选择溶质原子（使用 PyMol 语法 - 灵活指定残基）
# =====================================================================
print("\n[4/6] 选择溶质原子（PyMol 语法）...")

if SOLUTE_RESIDUES is not None:
    # 方法 1: 按指定的残基名选择（推荐，灵活可配置）
    if len(SOLUTE_RESIDUES) == 1:
        solute_query = f"resn {SOLUTE_RESIDUES[0]}"
    else:
        solute_query = " or ".join(f"resn {name}" for name in SOLUTE_RESIDUES)

    solute_idxs = topology.select(solute_query)
    print(f"✓ 方法 1: 按残基名选择（配置方式）")
    print(f"  - 溶质残基: {SOLUTE_RESIDUES}")
    print(f"  - 选择查询: {solute_query}")
    print(f"  - 结果: {len(solute_idxs)} 原子")
else:
    # 自动模式：选择所有非水分子
    solute_query = "not resn HOH"
    solute_idxs = topology.select(solute_query)
    print(f"✓ 方法 1: 自动选择（所有非水分子）")
    print(f"  - 选择查询: {solute_query}")
    print(f"  - 结果: {len(solute_idxs)} 原子")

# 方法 2: 选择非水分子（通用方法，用于验证）
non_water_idxs = topology.select("not resn HOH")
print(f"✓ 方法 2: 通用方法 (not resn HOH) → {len(non_water_idxs)} 原子")

# 验证两种方法一致（应该一致，因为系统只有溶质和水）
if set(solute_idxs) == set(non_water_idxs):
    print(f"✓ 验证通过: 两种方法选择的原子相同")
else:
    print(f"[WARN] 警告: 两种方法不一致")
    print(f"  - 方法1 选中: {len(solute_idxs)} 原子")
    print(f"  - 方法2 选中: {len(non_water_idxs)} 原子")
    print(f"  - 差异原因: 可能有其他分子（如离子、辅因子）")
    print(f"  - 使用方法1的结果（更精确）")

# =====================================================================
# 5. 应用 REST2 缩放（只缩放扭转和非键合）
# =====================================================================
print("\n[5/6] 应用 REST2 缩放...")

rest_config = femto.md.config.REST(
    scale_bonds=False,      # REST2 不缩放键
    scale_angles=False,     # REST2 不缩放角
    scale_torsions=True,    # 缩放扭转（降低能垒）
    scale_nonbonded=True    # 缩放非键合（静电和LJ）
)

print(f"✓ REST2 配置:")
print(f"  - scale_bonds:      False  ← REST2 特性")
print(f"  - scale_angles:     False  ← REST2 特性")
print(f"  - scale_torsions:   True   ← 降低扭转能垒")
print(f"  - scale_nonbonded:  True   ← 缩放静电和LJ")

femto.md.rest.apply_rest(system, solute_idxs, rest_config)
print(f"✓ REST2 已应用到 {len(solute_idxs)} 个溶质原子")

# 验证 REST 参数已添加
rest_params_found = []
for force in system.getForces():
    if hasattr(force, 'getNumGlobalParameters'):
        for i in range(force.getNumGlobalParameters()):
            param_name = force.getGlobalParameterName(i)
            if 'bm_b0' in param_name:
                rest_params_found.append(param_name)

if rest_params_found:
    print(f"✓ REST 全局参数已添加: {set(rest_params_found)}")
else:
    print(f"[WARN] 警告: 未检测到 REST 全局参数")

# =====================================================================
# 6. 保存系统（mdtop.Topology + OpenMM System）
# =====================================================================
print("\n[6/6] 保存系统...")

# 保存 OpenMM System（XML 格式）
pathlib.Path('system.xml').write_text(openmm.XmlSerializer.serialize(system))
print("✓ 保存: system.xml (OpenMM System)")

# 保存 mdtop.Topology（pickle 格式，含坐标）
pathlib.Path('topology.pkl').write_bytes(pickle.dumps(topology))
print("✓ 保存: topology.pkl (mdtop.Topology)")

# 保存 PDB 文件（可视化）
topology.to_file('system.pdb')

print("✓ 保存: system.pdb")

# 获取盒子尺寸
box_vectors = system.getDefaultPeriodicBoxVectors()
box_size = box_vectors[0][0].value_in_unit(openmm.unit.nanometers)
print(f"✓ 盒子尺寸: {box_size:.2f} × {box_size:.2f} × {box_size:.2f} nm³")

# =====================================================================
# 总结
# =====================================================================
print("\n" + "="*60)
print("[OK] 系统准备完成！")
print("="*60)
print(f"系统信息:")
print(f"  - 总原子数: {len(topology.atoms)}")
if SOLUTE_RESIDUES:
    print(f"  - 溶质原子: {len(solute_idxs)} (残基: {', '.join(SOLUTE_RESIDUES)})")
else:
    print(f"  - 溶质原子: {len(solute_idxs)} (自动选择)")
print(f"  - 水分子数: {n_water_molecules}")
print(f"  - REST2 已应用: 只缩放溶质的扭转+非键合")
print(f"  - 盒子尺寸: {box_size:.2f} nm (cube)")
print()
print(f"使用的 API:")
print(f"  - mdtop.Topology (标准 topology 格式)")
print(f"  - femto.md.prepare.prepare_system() (标准溶剂化 API)")
print(f"  - femto.md.config.Prepare (配置类)")
print(f"  - topology.select() (PyMol 选择语法 - 灵活指定残基)")
print(f"  - femto.md.rest.apply_rest() (REST2 API)")
print()
print(f"提示: 配置说明")
print(f"  - 溶质残基可通过修改文件开头的 SOLUTE_RESIDUES 变量配置")
if SOLUTE_RESIDUES:
    print(f"  - 当前配置: {SOLUTE_RESIDUES}")
else:
    print(f"  - 当前配置: None (自动选择所有非水分子)")
print(f"  - 支持单残基配体、多残基肽段、任意组合")
print("="*60)
print("\n下一步: 运行 python 02_run_rest2_hremd.py")
