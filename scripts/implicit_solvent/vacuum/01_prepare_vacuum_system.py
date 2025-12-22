#!/usr/bin/env python
"""
气相（真空）系统准备脚本

为丙氨酸二肽（ACE-ALA-NME）准备气相模拟系统。

关键差异（vs 显式溶剂）：
1. 不溶剂化，直接使用真空结构
2. 使用 NoCutoff 非键合方法（无截断、无周期性）
3. 不添加 Barostat（NVT 系综）

输入：data/alanine_dipeptide/alanine-dipeptide.pdb
输出：outputs/implicit_solvent/vacuum/alanine_dipeptide/
    - system.xml    OpenMM System 序列化文件
    - system.pdb    可视化用 PDB
"""

import pathlib
from openmm.app import PDBFile, ForceField, NoCutoff, HBonds
from openmm import XmlSerializer

# =============================================================================
# 配置
# =============================================================================
# 路径配置（相对于项目根目录）
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
INPUT_PDB = PROJECT_ROOT / "data" / "alanine_dipeptide" / "alanine-dipeptide.pdb"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "implicit_solvent" / "vacuum" / "alanine_dipeptide"

# 力场配置
FORCE_FIELD_FILES = [
    'amber14-all.xml',  # Amber14 蛋白质力场（无需水模型）
]

# =============================================================================
# 主程序
# =============================================================================
def main():
    print("=" * 60)
    print("气相系统准备: 丙氨酸二肽 (ACE-ALA-NME)")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. 加载分子
    # -------------------------------------------------------------------------
    print("\n[1/4] 加载丙氨酸二肽...")

    if not INPUT_PDB.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PDB}")

    pdb = PDBFile(str(INPUT_PDB))

    # 统计信息
    n_atoms = pdb.topology.getNumAtoms()
    n_residues = pdb.topology.getNumResidues()
    residue_names = [res.name for res in pdb.topology.residues()]

    print(f"  - 文件: {INPUT_PDB}")
    print(f"  - 原子数: {n_atoms}")
    print(f"  - 残基数: {n_residues}")
    print(f"  - 残基: {residue_names}")

    # 验证是否是封端的丙氨酸二肽
    expected_residues = ['ACE', 'ALA', 'NME']
    if residue_names != expected_residues:
        print(f"  [WARN] 残基顺序与预期不符: 预期 {expected_residues}")

    # -------------------------------------------------------------------------
    # 2. 加载力场
    # -------------------------------------------------------------------------
    print("\n[2/4] 加载力场...")

    forcefield = ForceField(*FORCE_FIELD_FILES)
    print(f"  - 力场文件: {FORCE_FIELD_FILES}")

    # -------------------------------------------------------------------------
    # 3. 创建气相系统
    # -------------------------------------------------------------------------
    print("\n[3/4] 创建气相系统...")

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=NoCutoff,  # 关键: 气相无截断，无周期性边界
        constraints=HBonds,        # 约束氢键，允许 2 fs 步长
    )

    # 打印系统中的力
    print(f"  - 非键合方法: NoCutoff（无周期性边界）")
    print(f"  - 约束: HBonds（氢键约束）")
    print(f"  - 系统中的力:")
    for i, force in enumerate(system.getForces()):
        print(f"    [{i}] {type(force).__name__}")

    # -------------------------------------------------------------------------
    # 4. 保存系统
    # -------------------------------------------------------------------------
    print("\n[4/4] 保存系统...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存 OpenMM System XML
    system_xml_path = OUTPUT_DIR / "system.xml"
    with open(system_xml_path, 'w') as f:
        f.write(XmlSerializer.serialize(system))
    print(f"  - 保存: {system_xml_path}")

    # 保存 PDB（用于可视化和拓扑加载）
    system_pdb_path = OUTPUT_DIR / "system.pdb"
    with open(system_pdb_path, 'w') as f:
        PDBFile.writeFile(pdb.topology, pdb.positions, f)
    print(f"  - 保存: {system_pdb_path}")

    # -------------------------------------------------------------------------
    # 完成
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[OK] 气相系统准备完成")
    print("=" * 60)
    print(f"\n下一步: 运行 02_run_vacuum_md.py 进行 MD 采样")


if __name__ == "__main__":
    main()
