#!/usr/bin/env python
"""
隐式溶剂系统准备脚本（通用版）

支持气相（vacuum）和隐式溶剂（GBSA）两种模式。

使用方式:
    python 01_prepare_system.py --solvent vacuum
    python 01_prepare_system.py --solvent gbsa

输入: 由配置文件指定的 PDB 文件
输出: system.xml, system.pdb
"""

import argparse
import pathlib
import yaml
from openmm.app import PDBFile, ForceField, NoCutoff, HBonds
from openmm import XmlSerializer


def load_config(solvent: str) -> dict:
    """加载溶剂配置文件"""
    config_path = pathlib.Path(__file__).parent / f"{solvent}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='隐式溶剂系统准备')
    parser.add_argument(
        '--solvent',
        choices=['vacuum', 'gbsa'],
        required=True,
        help='溶剂模型: vacuum (气相) 或 gbsa (隐式溶剂)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.solvent)

    solvent_config = config['solvent']
    input_config = config['input']
    output_config = config['output']

    # 项目根目录
    project_root = pathlib.Path(__file__).resolve().parents[2]

    print("=" * 60)
    print(f"系统准备: {solvent_config['name'].upper()} 模式")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. 加载分子
    # -------------------------------------------------------------------------
    print("\n[1/4] 加载分子...")

    input_pdb = project_root / input_config['pdb']
    if not input_pdb.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_pdb}")

    pdb = PDBFile(str(input_pdb))

    n_atoms = pdb.topology.getNumAtoms()
    residue_names = [res.name for res in pdb.topology.residues()]

    print(f"  - 文件: {input_pdb}")
    print(f"  - 原子数: {n_atoms}")
    print(f"  - 残基: {residue_names}")

    # -------------------------------------------------------------------------
    # 2. 加载力场
    # -------------------------------------------------------------------------
    print("\n[2/4] 加载力场...")

    force_fields = solvent_config['force_fields']
    forcefield = ForceField(*force_fields)
    print(f"  - 力场文件: {force_fields}")

    # -------------------------------------------------------------------------
    # 3. 创建系统
    # -------------------------------------------------------------------------
    print("\n[3/4] 创建系统...")

    # 基础参数
    create_system_kwargs = {
        'nonbondedMethod': NoCutoff,
        'constraints': HBonds,
    }

    # GBSA 特有参数
    if 'solute_dielectric' in solvent_config:
        create_system_kwargs['soluteDielectric'] = solvent_config['solute_dielectric']
        print(f"  - 溶质介电常数: {solvent_config['solute_dielectric']}")

    if 'solvent_dielectric' in solvent_config:
        create_system_kwargs['solventDielectric'] = solvent_config['solvent_dielectric']
        print(f"  - 溶剂介电常数: {solvent_config['solvent_dielectric']}")

    system = forcefield.createSystem(pdb.topology, **create_system_kwargs)

    print(f"  - 非键合方法: NoCutoff")
    print(f"  - 约束: HBonds")
    print(f"  - 系统中的力:")
    for i, force in enumerate(system.getForces()):
        print(f"    [{i}] {type(force).__name__}")

    # -------------------------------------------------------------------------
    # 4. 保存系统
    # -------------------------------------------------------------------------
    print("\n[4/4] 保存系统...")

    output_dir = project_root / output_config['system_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 OpenMM System XML
    system_xml_path = output_dir / "system.xml"
    with open(system_xml_path, 'w') as f:
        f.write(XmlSerializer.serialize(system))
    print(f"  - 保存: {system_xml_path}")

    # 保存 PDB
    system_pdb_path = output_dir / "system.pdb"
    with open(system_pdb_path, 'w') as f:
        PDBFile.writeFile(pdb.topology, pdb.positions, f)
    print(f"  - 保存: {system_pdb_path}")

    # -------------------------------------------------------------------------
    # 完成
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"[OK] {solvent_config['name'].upper()} 系统准备完成")
    print("=" * 60)
    print(f"\n下一步: python 02_run_md.py --solvent {args.solvent}")


if __name__ == "__main__":
    main()
