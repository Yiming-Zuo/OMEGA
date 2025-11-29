#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xyz 文件转换为 PDB/MOL2 格式工具

功能：
- 读取 xyz 文件（支持多种格式）
- 推断化学键连接
- 生成 PDB/MOL2/SDF 文件
- 验证分子结构合理性
- 适用于从量化计算结果准备 MD 模拟输入

"""

import sys
import os
import argparse
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import Lipinski
except ImportError:
    print("错误：需要安装 RDKit")
    print("安装命令：conda install -c conda-forge rdkit")
    sys.exit(1)


def parse_xyz_file(xyz_file):
    """
    解析 xyz 文件，支持多种格式

    支持的格式：
    1. 标准 xyz：
       <原子数>
       <注释行>
       <原子> <x> <y> <z>

    2. 带电荷信息的 xyz：
       <电荷> <自旋多重度>
       <原子序号> <x> <y> <z>

    3. 带注释头的 xyz（量化软件输出）：
       <注释行>
       <空行>
       <电荷> <自旋多重度>
       <原子序号> <x> <y> <z>

    Returns
    -------
    atoms : list of tuple
        [(element, x, y, z), ...]
    charge : int
        净电荷
    multiplicity : int
        自旋多重度
    """
    with open(xyz_file, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    atoms = []
    charge = 0
    multiplicity = 1
    start_line = 0

    # 寻找数据起始行
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 1:
            continue

        # 情况 1：标准 xyz（第一行是原子数）
        if i == 0 and len(parts) == 1 and parts[0].isdigit():
            # 跳过原子数和注释行
            start_line = 2
            break

        # 情况 2：电荷 + 自旋多重度行
        if len(parts) == 2:
            try:
                charge = int(parts[0])
                multiplicity = int(parts[1])
                start_line = i + 1
                break
            except ValueError:
                # 不是数字，继续寻找
                continue

        # 情况 3：直接是坐标数据（至少4列：元素 x y z）
        if len(parts) >= 4:
            try:
                # 尝试解析为坐标
                float(parts[1])
                float(parts[2])
                float(parts[3])
                # 成功，这是数据起始行
                start_line = i
                break
            except ValueError:
                # 不是坐标数据，是注释行，继续
                continue

    # 解析原子坐标
    for line in lines[start_line:]:
        parts = line.split()
        if len(parts) < 4:
            continue

        # 尝试解析坐标，如果失败则跳过（注释行）
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except (ValueError, IndexError):
            continue

        # 原子标识可能是元素符号或原子序号
        atom_id = parts[0]
        if atom_id.isdigit():
            # 原子序号，转换为元素符号
            atomic_num = int(atom_id)
            element = Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
        else:
            element = atom_id

        atoms.append((element, x, y, z))

    return atoms, charge, multiplicity


def create_mol_from_atoms(atoms, charge=0):
    """
    从原子列表创建 RDKit Mol 对象并推断键连接

    Parameters
    ----------
    atoms : list of tuple
        [(element, x, y, z), ...]
    charge : int
        净电荷

    Returns
    -------
    mol : rdkit.Chem.Mol
        RDKit 分子对象
    """
    # 创建可编辑的分子对象
    mol = Chem.RWMol()

    # 添加原子
    conf = Chem.Conformer(len(atoms))
    for i, (element, x, y, z) in enumerate(atoms):
        atom = Chem.Atom(element)
        mol.AddAtom(atom)
        conf.SetAtomPosition(i, (x, y, z))

    # 设置构象
    mol = mol.GetMol()
    mol.AddConformer(conf)

    # 推断键连接（关键步骤！）
    # 使用距离矩阵自动连接化学键
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    mol = Chem.Mol(mol)

    # 尝试确定键类型
    Chem.SanitizeMol(mol)

    # 设置总电荷
    if charge != 0:
        mol.SetProp("_TotalCharge", str(charge))

    return mol


def set_pdb_info(mol, residue_name="MOL", chain_id="A"):
    """
    为分子设置 PDB 残基信息

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        分子对象
    residue_name : str
        残基名称（默认 MOL）
    chain_id : str
        链 ID（默认 A）
    """
    for atom in mol.GetAtoms():
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(residue_name)
        info.SetResidueNumber(1)
        info.SetChainId(chain_id)

        # 原子名：元素符号 + 序号
        atom_name = f"{atom.GetSymbol()}{atom.GetIdx()+1:02d}"
        info.SetName(atom_name)
        info.SetIsHeteroAtom(True)  # 标记为 HETATM

        atom.SetMonomerInfo(info)


def validate_structure(mol):
    """
    验证分子结构合理性

    Returns
    -------
    issues : list of str
        发现的问题列表
    """
    issues = []

    # 检查是否有孤立原子
    fragments = Chem.GetMolFrags(mol, asMols=True)
    if len(fragments) > 1:
        issues.append(f"检测到 {len(fragments)} 个不连接的片段")

    # 检查键长合理性
    conf = mol.GetConformer()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        pos_i = conf.GetAtomPosition(i)
        pos_j = conf.GetAtomPosition(j)
        distance = pos_i.Distance(pos_j)

        # 合理键长范围：0.5-3.0 Å
        if distance < 0.5 or distance > 3.0:
            atom_i = mol.GetAtomWithIdx(i).GetSymbol()
            atom_j = mol.GetAtomWithIdx(j).GetSymbol()
            issues.append(f"异常键长：{atom_i}{i+1}-{atom_j}{j+1} = {distance:.2f} Å")

    # 检查形式电荷
    total_charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
    if total_charge != 0:
        issues.append(f"分子形式电荷：{total_charge:+d}")

    return issues


def print_molecule_info(mol):
    """
    打印分子详细信息
    """
    print("\n" + "="*60)
    print("分子信息摘要")
    print("="*60)

    # 基本信息
    formula = rdMolDescriptors.CalcMolFormula(mol)
    print(f"分子式：{formula}")
    print(f"原子总数：{mol.GetNumAtoms()}")
    print(f"重原子数：{mol.GetNumHeavyAtoms()}")
    print(f"氢原子数：{mol.GetNumAtoms() - mol.GetNumHeavyAtoms()}")
    print(f"化学键数：{mol.GetNumBonds()}")

    # SMILES 表示
    try:
        smiles = Chem.MolToSmiles(mol)
        print(f"SMILES：{smiles}")
    except:
        print("SMILES：无法生成")

    # 分子量
    mw = Descriptors.MolWt(mol)
    print(f"分子量：{mw:.2f} g/mol")

    # 氢键供体/受体
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    print(f"氢键供体：{hbd}")
    print(f"氢键受体：{hba}")

    # 可旋转键
    rotatable = Lipinski.NumRotatableBonds(mol)
    print(f"可旋转键：{rotatable}")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="将 xyz 文件转换为 PDB/MOL2 格式，用于分子动力学模拟",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本转换
  python xyz_to_pdb.py ethanol.xyz

  # 自定义输出名称和残基名
  python xyz_to_pdb.py ethanol.xyz -o my_ethanol -r ETH

  # 包含几何优化
  python xyz_to_pdb.py ethanol.xyz --optimize

  # 生成所有格式
  python xyz_to_pdb.py ethanol.xyz --all-formats
        """
    )

    parser.add_argument(
        "xyz_file",
        help="输入的 xyz 文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出文件名前缀（默认使用输入文件名）"
    )
    parser.add_argument(
        "-r", "--residue-name",
        default="MOL",
        help="PDB 残基名称（默认：MOL）"
    )
    parser.add_argument(
        "--chain",
        default="A",
        help="PDB 链 ID（默认：A）"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="使用 UFF 力场进行几何优化"
    )
    parser.add_argument(
        "--all-formats",
        action="store_true",
        help="生成所有支持的格式（PDB, MOL2, SDF）"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细信息"
    )

    args = parser.parse_args()

    # 检查输入文件
    xyz_file = Path(args.xyz_file)
    if not xyz_file.exists():
        print(f"错误：文件不存在：{xyz_file}")
        sys.exit(1)

    # 确定输出前缀
    if args.output:
        output_base = args.output
    else:
        output_base = xyz_file.stem

    print(f"\n🔄 正在转换：{xyz_file}")
    print(f"📁 输出前缀：{output_base}")
    print(f"🏷️  残基名称：{args.residue_name}\n")

    # 步骤 1：解析 xyz 文件
    print("步骤 1/5: 读取 xyz 文件...")
    try:
        atoms, charge, multiplicity = parse_xyz_file(xyz_file)
        print(f"读取 {len(atoms)} 个原子")
        print(f"净电荷 = {charge}, 自旋多重度 = {multiplicity}")
    except Exception as e:
        print(f"读取失败：{e}")
        sys.exit(1)

    # 步骤 2：创建分子对象并推断键连接
    print("\n步骤 2/5: 推断化学键连接...")
    try:
        mol = create_mol_from_atoms(atoms, charge)
        print(f"成功推断 {mol.GetNumBonds()} 个化学键")
    except Exception as e:
        print(f"键连接推断失败：{e}")
        sys.exit(1)

    # 步骤 3：设置 PDB 信息
    print("\n步骤 3/5: 设置 PDB 残基信息...")
    set_pdb_info(mol, args.residue_name, args.chain)
    print("PDB 信息设置完成")

    # 步骤 4：可选几何优化
    if args.optimize:
        print("\n步骤 4/5: 使用 UFF 力场优化几何结构...")
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            print("几何优化完成")
        except Exception as e:
            print(f"优化失败（将使用原始坐标）：{e}")
    else:
        print("\n步骤 4/5: 跳过几何优化")

    # 步骤 5：输出文件
    print("\n步骤 5/5: 生成输出文件...")

    # 5.1 PDB 文件
    pdb_file = f"{output_base}.pdb"
    try:
        Chem.MolToPDBFile(mol, pdb_file)
        print(f"PDB 文件：{pdb_file}")
    except Exception as e:
        print(f"PDB 输出失败：{e}")

    # 5.2 MOL2 文件（用于 ACPYPE）
    mol2_file = f"{output_base}.mol2"
    try:
        Chem.MolToMolFile(mol, mol2_file)
        print(f"MOL2 文件：{mol2_file}")
    except Exception as e:
        print(f"MOL2 输出失败：{e}")

    # 5.3 SDF 文件（可选）
    if args.all_formats:
        sdf_file = f"{output_base}.sdf"
        try:
            writer = Chem.SDWriter(sdf_file)
            writer.write(mol)
            writer.close()
            print(f"SDF 文件：{sdf_file}")
        except Exception as e:
            print(f"SDF 输出失败：{e}")

    # 验证结构
    print("\n结构验证...")
    issues = validate_structure(mol)
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("结构检查通过")

    # 输出分子信息
    if args.verbose:
        print_molecule_info(mol)

    # 生成下一步命令提示
    print("\n" + "="*60)
    print("转换完成！")
    print("="*60)
    print("\n下一步：使用 ACPYPE 生成力场参数")
    print(f"\n  acpype -i {mol2_file} -c bcc -n {charge}")
    print(f"\n然后运行 REST2：")
    print(f"\n  python bin/launch_REST2_small_molecule.py \\")
    print(f"      -pdb {pdb_file} \\")
    print(f"      -n {output_base}_rest2 \\")
    print(f"      -o output \\")
    print(f"      --extra-ff {output_base}.acpype/{output_base}_GMX.xml \\")
    print(f"      --solute-residues {args.residue_name} \\")
    print(f"      --time 10 \\")
    print(f"      --minimize\n")


if __name__ == "__main__":
    main()
