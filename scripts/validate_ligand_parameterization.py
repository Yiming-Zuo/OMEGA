#!/usr/bin/env python
"""
验证 FXR 配体是否可以被开源力场参数化（优化版 v2）

新增功能：
- 基于 pKa 预测生理 pH (7.4) 下的电离状态
- 检测酸性基团（羧酸、磺酸等）和碱性基团（胺等）
- 输出正确的生理 pH 电荷

参考：
- D3R GC2016 官方文档明确说明 SMILES 未处理质子化状态
- 实验条件：50 mM HEPES buffer, pH 7.4

依赖：
- rdkit
- openff-toolkit
"""

import csv
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

def log(msg: str):
    """带时间戳的日志输出"""
    print(msg, flush=True)

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
    log("[OK] RDKit 已加载")
except ImportError:
    RDKIT_AVAILABLE = False
    log("[FAIL] RDKit 未安装")

# OpenFF
try:
    from openff.toolkit import Molecule, ForceField, Topology
    from openff.toolkit.typing.engines.smirnoff.parameters import UnassignedValenceParameterException
    OPENFF_AVAILABLE = True
    log("[OK] OpenFF Toolkit 已加载")
except ImportError:
    OPENFF_AVAILABLE = False
    log("[FAIL] OpenFF Toolkit 未安装")


# 常见官能团的 pKa 值（用于 pH 7.4 下的电离状态预测）
# 参考: Chemicalize, DrugBank, 教科书数据
# 注意：这些是典型值，实际 pKa 受取代基效应影响
PKA_DATA = {
    # 酸性基团 (pKa < 7.4 时去质子化带负电)
    'carboxylic_acid': {'smarts': '[CX3](=O)[OX2H1]', 'pka': 4.5, 'charge_change': -1},
    'sulfate_ester': {'smarts': '[OX2][SX4](=[OX1])(=[OX1])[OX2H1]', 'pka': -3.0, 'charge_change': -1},  # -OSO3H
    'sulfonic_acid': {'smarts': '[CX4,c][SX4](=[OX1])(=[OX1])[OX2H1]', 'pka': -1.0, 'charge_change': -1},  # C-SO3H
    'phosphoric_acid': {'smarts': '[PX4](=O)([OX2H1])([OX2H1])[OX2H1]', 'pka': 2.1, 'charge_change': -1},
    'tetrazole': {'smarts': 'c1nnn[nH]1', 'pka': 4.9, 'charge_change': -1},  # 四唑，酸性

    # 不电离的基团 (记录但不改变电荷)
    'sulfonamide_nh2': {'smarts': '[SX4](=O)(=O)[NX3H2]', 'pka': 10.0, 'charge_change': 0},  # 弱酸性，pH 7.4不电离
    'phenol': {'smarts': '[OX2H1]c1ccccc1', 'pka': 10.0, 'charge_change': 0},  # 通常不去质子化

    # 碱性基团 (pKa > 7.4 时质子化带正电)
    # 注意：排除磺酰胺和酰胺中的 NH2，它们是弱酸性的
    'aliphatic_amine': {'smarts': '[NX3H2;!$(NC=O);!$(NS(=O)=O)]', 'pka': 10.5, 'charge_change': +1},
    'guanidine': {'smarts': '[NX3H2][CX3](=[NX2H1])[NX3H2]', 'pka': 12.5, 'charge_change': +1},

    # 弱碱，pH 7.4 下不质子化
    'aromatic_amine': {'smarts': '[nX2H1]', 'pka': 5.0, 'charge_change': 0},
    'imidazole': {'smarts': 'c1cnc[nH]1', 'pka': 6.0, 'charge_change': 0},
    'pyridine': {'smarts': 'n1ccccc1', 'pka': 5.2, 'charge_change': 0},
}

PHYSIOLOGICAL_PH = 7.4


@dataclass
class LigandResult:
    """单个配体的验证结果"""
    compound_id: str
    smiles: str
    ic50: Optional[float] = None
    ic50_modifier: str = ""
    ligand_class: str = ""

    # RDKit 验证
    rdkit_ok: bool = False
    rdkit_error: str = ""
    num_atoms: int = 0
    num_heavy_atoms: int = 0
    mol_weight: float = 0.0

    # 电荷分析
    formal_charge: int = 0  # SMILES 中的形式电荷
    physiological_charge: int = 0  # 生理 pH 下的预测电荷
    ionizable_groups: List[str] = field(default_factory=list)

    # OpenFF 验证
    openff_ok: bool = False
    openff_error: str = ""


def analyze_ionization_state(mol, ph: float = PHYSIOLOGICAL_PH) -> Tuple[int, List[str]]:
    """
    分析分子在指定 pH 下的电离状态

    返回：(预测的净电荷, 可电离基团列表)
    """
    # 先计算 SMILES 中已有的形式电荷
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    ionizable_groups = []
    charge_adjustment = 0

    for group_name, group_info in PKA_DATA.items():
        pattern = Chem.MolFromSmarts(group_info['smarts'])
        if pattern is None:
            continue

        matches = mol.GetSubstructMatches(pattern)
        n_matches = len(matches)

        if n_matches > 0:
            pka = group_info['pka']
            charge_change = group_info['charge_change']

            # 根据 Henderson-Hasselbalch 方程判断电离状态
            # 对于酸：pH > pKa 时去质子化（带负电）
            # 对于碱：pH < pKa 时质子化（带正电）

            if charge_change < 0:  # 酸性基团
                if ph > pka + 1:  # pH 比 pKa 高 1 个单位以上，几乎完全去质子化
                    charge_adjustment += charge_change * n_matches
                    ionizable_groups.append(f"{group_name}({n_matches}x, pKa={pka}, -> {charge_change:+d})")
                elif ph > pka:  # 部分去质子化，简化处理为去质子化
                    charge_adjustment += charge_change * n_matches
                    ionizable_groups.append(f"{group_name}({n_matches}x, pKa={pka}, -> {charge_change:+d})")

            elif charge_change > 0:  # 碱性基团
                if ph < pka - 1:  # pH 比 pKa 低 1 个单位以上，几乎完全质子化
                    charge_adjustment += charge_change * n_matches
                    ionizable_groups.append(f"{group_name}({n_matches}x, pKa={pka}, -> {charge_change:+d})")

    # 最终电荷 = 形式电荷 + 电离调整
    predicted_charge = formal_charge + charge_adjustment

    return predicted_charge, ionizable_groups


def validate_with_rdkit(smiles: str) -> Tuple[bool, str, dict]:
    """用 RDKit 验证 SMILES 并生成 3D 构象"""
    info = {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "无法解析 SMILES", info

    # 计算信息（在加氢前）
    info['formal_charge'] = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    # 分析电离状态
    predicted_charge, ionizable_groups = analyze_ionization_state(mol)
    info['physiological_charge'] = predicted_charge
    info['ionizable_groups'] = ionizable_groups

    # 添加氢原子
    mol = Chem.AddHs(mol)

    info['num_atoms'] = mol.GetNumAtoms()
    info['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
    info['mol_weight'] = Descriptors.MolWt(mol)

    # 生成 3D 构象
    try:
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.useSmallRingTorsions = True
            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                return False, "无法生成 3D 构象", info

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
        except:
            pass

        info['mol'] = mol

    except Exception as e:
        return False, f"3D 构象错误: {str(e)[:50]}", info

    return True, "", info


def validate_with_openff(rdkit_mol, forcefield) -> Tuple[bool, str]:
    """用 OpenFF label_molecules() 快速验证参数覆盖"""
    try:
        off_mol = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)
        topology = off_mol.to_topology()
        labels = forcefield.label_molecules(topology)
        return True, ""

    except UnassignedValenceParameterException as e:
        return False, "缺少价键参数"

    except Exception as e:
        return False, str(e)[:100]


def parse_ic50(value: str) -> Optional[float]:
    if value == "NA" or value == "" or value is None:
        return None
    try:
        return float(value)
    except:
        return None


def main():
    log("\n" + "="*80)
    log("FXR 配体力场参数化验证 v2（含电离状态预测）")
    log("="*80)
    log(f"生理 pH: {PHYSIOLOGICAL_PH}")
    log("参考: D3R GC2016 官方文档 - SMILES 未处理质子化状态")

    if not RDKIT_AVAILABLE or not OPENFF_AVAILABLE:
        log("\n[FAIL] 缺少必要依赖")
        sys.exit(1)

    log("\n加载 OpenFF Sage 2.2.0 力场...")
    start = time.time()
    forcefield = ForceField("openff-2.2.0.offxml")
    log(f"[OK] 力场加载完成 ({time.time()-start:.1f}s)")

    csv_path = Path(__file__).parent.parent / "FXR_experimental_data/Processed_Data/FXR_Affinities_20170210.csv"

    if not csv_path.exists():
        log(f"\n[FAIL] CSV 文件不存在: {csv_path}")
        sys.exit(1)

    log(f"\n读取数据: {csv_path.name}")

    results = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            compound_id = row['Compound ID'].strip()
            smiles = row['Smiles'].strip()

            if compound_id == "Apo" or not smiles:
                continue

            result = LigandResult(
                compound_id=compound_id,
                smiles=smiles,
                ic50=parse_ic50(row.get('IC50_SPA_BINDING [uM]', '')),
                ic50_modifier=row.get('IC50_SPA_BINDING:MODIFIER', ''),
                ligand_class=row.get('Class', '')
            )
            results.append(result)

    total = len(results)
    log(f"共 {total} 个配体待验证\n")

    # 表头
    log("-"*95)
    log(f"{'#':<4} {'ID':<12} {'RDKit':<7} {'OpenFF':<7} {'原子':<5} {'SMILES':<7} {'pH7.4':<7} {'MW':<8} {'类别'}")
    log(f"{'':4} {'':12} {'':7} {'':7} {'':5} {'电荷':<7} {'电荷':<7} {'':8} {''}")
    log("-"*95)

    rdkit_success = 0
    openff_success = 0
    start_time = time.time()

    for i, result in enumerate(results, 1):
        ok, error, info = validate_with_rdkit(result.smiles)
        result.rdkit_ok = ok
        result.rdkit_error = error

        if ok:
            rdkit_success += 1
            result.num_atoms = info.get('num_atoms', 0)
            result.num_heavy_atoms = info.get('num_heavy_atoms', 0)
            result.mol_weight = info.get('mol_weight', 0)
            result.formal_charge = info.get('formal_charge', 0)
            result.physiological_charge = info.get('physiological_charge', 0)
            result.ionizable_groups = info.get('ionizable_groups', [])

            mol = info.get('mol')
            if mol:
                ok2, error2 = validate_with_openff(mol, forcefield)
                result.openff_ok = ok2
                result.openff_error = error2
                if ok2:
                    openff_success += 1

        # 打印结果
        rdkit_status = "[OK]" if result.rdkit_ok else "[FAIL]"
        openff_status = "[OK]" if result.openff_ok else "[FAIL]" if result.rdkit_ok else "  -  "

        formal_str = f"{result.formal_charge:+d}" if result.formal_charge != 0 else "0"
        physio_str = f"{result.physiological_charge:+d}" if result.physiological_charge != 0 else "0"

        log(f"{i:<4} {result.compound_id:<12} {rdkit_status:<7} {openff_status:<7} "
            f"{result.num_atoms:<5} {formal_str:<7} {physio_str:<7} {result.mol_weight:<8.1f} {result.ligand_class}")

        # 如果有电离基团，打印详情
        if result.ionizable_groups:
            for group in result.ionizable_groups:
                log(f"     └─ {group}")

        if not result.rdkit_ok:
            log(f"     └─ RDKit: {result.rdkit_error}")
        elif not result.openff_ok:
            log(f"     └─ OpenFF: {result.openff_error}")

    elapsed = time.time() - start_time

    # 统计
    log("\n" + "="*80)
    log("验证结果统计")
    log("="*80)
    log(f"总配体数:        {total}")
    log(f"RDKit 成功:      {rdkit_success}/{total} ({100*rdkit_success/total:.1f}%)")
    log(f"OpenFF 成功:     {openff_success}/{total} ({100*openff_success/total:.1f}%)")
    log(f"验证耗时:        {elapsed:.1f} 秒")

    # 按电荷统计
    log("\n" + "="*80)
    log("生理 pH 7.4 下的电荷分布")
    log("="*80)

    charge_stats = {}
    for r in results:
        if r.rdkit_ok:
            c = r.physiological_charge
            if c not in charge_stats:
                charge_stats[c] = []
            charge_stats[c].append(r.compound_id)

    log(f"{'电荷':<8} {'数量':<8} {'配体 ID'}")
    log("-"*70)
    for charge in sorted(charge_stats.keys()):
        ids = charge_stats[charge]
        # 只显示前几个 ID
        if len(ids) > 8:
            id_str = ", ".join(ids[:8]) + f" ... (+{len(ids)-8} more)"
        else:
            id_str = ", ".join(ids)
        log(f"{charge:+d}       {len(ids):<8} {id_str}")

    # 按类别和电荷统计
    log("\n按化学类别和电荷统计:")
    class_charge_stats = {}
    for r in results:
        if r.rdkit_ok:
            cls = r.ligand_class or "unknown"
            if cls not in class_charge_stats:
                class_charge_stats[cls] = {}
            c = r.physiological_charge
            class_charge_stats[cls][c] = class_charge_stats[cls].get(c, 0) + 1

    log(f"{'类别':<20} {'中性(0)':<10} {'带电(-1)':<10} {'带电(-2)':<10} {'其他'}")
    log("-"*60)
    for cls in sorted(class_charge_stats.keys()):
        stats = class_charge_stats[cls]
        n0 = stats.get(0, 0)
        nm1 = stats.get(-1, 0)
        nm2 = stats.get(-2, 0)
        other = sum(v for k, v in stats.items() if k not in [0, -1, -2])
        log(f"{cls:<20} {n0:<10} {nm1:<10} {nm2:<10} {other}")

    # 保存结果
    output_path = Path(__file__).parent.parent / "results/ligand_parameterization_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Compound_ID', 'SMILES', 'IC50_uM', 'IC50_Modifier', 'Class',
            'RDKit_OK', 'OpenFF_OK', 'Num_Atoms', 'MW',
            'SMILES_Charge', 'Physiological_Charge_pH7.4', 'Ionizable_Groups'
        ])
        for r in results:
            writer.writerow([
                r.compound_id, r.smiles, r.ic50 or '', r.ic50_modifier, r.ligand_class,
                r.rdkit_ok, r.openff_ok, r.num_atoms, f"{r.mol_weight:.2f}",
                r.formal_charge, r.physiological_charge, "; ".join(r.ionizable_groups)
            ])

    log(f"\n详细结果已保存: {output_path}")

    # 最终总结
    charged_ligands = [r for r in results if r.rdkit_ok and r.physiological_charge != 0]
    neutral_ligands = [r for r in results if r.rdkit_ok and r.physiological_charge == 0]

    log("\n" + "="*80)
    log("最终总结")
    log("="*80)
    log(f"[OK] 可用于溶剂化采样的配体: {openff_success}/{total}")
    log(f"")
    log(f"电荷分布 (pH 7.4):")
    log(f"  - 中性配体:     {len(neutral_ligands)} 个")
    log(f"  - 带电配体:     {len(charged_ligands)} 个")

    if charged_ligands:
        log(f"\n[WARN] 带电配体需要在溶剂化时添加反离子中和系统！")
        log(f"       这些配体含有羧酸等可电离基团，在 pH 7.4 下会去质子化。")

    log("\n" + "="*80)


if __name__ == "__main__":
    main()
