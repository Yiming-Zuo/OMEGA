#!/usr/bin/env python
"""
使用多种工具交叉验证 FXR 配体的电离状态预测

工具：
1. SMARTS 匹配（我们的方法）
2. Dimorphite-DL（专业 pH 电离预测）
3. OpenBabel（obabel 的 pH 模型）

目标：验证 pH 7.4 下的电荷预测是否可靠
"""

import csv
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

def log(msg: str):
    print(msg, flush=True)

# ============================================================================
# 导入依赖
# ============================================================================
log("="*80)
log("加载依赖")
log("="*80)

# RDKit
from rdkit import Chem
log("[OK] RDKit")

# Dimorphite-DL
try:
    from dimorphite_dl import protonate_smiles
    DIMORPHITE_OK = True
    log("[OK] Dimorphite-DL")
except ImportError as e:
    DIMORPHITE_OK = False
    log(f"[FAIL] Dimorphite-DL: {e}")

# OpenBabel
try:
    from openbabel import openbabel as ob
    OPENBABEL_OK = True
    log("[OK] OpenBabel")
except ImportError as e:
    OPENBABEL_OK = False
    log(f"[FAIL] OpenBabel: {e}")


# ============================================================================
# SMARTS 方法（我们的实现）
# ============================================================================
PKA_DATA = {
    'carboxylic_acid': {'smarts': '[CX3](=O)[OX2H1]', 'pka': 4.5, 'charge_change': -1},
    'sulfate_ester': {'smarts': '[OX2][SX4](=[OX1])(=[OX1])[OX2H1]', 'pka': -3.0, 'charge_change': -1},
    'sulfonic_acid': {'smarts': '[CX4,c][SX4](=[OX1])(=[OX1])[OX2H1]', 'pka': -1.0, 'charge_change': -1},
    'tetrazole': {'smarts': 'c1nnn[nH]1', 'pka': 4.9, 'charge_change': -1},
    'aliphatic_amine': {'smarts': '[NX3H2;!$(NC=O);!$(NS(=O)=O)]', 'pka': 10.5, 'charge_change': +1},
}

def predict_charge_smarts(smiles: str, ph: float = 7.4) -> Tuple[int, List[str]]:
    """用 SMARTS 预测电荷"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0, ["无法解析"]

    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    groups = []
    charge_adj = 0

    for name, info in PKA_DATA.items():
        pattern = Chem.MolFromSmarts(info['smarts'])
        if pattern:
            n = len(mol.GetSubstructMatches(pattern))
            if n > 0:
                pka = info['pka']
                dc = info['charge_change']
                if dc < 0 and ph > pka:  # 酸性基团去质子化
                    charge_adj += dc * n
                    groups.append(f"{name}({n})")
                elif dc > 0 and ph < pka:  # 碱性基团质子化
                    charge_adj += dc * n
                    groups.append(f"{name}({n})")

    return formal_charge + charge_adj, groups


# ============================================================================
# Dimorphite-DL 方法
# ============================================================================
def predict_charge_dimorphite(smiles: str, ph: float = 7.4) -> Tuple[List[int], List[str]]:
    """用 Dimorphite-DL 预测电荷，返回所有可能的电荷状态"""
    if not DIMORPHITE_OK:
        return [], ["不可用"]

    try:
        # Dimorphite-DL 返回质子化后的 SMILES 列表
        # 新版 API: ph_min, ph_max (旧版是 min_ph, max_ph)
        protonated = protonate_smiles(smiles, ph_min=ph, ph_max=ph)

        if not protonated:
            return [], ["无输出"]

        # 解析所有结果，获取所有可能的电荷
        charges = []
        valid_smiles = []
        for prot_smiles in protonated:
            mol = Chem.MolFromSmiles(prot_smiles)
            if mol is not None:
                charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
                charges.append(charge)
                valid_smiles.append(prot_smiles)

        return charges, valid_smiles

    except Exception as e:
        return [], [str(e)[:30]]


# ============================================================================
# OpenBabel 方法
# ============================================================================
def predict_charge_openbabel(smiles: str, ph: float = 7.4) -> Tuple[int, str]:
    """用 OpenBabel 预测电荷"""
    if not OPENBABEL_OK:
        return None, "不可用"

    try:
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("smi", "smi")

        mol = ob.OBMol()
        obConversion.ReadString(mol, smiles)

        # 添加氢并设置 pH
        mol.AddHydrogens(False, True, ph)  # polarOnly=False, correctForPH=True

        # 计算形式电荷
        total_charge = mol.GetTotalCharge()

        # 输出 SMILES
        out_smiles = obConversion.WriteString(mol).strip()

        return int(total_charge), out_smiles

    except Exception as e:
        return None, str(e)[:30]


# ============================================================================
# 主程序
# ============================================================================
@dataclass
class ValidationResult:
    compound_id: str
    smiles: str
    smarts_charge: int
    smarts_groups: str
    dimorphite_charges: List[int]  # 所有可能的电荷
    dimorphite_smiles: List[str]
    openbabel_charge: Optional[int]
    openbabel_smiles: str
    smarts_in_dimorphite: bool  # SMARTS 结果是否在 Dimorphite 列表中
    ob_in_dimorphite: bool      # OpenBabel 结果是否在 Dimorphite 列表中
    smarts_ob_agree: bool       # SMARTS 和 OpenBabel 是否一致


def main():
    log("\n" + "="*80)
    log("FXR 配体电离状态交叉验证")
    log("="*80)
    log(f"目标 pH: 7.4")
    log(f"验证策略: 检查 SMARTS/OpenBabel 结果是否在 Dimorphite-DL 可能列表中")

    # 读取数据
    csv_path = Path(__file__).parent.parent / "FXR_experimental_data/Processed_Data/FXR_Affinities_20170210.csv"

    ligands = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['Compound ID'].strip()
            smiles = row['Smiles'].strip()
            if cid != "Apo" and smiles:
                ligands.append((cid, smiles))

    log(f"\n共 {len(ligands)} 个配体\n")

    # 验证
    results = []

    log("-"*110)
    log(f"{'ID':<12} {'SMARTS':<8} {'OpenBabel':<10} {'Dimorphite可能值':<20} {'S在D中?':<8} {'O在D中?':<8} {'S=O?':<6} {'基团'}")
    log("-"*110)

    # 统计计数
    smarts_in_dim_count = 0
    ob_in_dim_count = 0
    smarts_ob_agree_count = 0
    all_agree_count = 0

    for cid, smiles in ligands:
        # SMARTS 方法
        smarts_charge, smarts_groups = predict_charge_smarts(smiles)

        # Dimorphite-DL 方法 (返回所有可能电荷)
        dim_charges, dim_smiles_list = predict_charge_dimorphite(smiles)

        # OpenBabel 方法
        ob_charge, ob_smiles = predict_charge_openbabel(smiles)

        # 检查一致性
        smarts_in_dim = smarts_charge in dim_charges if dim_charges else False
        ob_in_dim = (ob_charge in dim_charges) if (ob_charge is not None and dim_charges) else False
        smarts_ob_agree = (smarts_charge == ob_charge) if ob_charge is not None else False

        if smarts_in_dim:
            smarts_in_dim_count += 1
        if ob_in_dim:
            ob_in_dim_count += 1
        if smarts_ob_agree:
            smarts_ob_agree_count += 1
        if smarts_in_dim and ob_in_dim and smarts_ob_agree:
            all_agree_count += 1

        result = ValidationResult(
            compound_id=cid,
            smiles=smiles,
            smarts_charge=smarts_charge,
            smarts_groups=", ".join(smarts_groups) if smarts_groups else "-",
            dimorphite_charges=dim_charges,
            dimorphite_smiles=dim_smiles_list,
            openbabel_charge=ob_charge,
            openbabel_smiles=ob_smiles,
            smarts_in_dimorphite=smarts_in_dim,
            ob_in_dimorphite=ob_in_dim,
            smarts_ob_agree=smarts_ob_agree
        )
        results.append(result)

        # 打印
        smarts_str = f"{smarts_charge:+d}"
        ob_str = f"{ob_charge:+d}" if ob_charge is not None else "N/A"
        dim_str = ",".join(f"{c:+d}" for c in dim_charges) if dim_charges else "N/A"
        s_in_d = "[OK]" if smarts_in_dim else "[X]"
        o_in_d = "[OK]" if ob_in_dim else "[X]"
        s_eq_o = "[OK]" if smarts_ob_agree else "[X]"

        log(f"{cid:<12} {smarts_str:<8} {ob_str:<10} {dim_str:<20} {s_in_d:<8} {o_in_d:<8} {s_eq_o:<6} {result.smarts_groups}")

    # 统计
    log("\n" + "="*80)
    log("验证结果统计")
    log("="*80)

    log(f"总配体数:                    {len(ligands)}")
    log(f"SMARTS = OpenBabel:          {smarts_ob_agree_count}/{len(ligands)} ({100*smarts_ob_agree_count/len(ligands):.1f}%)")
    log(f"SMARTS 在 Dimorphite 中:     {smarts_in_dim_count}/{len(ligands)} ({100*smarts_in_dim_count/len(ligands):.1f}%)")
    log(f"OpenBabel 在 Dimorphite 中:  {ob_in_dim_count}/{len(ligands)} ({100*ob_in_dim_count/len(ligands):.1f}%)")
    log(f"三者完全一致:                {all_agree_count}/{len(ligands)} ({100*all_agree_count/len(ligands):.1f}%)")

    # 列出不一致的情况
    not_in_dim = [r for r in results if not r.smarts_in_dimorphite]
    if not_in_dim:
        log(f"\n[WARN] SMARTS 结果不在 Dimorphite 列表中的配体 ({len(not_in_dim)} 个):")
        log(f"{'ID':<12} {'SMARTS':<8} {'Dimorphite可能值':<25} {'基团'}")
        log("-"*70)
        for r in not_in_dim:
            dim_str = ",".join(f"{c:+d}" for c in r.dimorphite_charges) if r.dimorphite_charges else "N/A"
            log(f"{r.compound_id:<12} {r.smarts_charge:+d}      {dim_str:<25} {r.smarts_groups}")

    smarts_ob_diff = [r for r in results if not r.smarts_ob_agree]
    if smarts_ob_diff:
        log(f"\n[WARN] SMARTS 和 OpenBabel 不一致的配体 ({len(smarts_ob_diff)} 个):")
        log(f"{'ID':<12} {'SMARTS':<8} {'OpenBabel':<10} {'基团'}")
        log("-"*50)
        for r in smarts_ob_diff:
            ob_str = f"{r.openbabel_charge:+d}" if r.openbabel_charge is not None else "N/A"
            log(f"{r.compound_id:<12} {r.smarts_charge:+d}      {ob_str:<10} {r.smarts_groups}")

    # 按电荷统计
    log("\n" + "="*80)
    log("各方法电荷分布对比")
    log("="*80)

    from collections import Counter

    smarts_dist = Counter(r.smarts_charge for r in results)
    # Dimorphite: 统计所有可能电荷的出现次数
    dim_all_charges = []
    for r in results:
        dim_all_charges.extend(r.dimorphite_charges)
    dim_dist = Counter(dim_all_charges)
    ob_dist = Counter(r.openbabel_charge for r in results if r.openbabel_charge is not None)

    all_charges = sorted(set(smarts_dist.keys()) | set(dim_dist.keys()) | set(ob_dist.keys()))

    log(f"{'电荷':<8} {'SMARTS':<12} {'Dimorphite(总)':<15} {'OpenBabel'}")
    log("-"*55)
    for c in all_charges:
        log(f"{c:+d}       {smarts_dist.get(c, 0):<12} {dim_dist.get(c, 0):<15} {ob_dist.get(c, 0)}")

    # 保存结果
    output_path = Path(__file__).parent.parent / "results/ionization_cross_validation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Compound_ID', 'Original_SMILES',
            'SMARTS_Charge', 'SMARTS_Groups',
            'Dimorphite_Charges', 'Dimorphite_Count',
            'OpenBabel_Charge',
            'SMARTS_in_Dimorphite', 'OpenBabel_in_Dimorphite', 'SMARTS_eq_OpenBabel'
        ])
        for r in results:
            writer.writerow([
                r.compound_id, r.smiles,
                r.smarts_charge, r.smarts_groups,
                ";".join(str(c) for c in r.dimorphite_charges), len(r.dimorphite_charges),
                r.openbabel_charge if r.openbabel_charge is not None else '',
                r.smarts_in_dimorphite, r.ob_in_dimorphite, r.smarts_ob_agree
            ])

    log(f"\n详细结果已保存: {output_path}")
    log("="*80)


if __name__ == "__main__":
    main()
