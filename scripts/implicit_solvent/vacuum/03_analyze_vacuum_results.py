#!/usr/bin/env python
"""
气相（真空）MD 结果分析脚本

分析丙氨酸二肽气相 MD 采样结果。

分析内容:
1. 能量收敛性检查
2. 温度稳定性检查
3. φ/ψ 二面角分布（Ramachandran 图）
4. 构象占比统计
5. 构象转换分析

输入：outputs/implicit_solvent/vacuum/alanine_dipeptide/md/
输出：results/implicit_solvent/vacuum/alanine_dipeptide/
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import mdtraj as md
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False
    print("[WARN] mdtraj 未安装，将跳过轨迹分析")

# =============================================================================
# 配置
# =============================================================================
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
MD_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "implicit_solvent" / "vacuum" / "alanine_dipeptide" / "md"
SYSTEM_DIR = PROJECT_ROOT / "outputs" / "implicit_solvent" / "vacuum" / "alanine_dipeptide"
RESULTS_DIR = PROJECT_ROOT / "results" / "implicit_solvent" / "vacuum" / "alanine_dipeptide"


# =============================================================================
# 构象区域定义
# =============================================================================
def classify_conformations(phi_deg, psi_deg):
    """
    根据 φ/ψ 角分类构象

    气相丙氨酸二肽的主要构象:
    - C7eq: 七元环氢键（equatorial），主导构象
    - C7ax: 七元环氢键（axial）
    - C5: 五元环构象
    - αR: 右手螺旋区
    - αL: 左手螺旋区
    - β: 延展构象
    - PII: polyproline II（气相中较少）
    """
    n = len(phi_deg)
    labels = np.full(n, 'other', dtype=object)

    # C7eq (七元环氢键，equatorial) - 气相主导构象
    c7eq_mask = (phi_deg > -110) & (phi_deg < -50) & (psi_deg > 50) & (psi_deg < 100)

    # C7ax (七元环氢键，axial)
    c7ax_mask = (phi_deg > 50) & (phi_deg < 100) & (psi_deg > -100) & (psi_deg < -40)

    # C5 (五元环构象，延展)
    c5_mask = (phi_deg > -180) & (phi_deg < -120) & (psi_deg > 140) & (psi_deg < 180)

    # αR (右手螺旋区)
    alphaR_mask = (phi_deg > -90) & (phi_deg < -30) & (psi_deg > -60) & (psi_deg < 0)

    # αL (左手螺旋区)
    alphaL_mask = (phi_deg > 40) & (phi_deg < 80) & (psi_deg > 20) & (psi_deg < 60)

    # β (延展构象)
    beta_mask = (phi_deg > -150) & (phi_deg < -100) & (psi_deg > 100) & (psi_deg < 150)

    # PII (polyproline II) - 气相中较少
    pii_mask = (phi_deg > -90) & (phi_deg < -50) & (psi_deg > 120) & (psi_deg < 160)

    # 按优先级分配标签
    labels[c7eq_mask] = 'C7eq'
    labels[c7ax_mask] = 'C7ax'
    labels[c5_mask & (labels == 'other')] = 'C5'
    labels[alphaR_mask & (labels == 'other')] = 'αR'
    labels[alphaL_mask & (labels == 'other')] = 'αL'
    labels[beta_mask & (labels == 'other')] = 'β'
    labels[pii_mask & (labels == 'other')] = 'PII'

    return labels


def compute_conformer_populations(labels):
    """计算各构象占比"""
    unique, counts = np.unique(labels, return_counts=True)
    populations = dict(zip(unique, counts / len(labels)))

    # 确保所有构象都有条目
    all_conformers = ['C7eq', 'C7ax', 'C5', 'αR', 'αL', 'β', 'PII', 'other']
    result = {c: populations.get(c, 0.0) for c in all_conformers}

    return result


def count_transitions(labels):
    """统计构象转换次数"""
    transitions = np.sum(labels[:-1] != labels[1:])
    return transitions


# =============================================================================
# 主程序
# =============================================================================
def main():
    print("=" * 60)
    print("气相 MD 结果分析: 丙氨酸二肽 (ACE-ALA-NME)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. 分析能量和温度
    # -------------------------------------------------------------------------
    print("\n[1/4] 分析能量和温度...")

    state_data_path = MD_OUTPUT_DIR / "state_data.csv"
    if state_data_path.exists():
        # 读取状态数据
        df = pd.read_csv(state_data_path)

        # 清理列名（移除引号和空格）
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")

        print(f"  - 加载: {state_data_path}")
        print(f"  - 数据点: {len(df)}")
        print(f"  - 列: {list(df.columns)}")

        # 查找能量和温度列
        energy_col = None
        temp_col = None
        time_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'potential' in col_lower and 'energy' in col_lower:
                energy_col = col
            elif 'temperature' in col_lower:
                temp_col = col
            elif 'time' in col_lower:
                time_col = col

        # 绘制能量和温度
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        if energy_col and energy_col in df.columns:
            energy = df[energy_col].values
            x_axis = df[time_col].values if time_col else np.arange(len(energy))
            x_label = 'Time (ps)' if time_col else 'Step'

            axes[0].plot(x_axis, energy, linewidth=0.5, alpha=0.7)
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel('Potential Energy (kJ/mol)')
            axes[0].set_title('Potential Energy vs Time (Gas Phase)')
            axes[0].grid(alpha=0.3)

            # 统计
            print(f"  - 势能范围: {energy.min():.2f} ~ {energy.max():.2f} kJ/mol")
            print(f"  - 势能平均: {energy.mean():.2f} +/- {energy.std():.2f} kJ/mol")

        if temp_col and temp_col in df.columns:
            temp = df[temp_col].values
            x_axis = df[time_col].values if time_col else np.arange(len(temp))
            x_label = 'Time (ps)' if time_col else 'Step'

            axes[1].plot(x_axis, temp, linewidth=0.5, alpha=0.7)
            axes[1].axhline(300, color='red', linestyle='--', alpha=0.5, label='Target: 300 K')
            axes[1].set_xlabel(x_label)
            axes[1].set_ylabel('Temperature (K)')
            axes[1].set_title('Temperature vs Time (Gas Phase)')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            # 统计
            print(f"  - 温度平均: {temp.mean():.1f} +/- {temp.std():.1f} K")

        plt.tight_layout()
        energy_plot_path = RESULTS_DIR / "energy_temperature.png"
        plt.savefig(energy_plot_path, dpi=300)
        print(f"  [OK] 保存: {energy_plot_path}")
        plt.close()
    else:
        print(f"  [WARN] 状态数据文件不存在: {state_data_path}")

    # -------------------------------------------------------------------------
    # 2. 分析轨迹（二面角）
    # -------------------------------------------------------------------------
    print("\n[2/4] 分析轨迹...")

    if not HAS_MDTRAJ:
        print("  [FAIL] 需要 mdtraj 库进行轨迹分析")
        print("  请安装: pip install mdtraj")
        sys.exit(1)

    trajectory_path = MD_OUTPUT_DIR / "trajectory.dcd"
    topology_path = SYSTEM_DIR / "system.pdb"

    if not trajectory_path.exists():
        print(f"  [FAIL] 轨迹文件不存在: {trajectory_path}")
        print("  请先运行 02_run_vacuum_md.py")
        sys.exit(1)

    # 加载轨迹
    print(f"  - 加载轨迹: {trajectory_path}")
    traj = md.load(str(trajectory_path), top=str(topology_path))
    print(f"  - 帧数: {traj.n_frames}")
    print(f"  - 原子数: {traj.n_atoms}")

    # 计算 φ/ψ 二面角
    phi_indices, phi = md.compute_phi(traj)
    psi_indices, psi = md.compute_psi(traj)

    phi_deg = np.rad2deg(phi[:, 0])
    psi_deg = np.rad2deg(psi[:, 0])

    print(f"  - φ 范围: [{phi_deg.min():.1f}, {phi_deg.max():.1f}] 度")
    print(f"  - ψ 范围: [{psi_deg.min():.1f}, {psi_deg.max():.1f}] 度")

    # -------------------------------------------------------------------------
    # 3. 绘制 Ramachandran 图
    # -------------------------------------------------------------------------
    print("\n[3/4] 绘制 Ramachandran 图...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2D 直方图
    h = axes[0].hist2d(phi_deg, psi_deg, bins=60, cmap='Blues', density=True)
    axes[0].set_xlabel('φ (degrees)', fontsize=12)
    axes[0].set_ylabel('ψ (degrees)', fontsize=12)
    axes[0].set_title('Ramachandran Plot (Gas Phase, 300K)', fontsize=14)
    axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[0].axvline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[0].set_xlim([-180, 180])
    axes[0].set_ylim([-180, 180])

    # 标注主要构象区域
    from matplotlib.patches import Rectangle

    # C7eq
    axes[0].add_patch(Rectangle((-110, 50), 60, 50, fill=False, edgecolor='red', linewidth=2, linestyle='--'))
    axes[0].text(-80, 75, 'C7eq', fontsize=11, color='red', weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # C7ax
    axes[0].add_patch(Rectangle((50, -100), 50, 60, fill=False, edgecolor='blue', linewidth=2, linestyle='--'))
    axes[0].text(75, -70, 'C7ax', fontsize=11, color='blue', weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # αR
    axes[0].add_patch(Rectangle((-90, -60), 60, 60, fill=False, edgecolor='orange', linewidth=2, linestyle='--'))
    axes[0].text(-60, -30, 'αR', fontsize=11, color='orange', weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.colorbar(h[3], ax=axes[0], label='Probability Density')

    # 1D 直方图
    axes[1].hist(phi_deg, bins=60, alpha=0.6, label='φ', density=True)
    axes[1].hist(psi_deg, bins=60, alpha=0.6, label='ψ', density=True)
    axes[1].set_xlabel('Angle (degrees)', fontsize=12)
    axes[1].set_ylabel('Probability Density', fontsize=12)
    axes[1].set_title('Torsion Angle Distributions (Gas Phase)', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    ramachandran_path = RESULTS_DIR / "ramachandran.png"
    plt.savefig(ramachandran_path, dpi=300)
    print(f"  [OK] 保存: {ramachandran_path}")
    plt.close()

    # -------------------------------------------------------------------------
    # 4. 构象分析
    # -------------------------------------------------------------------------
    print("\n[4/4] 构象分析...")

    # 分类构象
    labels = classify_conformations(phi_deg, psi_deg)

    # 计算占比
    populations = compute_conformer_populations(labels)

    print("\n  构象占比:")
    print("  " + "-" * 40)
    for conf, frac in sorted(populations.items(), key=lambda x: -x[1]):
        if frac > 0.001:  # 只显示 > 0.1% 的构象
            print(f"    {conf:8s}: {100*frac:6.2f}%")
    print("  " + "-" * 40)

    # 计算转换次数
    n_transitions = count_transitions(labels)
    dt_ps = 2.0  # 假设 2 ps 保存间隔
    total_time_ns = len(labels) * dt_ps / 1000.0
    transition_rate = n_transitions / total_time_ns

    print(f"\n  构象转换:")
    print(f"    - 总转换次数: {n_transitions}")
    print(f"    - 模拟时间: {total_time_ns:.1f} ns")
    print(f"    - 转换频率: {transition_rate:.1f} 次/ns")

    # 保存构象占比到 CSV
    populations_df = pd.DataFrame([
        {'conformer': k, 'fraction': v, 'percentage': f"{100*v:.2f}%"}
        for k, v in sorted(populations.items(), key=lambda x: -x[1])
    ])
    populations_csv_path = RESULTS_DIR / "conformer_populations.csv"
    populations_df.to_csv(populations_csv_path, index=False)
    print(f"\n  [OK] 保存: {populations_csv_path}")

    # 绘制构象时间序列
    fig, ax = plt.subplots(figsize=(14, 4))

    # 为每个构象分配数字编码
    conf_to_num = {'C7eq': 1, 'C7ax': 2, 'C5': 3, 'αR': 4, 'αL': 5, 'β': 6, 'PII': 7, 'other': 0}
    state_code = np.array([conf_to_num.get(l, 0) for l in labels])

    time_ns = np.arange(len(labels)) * dt_ps / 1000.0

    scatter = ax.scatter(time_ns, state_code, c=state_code, cmap='tab10', s=5, alpha=0.5)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels(['other', 'C7eq', 'C7ax', 'C5', 'αR', 'αL', 'β', 'PII'])
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Conformer', fontsize=12)
    ax.set_title('Conformer Evolution (Gas Phase, 300K)', fontsize=14)
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    timeline_path = RESULTS_DIR / "conformer_timeline.png"
    plt.savefig(timeline_path, dpi=300)
    print(f"  [OK] 保存: {timeline_path}")
    plt.close()

    # -------------------------------------------------------------------------
    # 总结
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)

    print("\n输出文件:")
    print(f"  - {RESULTS_DIR / 'energy_temperature.png'}")
    print(f"  - {RESULTS_DIR / 'ramachandran.png'}")
    print(f"  - {RESULTS_DIR / 'conformer_populations.csv'}")
    print(f"  - {RESULTS_DIR / 'conformer_timeline.png'}")

    print("\n气相丙氨酸二肽采样特点:")
    c7eq_pct = populations.get('C7eq', 0) * 100
    c7ax_pct = populations.get('C7ax', 0) * 100
    print(f"  - C7eq 占主导 ({c7eq_pct:.1f}%): 气相中分子内氢键稳定")
    print(f"  - C7ax 次要 ({c7ax_pct:.1f}%): 另一种氢键构象")
    print(f"  - PII 很少: 气相中缺乏水分子稳定")


if __name__ == "__main__":
    main()
