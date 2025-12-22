#!/usr/bin/env python
"""
隐式溶剂 MD 结果分析脚本（通用版）

支持气相（vacuum）和隐式溶剂（GBSA）两种模式。

使用方式:
    python 03_analyze_results.py --solvent vacuum
    python 03_analyze_results.py --solvent gbsa

分析内容:
1. 能量和温度收敛性
2. φ/ψ 二面角分布（Ramachandran 图）
3. 构象占比统计
4. 构象转换分析
"""

import argparse
import pathlib
import sys
import yaml
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


def load_config(solvent: str) -> dict:
    """加载溶剂配置文件"""
    config_path = pathlib.Path(__file__).parent / f"{solvent}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='隐式溶剂 MD 结果分析')
    parser.add_argument(
        '--solvent',
        choices=['vacuum', 'gbsa'],
        required=True,
        help='溶剂模型: vacuum (气相) 或 gbsa (隐式溶剂)'
    )
    return parser.parse_args()


def classify_conformations(phi_deg, psi_deg):
    """
    根据 φ/ψ 角分类构象

    主要构象:
    - C7eq: 七元环氢键（equatorial）
    - C7ax: 七元环氢键（axial）
    - C5: 五元环构象
    - αR: 右手螺旋区
    - αL: 左手螺旋区
    - β: 延展构象
    - PII: polyproline II
    """
    n = len(phi_deg)
    labels = np.full(n, 'other', dtype=object)

    # C7eq
    c7eq_mask = (phi_deg > -110) & (phi_deg < -50) & (psi_deg > 50) & (psi_deg < 100)
    # C7ax
    c7ax_mask = (phi_deg > 50) & (phi_deg < 100) & (psi_deg > -100) & (psi_deg < -40)
    # C5
    c5_mask = (phi_deg > -180) & (phi_deg < -120) & (psi_deg > 140) & (psi_deg < 180)
    # αR
    alphaR_mask = (phi_deg > -90) & (phi_deg < -30) & (psi_deg > -60) & (psi_deg < 0)
    # αL
    alphaL_mask = (phi_deg > 40) & (phi_deg < 80) & (psi_deg > 20) & (psi_deg < 60)
    # β
    beta_mask = (phi_deg > -150) & (phi_deg < -100) & (psi_deg > 100) & (psi_deg < 150)
    # PII
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

    all_conformers = ['C7eq', 'C7ax', 'C5', 'αR', 'αL', 'β', 'PII', 'other']
    result = {c: populations.get(c, 0.0) for c in all_conformers}

    return result


def main():
    args = parse_args()
    config = load_config(args.solvent)

    solvent_config = config['solvent']
    sim_config = config['simulation']
    output_config = config['output']

    project_root = pathlib.Path(__file__).resolve().parents[2]
    system_dir = project_root / output_config['system_dir']
    results_dir = project_root / output_config['results_dir']
    md_dir = system_dir / "md"

    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"结果分析: {solvent_config['name'].upper()} 模式")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. 分析能量和温度
    # -------------------------------------------------------------------------
    print("\n[1/4] 分析能量和温度...")

    state_data_path = md_dir / "state_data.csv"
    if state_data_path.exists():
        df = pd.read_csv(state_data_path)
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")

        print(f"  - 加载: {state_data_path}")
        print(f"  - 数据点: {len(df)}")

        # 查找列
        energy_col = next((c for c in df.columns if 'potential' in c.lower() and 'energy' in c.lower()), None)
        temp_col = next((c for c in df.columns if 'temperature' in c.lower()), None)
        time_col = next((c for c in df.columns if 'time' in c.lower()), None)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        if energy_col:
            energy = df[energy_col].values
            x_axis = df[time_col].values if time_col else np.arange(len(energy))
            x_label = 'Time (ps)' if time_col else 'Step'

            axes[0].plot(x_axis, energy, linewidth=0.5, alpha=0.7)
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel('Potential Energy (kJ/mol)')
            axes[0].set_title(f'Potential Energy ({solvent_config["name"].upper()})')
            axes[0].grid(alpha=0.3)

            print(f"  - 势能: {energy.mean():.2f} +/- {energy.std():.2f} kJ/mol")

        if temp_col:
            temp = df[temp_col].values
            x_axis = df[time_col].values if time_col else np.arange(len(temp))

            axes[1].plot(x_axis, temp, linewidth=0.5, alpha=0.7)
            axes[1].axhline(sim_config['temperature'], color='red', linestyle='--', alpha=0.5)
            axes[1].set_xlabel(x_label)
            axes[1].set_ylabel('Temperature (K)')
            axes[1].set_title(f'Temperature ({solvent_config["name"].upper()})')
            axes[1].grid(alpha=0.3)

            print(f"  - 温度: {temp.mean():.1f} +/- {temp.std():.1f} K")

        plt.tight_layout()
        energy_plot_path = results_dir / "energy_temperature.png"
        plt.savefig(energy_plot_path, dpi=300)
        print(f"  [OK] 保存: {energy_plot_path}")
        plt.close()
    else:
        print(f"  [WARN] 状态数据文件不存在: {state_data_path}")

    # -------------------------------------------------------------------------
    # 2. 分析轨迹
    # -------------------------------------------------------------------------
    print("\n[2/4] 分析轨迹...")

    if not HAS_MDTRAJ:
        print("  [FAIL] 需要 mdtraj 库")
        sys.exit(1)

    trajectory_path = md_dir / "trajectory.dcd"
    topology_path = system_dir / "system.pdb"

    if not trajectory_path.exists():
        print(f"  [FAIL] 轨迹文件不存在: {trajectory_path}")
        print(f"  请先运行: python 02_run_md.py --solvent {args.solvent}")
        sys.exit(1)

    print(f"  - 加载轨迹: {trajectory_path}")
    traj = md.load(str(trajectory_path), top=str(topology_path))
    print(f"  - 帧数: {traj.n_frames}")

    phi_indices, phi = md.compute_phi(traj)
    psi_indices, psi = md.compute_psi(traj)

    phi_deg = np.rad2deg(phi[:, 0])
    psi_deg = np.rad2deg(psi[:, 0])

    print(f"  - φ 范围: [{phi_deg.min():.1f}, {phi_deg.max():.1f}]")
    print(f"  - ψ 范围: [{psi_deg.min():.1f}, {psi_deg.max():.1f}]")

    # -------------------------------------------------------------------------
    # 3. 绘制 Ramachandran 图
    # -------------------------------------------------------------------------
    print("\n[3/4] 绘制 Ramachandran 图...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    h = axes[0].hist2d(phi_deg, psi_deg, bins=60, cmap='Blues', density=True)
    axes[0].set_xlabel('φ (degrees)', fontsize=12)
    axes[0].set_ylabel('ψ (degrees)', fontsize=12)
    axes[0].set_title(f'Ramachandran Plot ({solvent_config["name"].upper()}, {sim_config["temperature"]}K)', fontsize=14)
    axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[0].axvline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[0].set_xlim([-180, 180])
    axes[0].set_ylim([-180, 180])

    # 标注构象区域
    from matplotlib.patches import Rectangle
    axes[0].add_patch(Rectangle((-110, 50), 60, 50, fill=False, edgecolor='red', linewidth=2, linestyle='--'))
    axes[0].text(-80, 75, 'C7eq', fontsize=10, color='red', weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[0].add_patch(Rectangle((50, -100), 50, 60, fill=False, edgecolor='blue', linewidth=2, linestyle='--'))
    axes[0].text(75, -70, 'C7ax', fontsize=10, color='blue', weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[0].add_patch(Rectangle((-90, -60), 60, 60, fill=False, edgecolor='orange', linewidth=2, linestyle='--'))
    axes[0].text(-60, -30, 'αR', fontsize=10, color='orange', weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[0].add_patch(Rectangle((-90, 120), 40, 40, fill=False, edgecolor='green', linewidth=2, linestyle='--'))
    axes[0].text(-70, 140, 'PII', fontsize=10, color='green', weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.colorbar(h[3], ax=axes[0], label='Probability Density')

    axes[1].hist(phi_deg, bins=60, alpha=0.6, label='φ', density=True)
    axes[1].hist(psi_deg, bins=60, alpha=0.6, label='ψ', density=True)
    axes[1].set_xlabel('Angle (degrees)', fontsize=12)
    axes[1].set_ylabel('Probability Density', fontsize=12)
    axes[1].set_title('Torsion Angle Distributions', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    ramachandran_path = results_dir / "ramachandran.png"
    plt.savefig(ramachandran_path, dpi=300)
    print(f"  [OK] 保存: {ramachandran_path}")
    plt.close()

    # -------------------------------------------------------------------------
    # 4. 构象分析
    # -------------------------------------------------------------------------
    print("\n[4/4] 构象分析...")

    labels = classify_conformations(phi_deg, psi_deg)
    populations = compute_conformer_populations(labels)

    print("\n  构象占比:")
    print("  " + "-" * 40)
    for conf, frac in sorted(populations.items(), key=lambda x: -x[1]):
        if frac > 0.001:
            print(f"    {conf:8s}: {100*frac:6.2f}%")
    print("  " + "-" * 40)

    # 转换次数
    n_transitions = np.sum(labels[:-1] != labels[1:])
    dt_ps = sim_config['timestep'] * sim_config['trajectory_interval'] / 1000  # ps
    total_time_ns = len(labels) * dt_ps / 1000.0
    transition_rate = n_transitions / total_time_ns if total_time_ns > 0 else 0

    print(f"\n  构象转换:")
    print(f"    - 总转换次数: {n_transitions}")
    print(f"    - 模拟时间: {total_time_ns:.1f} ns")
    print(f"    - 转换频率: {transition_rate:.1f} 次/ns")

    # 保存 CSV
    populations_df = pd.DataFrame([
        {'conformer': k, 'fraction': v, 'percentage': f"{100*v:.2f}%"}
        for k, v in sorted(populations.items(), key=lambda x: -x[1])
    ])
    populations_csv_path = results_dir / "conformer_populations.csv"
    populations_df.to_csv(populations_csv_path, index=False)
    print(f"\n  [OK] 保存: {populations_csv_path}")

    # 构象时间序列图
    fig, ax = plt.subplots(figsize=(14, 4))

    conf_to_num = {'C7eq': 1, 'C7ax': 2, 'C5': 3, 'αR': 4, 'αL': 5, 'β': 6, 'PII': 7, 'other': 0}
    state_code = np.array([conf_to_num.get(l, 0) for l in labels])

    time_ns = np.arange(len(labels)) * dt_ps / 1000.0

    scatter = ax.scatter(time_ns, state_code, c=state_code, cmap='tab10', s=5, alpha=0.5)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels(['other', 'C7eq', 'C7ax', 'C5', 'αR', 'αL', 'β', 'PII'])
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Conformer', fontsize=12)
    ax.set_title(f'Conformer Evolution ({solvent_config["name"].upper()}, {sim_config["temperature"]}K)', fontsize=14)
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    timeline_path = results_dir / "conformer_timeline.png"
    plt.savefig(timeline_path, dpi=300)
    print(f"  [OK] 保存: {timeline_path}")
    plt.close()

    # -------------------------------------------------------------------------
    # 总结
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"分析完成: {solvent_config['name'].upper()}")
    print("=" * 60)

    print("\n输出文件:")
    print(f"  - {results_dir / 'energy_temperature.png'}")
    print(f"  - {results_dir / 'ramachandran.png'}")
    print(f"  - {results_dir / 'conformer_populations.csv'}")
    print(f"  - {results_dir / 'conformer_timeline.png'}")

    # 溶剂模型特点说明
    if solvent_config['name'] == 'vacuum':
        print("\n气相采样特点:")
        print(f"  - C7eq 占主导: 分子内氢键在气相中稳定")
    else:
        print("\n隐式溶剂采样特点:")
        print(f"  - PII/β 可能增加: 水的介电效应削弱分子内氢键")
        print(f"  - 构象转换可能更频繁")


if __name__ == "__main__":
    main()
