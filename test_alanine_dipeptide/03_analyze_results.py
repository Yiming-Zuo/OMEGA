#!/usr/bin/env python
"""
步骤 3: 分析 REST2 HREMD 结果

任务:
1. 加载采样统计数据
2. 计算交换接受率
3. 绘制能量收敛曲线
4. 分析扭转角分布（φ/ψ Ramachandran 图）
5. 评估 REST2 的采样效率
"""

import sys

import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FEMTO_SRC = REPO_ROOT / "femto"
if FEMTO_SRC.exists() and str(FEMTO_SRC) not in sys.path:
    sys.path.insert(0, str(FEMTO_SRC))

import pyarrow
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 尝试导入 mdtraj（用于轨迹分析）
try:
    import mdtraj as md
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False
    print("⚠️ mdtraj 未安装，将跳过轨迹分析")
    print("   安装: conda activate fm && pip install mdtraj")

print("="*60)
print("Step 3: 分析 REST2 HREMD 结果")
print("="*60)

# =====================================================================
# 1. 加载采样数据
# =====================================================================
print("\n[1/5] 加载 HREMD 采样数据...")

samples_file = pathlib.Path('outputs/samples.arrow')
if not samples_file.exists():
    print(f"❌ 错误: 未找到 {samples_file}")
    print("   请先运行 python 02_run_rest2_hremd.py")
    sys.exit(1)

with pyarrow.OSFile(str(samples_file), 'rb') as file:
    reader = pyarrow.RecordBatchStreamReader(file)
    table = reader.read_all()
    df = table.to_pandas()

print(f"✓ 加载完成: {len(df)} 个采样循环")
print(f"  - 数据列: {list(df.columns)}")

if df.empty:
    print("❌ 错误: 采样文件为空，无法分析")
    sys.exit(1)

# =====================================================================
# 2. 计算交换接受率
# =====================================================================
print("\n[2/5] 分析交换接受率...")

def safe_flatten(obj):
    """递归展平嵌套的数组/列表，返回 Python 原生列表"""
    if isinstance(obj, (list, tuple)):
        flattened = []
        for item in obj:
            flattened.extend(safe_flatten(item))
        return flattened
    if isinstance(obj, np.ndarray):
        return safe_flatten(obj.tolist())
    return [obj]

def to_scalar(val):
    """将 numpy 标量/数组安全转换为 Python 标量"""
    if isinstance(val, np.ndarray):
        if val.size > 1:
            val = val.flatten()[0]
        elif val.size == 1:
            val = val.item()
        else:
            return 0.0
    if hasattr(val, 'item') and callable(getattr(val, 'item', None)):
        try:
            return val.item()
        except (ValueError, AttributeError):
            pass
    return float(val) if not isinstance(val, (int, float)) else val

acceptance_rates = None
total_proposed = None
total_accepted = None
n_states = None

required_swap_cols = {'n_proposed_swaps', 'n_accepted_swaps'}
if not required_swap_cols.issubset(df.columns):
    print("⚠️ 采样数据缺少交换统计列，跳过交换率分析")
else:
    proposed_series = df['n_proposed_swaps'].dropna()
    accepted_series = df['n_accepted_swaps'].dropna()

    if proposed_series.empty or accepted_series.empty:
        print("⚠️ 交换统计列没有有效数据，跳过交换率分析")
    else:
        n_proposed_raw = proposed_series.iloc[-1]
        n_accepted_raw = accepted_series.iloc[-1]

        n_proposed_list = safe_flatten(n_proposed_raw)
        n_accepted_list = safe_flatten(n_accepted_raw)

        if not n_proposed_list or not n_accepted_list:
            print("⚠️ 交换统计格式异常，跳过交换率分析")
        else:
            n_pairs = min(len(n_proposed_list), len(n_accepted_list))

            if n_pairs == 0:
                print("⚠️ 未检测到交换配对，跳过交换率分析")
            else:
                n_proposed = np.asarray(n_proposed_list[:n_pairs], dtype=float)
                n_accepted = np.asarray(n_accepted_list[:n_pairs], dtype=float)

                acceptance_rates = np.divide(
                    n_accepted,
                    n_proposed,
                    out=np.zeros_like(n_accepted, dtype=float),
                    where=n_proposed > 0,
                )
                n_states = n_pairs + 1
                total_proposed = float(n_proposed.sum())
                total_accepted = float(n_accepted.sum())

                print(f"  - n_proposed = {n_proposed}")
                print(f"  - n_accepted = {n_accepted}")

                if total_proposed > 0:
                    global_rate = 100.0 * total_accepted / total_proposed
                    print(f"\n✓ 副本交换统计:")
                    print(f"  - 总交换提议: {int(total_proposed)}")
                    print(f"  - 总接受次数: {int(total_accepted)}")
                    print(f"  - 全局接受率: {global_rate:.2f}%")
                else:
                    print("⚠️ 交换提议总数为 0，无法计算全局接受率")

                if acceptance_rates.size:
                    print(f"\n  相邻态接受率:")
                    for i, rate in enumerate(acceptance_rates):
                        status = "✅" if 0.15 <= rate <= 0.35 else "⚠️"
                        print(f"    State {i} ↔ {i+1}: {100.0 * rate:.2f}% {status}")

if acceptance_rates is not None and acceptance_rates.size:
    acceptance_matrix = np.zeros((n_states, n_states))
    for i, rate in enumerate(acceptance_rates):
        acceptance_matrix[i, i + 1] = rate
        acceptance_matrix[i + 1, i] = rate

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(acceptance_matrix, cmap='RdYlGn', vmin=0, vmax=0.5)
    axes[0].set_title('REST2 Exchange Acceptance Matrix')
    axes[0].set_xlabel('State j')
    axes[0].set_ylabel('State i')
    plt.colorbar(im, ax=axes[0], label='Acceptance Rate')

    for i in range(n_states):
        for j in range(n_states):
            if acceptance_matrix[i, j] > 0.01:
                axes[0].text(
                    j,
                    i,
                    f'{acceptance_matrix[i, j]:.2f}',
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    diag_rates = [to_scalar(rate) for rate in acceptance_rates]
    colors = ['green' if 0.15 <= r <= 0.35 else 'orange' for r in diag_rates]
    axes[1].bar(range(len(diag_rates)), diag_rates, color=colors)
    axes[1].axhline(0.15, color='red', linestyle='--', alpha=0.5, label='理想下限')
    axes[1].axhline(0.35, color='red', linestyle='--', alpha=0.5, label='理想上限')
    axes[1].set_title('相邻态接受率')
    axes[1].set_xlabel('State Pair (i, i+1)')
    axes[1].set_ylabel('Acceptance Rate')
    axes[1].set_ylim([0, 0.6])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('acceptance_rates.png', dpi=300)
    print(f"\n✅ 保存: acceptance_rates.png")
else:
    print("⚠️ 跳过交换率图表生成")

# =====================================================================
# 3. 分析能量收敛性
# =====================================================================
print("\n[3/5] 分析能量收敛性...")

energies = None
u_kn_array = None

if 'u_kn' not in df.columns:
    print("⚠️ 采样数据缺少 u_kn 列，跳过能量分析")
else:
    u_kn_series = df['u_kn'].dropna()

    if u_kn_series.empty:
        print("⚠️ u_kn 列没有有效数据，跳过能量分析")
    else:
        u_kn_list = []
        invalid_shape = False

        for entry in u_kn_series:
            entry_array = np.asarray(entry)
            if entry_array.ndim != 2:
                invalid_shape = True
                break
            u_kn_list.append(entry_array)

        if invalid_shape or not u_kn_list:
            print("⚠️ u_kn 数据形状异常，跳过能量分析")
        else:
            try:
                u_kn_array = np.stack(u_kn_list, axis=0)
            except ValueError:
                u_kn_array = None
                print("⚠️ u_kn 数据不可堆叠，跳过能量分析")

if u_kn_array is not None:
    if u_kn_array.shape[1] != u_kn_array.shape[2]:
        print("⚠️ u_kn 数据不是方阵，跳过能量分析")
    else:
        if n_states is None:
            n_states = u_kn_array.shape[1]
        elif n_states != u_kn_array.shape[1]:
            print(f"⚠️ 交换统计副本数 {n_states} 与 u_kn 副本数 {u_kn_array.shape[1]} 不一致，以 u_kn 为准")
            n_states = u_kn_array.shape[1]

        energies = np.diagonal(u_kn_array, axis1=1, axis2=2)

if energies is not None:
    print(f"✓ 能量数据:")
    print(f"  - 采样循环: {energies.shape[0]}")
    print(f"  - 副本数: {energies.shape[1]}")

    for i in range(energies.shape[1]):
        mean_e = to_scalar(energies[:, i].mean())
        std_e = to_scalar(energies[:, i].std())
        print(f"  - State {i}: 平均 = {mean_e:.2f} kT, 标准差 = {std_e:.2f} kT")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for i in range(energies.shape[1]):
        axes[0].plot(energies[:, i], alpha=0.6, label=f'State {i}', linewidth=1)
    axes[0].set_xlabel('Cycle')
    axes[0].set_ylabel('Reduced Potential (kT)')
    axes[0].set_title('Energy Time Series by State')
    axes[0].legend(ncol=energies.shape[1], fontsize=8)
    axes[0].grid(alpha=0.3)

    window = max(1, min(50, energies.shape[0] // 5))
    if window > 1:
        kernel = np.ones(window) / window
        for i in range(energies.shape[1]):
            moving_avg = np.convolve(energies[:, i], kernel, mode='valid')
            axes[1].plot(moving_avg, label=f'State {i}', linewidth=1.5)
        axes[1].set_xlabel('Cycle')
        axes[1].set_ylabel(f'Reduced Potential (kT, {window}-cycle MA)')
        axes[1].set_title('能量移动平均（收敛性检查）')
        axes[1].legend(ncol=energies.shape[1], fontsize=8)
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(
            0.5,
            0.5,
            '数据点太少，无法计算移动平均',
            ha='center',
            va='center',
            transform=axes[1].transAxes,
        )

    plt.tight_layout()
    plt.savefig('energy_convergence.png', dpi=300)
    print(f"\n✅ 保存: energy_convergence.png")
else:
    print("⚠️ 跳过能量分析图表生成")

# =====================================================================
# 4. 分析扭转角分布（需要 mdtraj）
# =====================================================================
print("\n[4/5] 分析扭转角分布...")

c7eq_mask = None
transitions = 0
phi_deg = None
psi_deg = None

if HAS_MDTRAJ:
    # 只分析状态 0（目标温度 300K）
    traj_path = pathlib.Path('outputs/trajectories/r0.dcd')

    if traj_path.exists():
        print(f"✓ 加载轨迹: {traj_path}")
        traj = md.load(str(traj_path), top='system.pdb')
        print(f"  - 总帧数: {len(traj)}")
        print(f"  - 原子数: {traj.n_atoms}")

        # 计算 φ, ψ 角
        phi_indices, phi = md.compute_phi(traj)
        psi_indices, psi = md.compute_psi(traj)

        # 转换为度
        phi_deg = np.rad2deg(phi[:, 0])
        psi_deg = np.rad2deg(psi[:, 0])

        print(f"  - φ 范围: [{phi_deg.min():.1f}°, {phi_deg.max():.1f}°]")
        print(f"  - ψ 范围: [{psi_deg.min():.1f}°, {psi_deg.max():.1f}°]")

        # 绘制 Ramachandran 图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：2D 直方图
        h = axes[0].hist2d(phi_deg, psi_deg, bins=50, cmap='Blues', density=True)
        axes[0].set_xlabel('φ (degrees)', fontsize=12)
        axes[0].set_ylabel('ψ (degrees)', fontsize=12)
        axes[0].set_title('Ramachandran Plot (State 0, 300K)', fontsize=14)
        axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')
        axes[0].axvline(0, color='gray', linewidth=0.5, linestyle='--')
        axes[0].set_xlim([-180, 180])
        axes[0].set_ylim([-180, 180])
        plt.colorbar(h[3], ax=axes[0], label='Probability Density')

        # 右图：φ 和 ψ 的 1D 分布
        axes[1].hist(phi_deg, bins=50, alpha=0.6, label='φ', density=True, color='blue')
        axes[1].hist(psi_deg, bins=50, alpha=0.6, label='ψ', density=True, color='red')
        axes[1].set_xlabel('Angle (degrees)', fontsize=12)
        axes[1].set_ylabel('Probability Density', fontsize=12)
        axes[1].set_title('Torsion Angle Distributions', fontsize=14)
        axes[1].legend(fontsize=12)
        axes[1].grid(alpha=0.3)
        axes[1].set_xlim([-180, 180])

        plt.tight_layout()
        plt.savefig('ramachandran.png', dpi=300)
        print(f"\n✅ 保存: ramachandran.png")

        # 统计构象占比
        # C7eq (αR): φ ~ -80°, ψ ~ 80°
        # C7ax (αL): φ ~ 60°, ψ ~ -60°
        # α-helix: φ ~ -60°, ψ ~ -45°
        c7eq_mask = (phi_deg < -30) & (psi_deg > 30)
        c7ax_mask = (phi_deg > 30) & (psi_deg < -30)
        alpha_mask = (phi_deg > -100) & (phi_deg < -30) & (psi_deg > -70) & (psi_deg < -20)

        print(f"\n  构象占比:")
        print(f"    - C7eq (αR):  {100.0 * c7eq_mask.sum() / len(phi_deg):.1f}%")
        print(f"    - C7ax (αL):  {100.0 * c7ax_mask.sum() / len(psi_deg):.1f}%")
        print(f"    - α-helix:    {100.0 * alpha_mask.sum() / len(phi_deg):.1f}%")
        print(f"    - 其他:       {100.0 * (~c7eq_mask & ~c7ax_mask & ~alpha_mask).sum() / len(phi_deg):.1f}%")

        # 构象转换次数
        transitions = int(np.sum(np.abs(np.diff(c7eq_mask.astype(int)))) // 2)
        print(f"\n  构象转换:")
        print(f"    - C7eq ↔ 其他: {transitions} 次")
        print(f"    - 平均停留时间: {len(phi_deg) / (transitions + 1):.1f} 帧")

    else:
        print(f"⚠️ 未找到轨迹文件: {traj_path}")
        print("   跳过扭转角分析")
else:
    print("⚠️ mdtraj 未安装，跳过扭转角分析")

# =====================================================================
# 5. 生成总结报告
# =====================================================================
print("\n[5/5] 生成总结报告...")

print("\n" + "="*60)
print("REST2 HREMD 测试总结")
print("="*60)
print(f"\n系统信息:")
replica_info = n_states if n_states is not None else '未知'
print(f"  - 副本数: {replica_info}")
print(f"  - 采样循环: {len(df)}")
if isinstance(replica_info, int):
    print(f"  - 总采样点: {len(df) * replica_info}")
else:
    print("  - 总采样点: 未知（缺少副本数信息）")

print(f"\n交换统计:")
avg_neighbor_rate = None
if acceptance_rates is None or not acceptance_rates.size:
    print("  - 未生成交换统计（缺少或无效数据）")
else:
    if total_proposed and total_proposed > 0:
        global_rate = 100.0 * total_accepted / total_proposed
        print(f"  - 全局接受率: {global_rate:.2f}%")
    else:
        print("  - 全局接受率: 未知（总提议为 0）")
    avg_neighbor_rate = to_scalar(np.mean(acceptance_rates))
    print(f"  - 平均相邻态接受率: {100.0 * avg_neighbor_rate:.2f}%")

print(f"\n能量统计 (State 0):")
if energies is None or energies.size == 0 or energies.shape[1] == 0:
    print("  - 未生成能量统计（缺少或无效数据）")
else:
    state0_mean = to_scalar(energies[:, 0].mean())
    state0_std = to_scalar(energies[:, 0].std())
    print(f"  - 平均: {state0_mean:.2f} kT")
    print(f"  - 标准差: {state0_std:.2f} kT")

if HAS_MDTRAJ and phi_deg is not None:
    print(f"\n构象采样 (State 0):")
    print(f"  - C7eq 占比: {100.0 * c7eq_mask.sum() / len(phi_deg):.1f}%")
    print(f"  - 构象转换: {transitions} 次")

print(f"\n输出文件:")
print(f"  ✓ acceptance_rates.png")
print(f"  ✓ energy_convergence.png")
if HAS_MDTRAJ and phi_deg is not None:
    print(f"  ✓ ramachandran.png")

print(f"\n评估:")
if avg_neighbor_rate is not None:
    if 0.15 <= avg_neighbor_rate <= 0.35:
        print(f"  ✅ 接受率在理想范围内 (15-35%)")
    else:
        print(f"  ⚠️ 接受率不理想 (建议: 调整温度梯度)")
else:
    print(f"  ⚠️ 未能评估接受率（缺少交换统计）")

if energies is not None and energies.size > 0 and energies.shape[1] > 0:
    mean_energy = to_scalar(energies[:, 0].mean())
    std_energy = to_scalar(energies[:, 0].std())
    if mean_energy != 0 and std_energy / abs(mean_energy) < 0.1:
        print(f"  ✅ 能量收敛良好 (CV < 10%)")
    else:
        print(f"  ⚠️ 能量仍在涨落（可能需要更长采样）")
else:
    print(f"  ⚠️ 未能评估能量收敛性（缺少能量数据）")

print("="*60)
print("\n✅ 分析完成！")
print("\n请查看生成的图片文件以获取详细结果。")
