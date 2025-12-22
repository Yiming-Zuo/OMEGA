#!/usr/bin/env python
"""
步骤 3 改进版: 分析优化后的 REST2 HREMD 结果

修复:
1. 正确处理相邻态交换矩阵（不是全对）
2. 修复 u_kn 数据解析（支持1D和2D）
3. 增强构象分析（计算自由能）
4. 添加副本游走分析
"""

import sys
import pathlib

import pyarrow
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import mdtraj as md
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False
    print("[WARN] mdtraj 未安装，将跳过轨迹分析")

print("="*60)
print("Step 3 改进版: REST2 HREMD 结果分析")
print("="*60)

# =====================================================================
# 1. 加载采样数据
# =====================================================================
print("\n[1/8] 加载 HREMD 采样数据...")

samples_file = pathlib.Path('outputs_v2_gpu/samples.arrow')
if not samples_file.exists():
    print(f"[WARN] 未找到 {samples_file}，尝试使用旧版本...")
    samples_file = pathlib.Path('outputs/samples.arrow')
    if not samples_file.exists():
        print(f"[FAIL] 错误: 未找到采样文件")
        sys.exit(1)

with pyarrow.OSFile(str(samples_file), 'rb') as file:
    reader = pyarrow.RecordBatchStreamReader(file)
    table = reader.read_all()
    df = table.to_pandas()

print(f"✓ 加载完成: {len(df)} 个采样循环")
print(f"  - 数据列: {list(df.columns)}")

# =====================================================================
# 2. 分析交换接受率（修复版）
# =====================================================================
print("\n[2/8] 分析交换接受率...")

n_replicas = None
acceptance_rates = None

if 'n_proposed_swaps' in df.columns and 'n_accepted_swaps' in df.columns:
    n_prop_raw = df['n_proposed_swaps'].iloc[-1]
    n_acc_raw = df['n_accepted_swaps'].iloc[-1]

    # 检测数据格式
    if isinstance(n_prop_raw, np.ndarray) and len(n_prop_raw) > 0:
        first_elem = n_prop_raw[0]

        # 全对交换矩阵格式
        if isinstance(first_elem, np.ndarray):
            print("  检测到全对交换矩阵格式")
            prop_matrix = np.array([np.array(row) for row in n_prop_raw])
            acc_matrix = np.array([np.array(row) for row in n_acc_raw])
            n_replicas = len(prop_matrix)

            print(f"\n✓ 副本数: {n_replicas}")
            print(f"\n  所有交换对的接受率:")

            acceptance_rates_list = []
            for i in range(n_replicas):
                for j in range(i+1, n_replicas):
                    if prop_matrix[i, j] > 0:
                        rate = acc_matrix[i, j] / prop_matrix[i, j]
                        status = "[OK]" if 0.20 <= rate <= 0.40 else "[WARN]"
                        print(f"    State {i} ↔ {j}: {rate*100:.1f}% (提议={int(prop_matrix[i, j])}, 接受={int(acc_matrix[i, j])}) {status}")
                        if j == i + 1:  # 相邻态
                            acceptance_rates_list.append(rate)

            acceptance_rates = np.array(acceptance_rates_list)
            total_proposed = int(prop_matrix[np.triu_indices(n_replicas, k=1)].sum())
            total_accepted = int(acc_matrix[np.triu_indices(n_replicas, k=1)].sum())

        # 相邻态数组格式
        else:
            print("  检测到相邻态交换数组格式")
            n_prop = np.array(n_prop_raw)
            n_acc = np.array(n_acc_raw)
            n_replicas = len(n_prop) + 1

            acceptance_rates = np.divide(
                n_acc, n_prop,
                out=np.zeros_like(n_acc, dtype=float),
                where=n_prop > 0
            )

            total_proposed = int(n_prop.sum())
            total_accepted = int(n_acc.sum())

            print(f"\n✓ 副本数: {n_replicas}")
            print(f"  - n_proposed = {n_prop}")
            print(f"  - n_accepted = {n_acc}")

        # 统计
        if total_proposed > 0:
            global_rate = 100.0 * total_accepted / total_proposed
            print(f"\n✓ 交换统计:")
            print(f"  - 总提议: {total_proposed}")
            print(f"  - 总接受: {total_accepted}")
            print(f"  - 全局接受率: {global_rate:.2f}%")

            if acceptance_rates is not None and len(acceptance_rates) > 0:
                avg_rate = np.mean(acceptance_rates)
                print(f"  - 相邻态平均接受率: {100*avg_rate:.2f}%")

                print(f"\n  相邻态接受率:")
                for i, rate in enumerate(acceptance_rates):
                    status = "[OK]" if 0.20 <= rate <= 0.40 else "[WARN]"
                    print(f"    State {i} ↔ {i+1}: {100.0 * rate:.2f}% {status}")

    # 绘图
    if acceptance_rates is not None and len(acceptance_rates) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        colors = ['green' if 0.20 <= r <= 0.40 else 'orange' for r in acceptance_rates]
        ax.bar(range(len(acceptance_rates)), acceptance_rates, color=colors)
        ax.axhline(0.20, color='red', linestyle='--', alpha=0.5, label='Ideal range')
        ax.axhline(0.40, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Neighbor State Acceptance Rates')
        ax.set_xlabel('State Pair (i, i+1)')
        ax.set_ylabel('Acceptance Rate')
        ax.set_ylim([0, 1.0])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('acceptance_rates_v2.png', dpi=300)
        print(f"\n[OK] 保存: acceptance_rates_v2.png")

# =====================================================================
# 3. 分析副本游走（Replica Random Walk）
# =====================================================================
print("\n[3/8] 分析副本游走...")

if 'replica_to_state_idx' in df.columns and n_replicas is not None:
    replica_indices = np.array([np.array(x) for x in df['replica_to_state_idx']])

    print(f"✓ 副本游走数据: {replica_indices.shape}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 副本轨迹
    for i in range(min(n_replicas, 10)):  # 最多显示10个副本
        axes[0].plot(replica_indices[:, i], alpha=0.6, linewidth=0.8, label=f'Replica {i}')
    axes[0].set_xlabel('Cycle')
    axes[0].set_ylabel('State Index')
    axes[0].set_title('Replica Random Walk (副本在温度态间的游走)')
    axes[0].legend(ncol=5, fontsize=8)
    axes[0].grid(alpha=0.3)

    # 计算每个副本访问的状态范围
    state_coverage = []
    for i in range(n_replicas):
        unique_states = len(np.unique(replica_indices[:, i]))
        state_coverage.append(unique_states)

    axes[1].bar(range(n_replicas), state_coverage)
    axes[1].axhline(n_replicas, color='green', linestyle='--', label=f'Full coverage ({n_replicas} states)')
    axes[1].set_xlabel('Replica Index')
    axes[1].set_ylabel('Number of Unique States Visited')
    axes[1].set_title('State Coverage per Replica (理想情况所有副本都访问所有状态)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('replica_walk_v2.png', dpi=300)
    print(f"[OK] 保存: replica_walk_v2.png")

    print(f"\n  状态覆盖度:")
    for i in range(n_replicas):
        coverage_pct = 100.0 * state_coverage[i] / n_replicas
        status = "[OK]" if coverage_pct > 80 else "[WARN]"
        print(f"    Replica {i}: {state_coverage[i]}/{n_replicas} states ({coverage_pct:.1f}%) {status}")

# =====================================================================
# 4. 分析能量（修复版）
# =====================================================================
print("\n[4/8] 分析能量收敛性...")

energies = None

if 'u_kn' in df.columns:
    u_kn_series = df['u_kn'].dropna()

    if not u_kn_series.empty:
        # 尝试解析第一个样本
        first_sample = u_kn_series.iloc[0]
        first_array = np.asarray(first_sample)

        print(f"  u_kn 第一个样本形状: {first_array.shape}, 维度: {first_array.ndim}")
        print(f"  u_kn 第一个样本 dtype: {first_array.dtype}")
        print(f"  u_kn 第一个元素类型: {type(first_array.flat[0])}")

        # 检查是否是嵌套数组
        first_elem = first_array.flat[0]
        if isinstance(first_elem, np.ndarray):
            print(f"  检测到嵌套数组！第一个元素本身是数组，形状: {first_elem.shape}")
            # 直接使用第一个嵌套元素作为能量值
            energy_list = []
            for x in u_kn_series:
                arr = np.asarray(x, dtype=object)
                # 取第一个嵌套的数组
                if arr.size > 0 and isinstance(arr.flat[0], np.ndarray):
                    energy_list.append(arr.flat[0].astype(float))
            energies = np.array(energy_list, dtype=float)
            print(f"✓ 能量数据（嵌套格式）: {energies.shape}")

        # 如果是1D标量数组，直接堆叠
        elif first_array.ndim == 1:
            energy_list = []
            for x in u_kn_series:
                arr = np.asarray(x).astype(float)
                energy_list.append(arr)
            energies = np.array(energy_list, dtype=float)
            print(f"✓ 能量数据（1D格式）: {energies.shape}")

        # 如果是2D数组，提取对角线
        elif first_array.ndim == 2:
            energy_list = []
            for x in u_kn_series:
                arr = np.asarray(x)
                diag_vals = np.diag(arr).astype(float)
                energy_list.append(diag_vals)
            energies = np.array(energy_list, dtype=float)
            print(f"✓ 能量数据（2D格式）: {energies.shape}")

if energies is not None and energies.size > 0:
    print(f"\n  能量统计:")
    for i in range(min(energies.shape[1], 10)):
        mean_e = energies[:, i].mean()
        std_e = energies[:, i].std()
        print(f"    State {i}: 平均 = {mean_e:.2f} kT, 标准差 = {std_e:.2f} kT")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for i in range(energies.shape[1]):
        axes[0].plot(energies[:, i], alpha=0.6, label=f'State {i}', linewidth=1)
    axes[0].set_xlabel('Cycle')
    axes[0].set_ylabel('Reduced Potential (kT)')
    axes[0].set_title('Energy Time Series')
    axes[0].legend(ncol=energies.shape[1], fontsize=8)
    axes[0].grid(alpha=0.3)

    # 移动平均
    window = max(1, min(50, len(energies) // 10))
    if window > 1:
        kernel = np.ones(window) / window
        for i in range(energies.shape[1]):
            moving_avg = np.convolve(energies[:, i], kernel, mode='valid')
            axes[1].plot(moving_avg, label=f'State {i}', linewidth=1.5)
        axes[1].set_xlabel('Cycle')
        axes[1].set_ylabel(f'Reduced Potential (kT, {window}-cycle MA)')
        axes[1].set_title('Energy Convergence Check')
        axes[1].legend(ncol=energies.shape[1], fontsize=8)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('energy_convergence_v2.png', dpi=300)
    print(f"\n[OK] 保存: energy_convergence_v2.png")

# =====================================================================
# 5. 分析扭转角（增强版）
# =====================================================================
print("\n[5/8] 分析扭转角分布...")

phi_deg = None
psi_deg = None
c7eq_frac = 0
c7ax_frac = 0
transitions_all = 0

if HAS_MDTRAJ:
    traj_path = pathlib.Path('outputs_v2_gpu/trajectories/r0.dcd')
    if not traj_path.exists():
        traj_path = pathlib.Path('outputs/trajectories/r0.dcd')

    if traj_path.exists():
        print(f"✓ 加载轨迹: {traj_path}")
        traj = md.load(str(traj_path), top='system.pdb')
        print(f"  - 总帧数: {len(traj)}")

        phi_indices, phi = md.compute_phi(traj)
        psi_indices, psi = md.compute_psi(traj)

        # 验证选中了正确的残基
        print(f"  - φ 索引 (原子编号): {phi_indices}")
        print(f"  - ψ 索引 (原子编号): {psi_indices}")

        phi_deg = np.rad2deg(phi[:, 0])
        psi_deg = np.rad2deg(psi[:, 0])

        print(f"  - φ 范围: [{phi_deg.min():.1f}°, {phi_deg.max():.1f}°]")
        print(f"  - ψ 范围: [{psi_deg.min():.1f}°, {psi_deg.max():.1f}°]")

        # Ramachandran 图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        h = axes[0].hist2d(phi_deg, psi_deg, bins=50, cmap='Blues', density=True)
        axes[0].set_xlabel('φ (degrees)', fontsize=12)
        axes[0].set_ylabel('ψ (degrees)', fontsize=12)
        axes[0].set_title('Ramachandran Plot (State 0, 300K)', fontsize=14)
        axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')
        axes[0].axvline(0, color='gray', linewidth=0.5, linestyle='--')
        axes[0].set_xlim([-180, 180])
        axes[0].set_ylim([-180, 180])

        # 标注主要构象区域（带半透明背景框）
        from matplotlib.patches import Rectangle

        axes[0].text(-80, 80, 'C7eq', fontsize=11, color='darkred', weight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='darkred', linewidth=1.5))
        axes[0].text(75, -70, 'C7ax', fontsize=11, color='darkblue', weight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='darkblue', linewidth=1.5))
        axes[0].text(-70, 140, 'PII', fontsize=11, color='darkgreen', weight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='darkgreen', linewidth=1.5))
        axes[0].text(-55, -40, 'αR', fontsize=11, color='darkorange', weight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='darkorange', linewidth=1.5))
        axes[0].text(-150, 150, 'β', fontsize=11, color='purple', weight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='purple', linewidth=1.5))

        # 用虚线矩形框标出定义的边界（便于检查掩码是否合理）
        axes[0].add_patch(Rectangle((-110, 60), 60, 40, fill=False, edgecolor='darkred', linewidth=1.5, linestyle='--', alpha=0.7))
        axes[0].add_patch(Rectangle((50, -100), 50, 60, fill=False, edgecolor='darkblue', linewidth=1.5, linestyle='--', alpha=0.7))
        axes[0].add_patch(Rectangle((-90, 120), 40, 40, fill=False, edgecolor='darkgreen', linewidth=1.5, linestyle='--', alpha=0.7))
        axes[0].add_patch(Rectangle((-70, -60), 30, 40, fill=False, edgecolor='darkorange', linewidth=1.5, linestyle='--', alpha=0.7))
        axes[0].add_patch(Rectangle((-180, 120), 60, 60, fill=False, edgecolor='purple', linewidth=1.5, linestyle='--', alpha=0.7))

        plt.colorbar(h[3], ax=axes[0], label='Probability Density')

        axes[1].hist(phi_deg, bins=50, alpha=0.6, label='φ', density=True)
        axes[1].hist(psi_deg, bins=50, alpha=0.6, label='ψ', density=True)
        axes[1].set_xlabel('Angle (degrees)', fontsize=12)
        axes[1].set_ylabel('Probability Density', fontsize=12)
        axes[1].set_title('Torsion Angle Distributions', fontsize=14)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('ramachandran_v2.png', dpi=300)
        print(f"\n[OK] 保存: ramachandran_v2.png")

        # ========== 构象区域定义（基于文献 + 实际热图微调）==========
        # C7eq (七元环氢键，equatorial)
        c7eq_mask = (phi_deg > -110) & (phi_deg < -50) & (psi_deg > 60) & (psi_deg < 100)

        # C7ax (七元环氢键，axial)
        c7ax_mask = (phi_deg > 50) & (phi_deg < 100) & (psi_deg > -100) & (psi_deg < -40)

        # PII (polyproline II，水中的主要构象之一)
        pii_mask = (phi_deg > -90) & (phi_deg < -50) & (psi_deg > 120) & (psi_deg < 160)

        # αR (右手螺旋区)
        alphaR_mask = (phi_deg > -70) & (phi_deg < -40) & (psi_deg > -60) & (psi_deg < -20)

        # β-sheet (延展构象)
        beta_mask = (phi_deg > -180) & (phi_deg < -120) & (psi_deg > 120) & (psi_deg < 180)

        # αL (左手螺旋区，少见但存在)
        alphaL_mask = (phi_deg > 40) & (phi_deg < 80) & (psi_deg > 20) & (psi_deg < 60)

        # 计算占比（避免重叠计数，按优先级）
        c7eq_frac = c7eq_mask.sum() / len(phi_deg)
        c7ax_frac = c7ax_mask.sum() / len(phi_deg)
        pii_frac = (pii_mask & ~c7eq_mask & ~c7ax_mask).sum() / len(phi_deg)  # 排除重叠
        alphaR_frac = (alphaR_mask & ~c7eq_mask & ~c7ax_mask & ~pii_mask).sum() / len(phi_deg)
        beta_frac = (beta_mask & ~c7eq_mask & ~c7ax_mask & ~pii_mask & ~alphaR_mask).sum() / len(phi_deg)
        alphaL_frac = (alphaL_mask & ~c7eq_mask & ~c7ax_mask & ~pii_mask & ~alphaR_mask & ~beta_mask).sum() / len(phi_deg)
        other_frac = (~(c7eq_mask | c7ax_mask | pii_mask | alphaR_mask | beta_mask | alphaL_mask)).sum() / len(phi_deg)

        print(f"\n  ══════════════════════════════════════")
        print(f"  构象占比统计 (300K)")
        print(f"  ══════════════════════════════════════")
        print(f"    C7eq (七元环 equatorial): {100.0 * c7eq_frac:6.1f}%")
        print(f"    C7ax (七元环 axial):      {100.0 * c7ax_frac:6.1f}%")
        print(f"    PII  (polyproline II):    {100.0 * pii_frac:6.1f}%")
        print(f"    αR   (右手螺旋):          {100.0 * alphaR_frac:6.1f}%")
        print(f"    β    (延展构象):          {100.0 * beta_frac:6.1f}%")
        print(f"    αL   (左手螺旋):          {100.0 * alphaL_frac:6.1f}%")
        print(f"    其他 (过渡/浅谷):         {100.0 * other_frac:6.1f}%")
        print(f"  ──────────────────────────────────────")
        total_pct = c7eq_frac + c7ax_frac + pii_frac + alphaR_frac + beta_frac + alphaL_frac + other_frac
        print(f"    总计:                     {100.0 * total_pct:.1f}%")

        # ========== 构象转换分析 ==========
        # 定义状态标签（优先级顺序）
        state_labels = np.full(len(phi_deg), 'other', dtype=object)
        state_labels[c7eq_mask] = 'C7eq'
        state_labels[c7ax_mask] = 'C7ax'
        state_labels[pii_mask & ~(c7eq_mask | c7ax_mask)] = 'PII'
        state_labels[alphaR_mask & ~(c7eq_mask | c7ax_mask | pii_mask)] = 'αR'
        state_labels[beta_mask & ~(c7eq_mask | c7ax_mask | pii_mask | alphaR_mask)] = 'β'
        state_labels[alphaL_mask & ~(c7eq_mask | c7ax_mask | pii_mask | alphaR_mask | beta_mask)] = 'αL'

        # 检测所有状态转换
        transitions_all = np.sum(state_labels[:-1] != state_labels[1:])

        # 检测特定转换（C7eq ↔ C7ax）
        c7_transitions = np.sum((state_labels[:-1] == 'C7eq') & (state_labels[1:] == 'C7ax')) + \
                         np.sum((state_labels[:-1] == 'C7ax') & (state_labels[1:] == 'C7eq'))

        # 获取真实时间间隔
        if hasattr(traj, 'timestep') and traj.timestep:
            dt_ps = traj.timestep
            print(f"\n  ✓ 轨迹时间间隔: {dt_ps} ps/帧")
        else:
            dt_ps = 20.0  # 默认假设
            print(f"\n  [WARN] 无法读取时间间隔，假设为 {dt_ps} ps/帧")

        total_time_ns = len(phi_deg) * dt_ps / 1000.0

        print(f"\n  ══════════════════════════════════════")
        print(f"  构象转换统计")
        print(f"  ══════════════════════════════════════")
        print(f"    总转换次数:           {transitions_all} 次")
        print(f"    C7eq ↔ C7ax 转换:     {c7_transitions} 次")
        print(f"    转换频率 (所有):      {transitions_all / total_time_ns:.2f} 次/ns")
        print(f"    转换频率 (C7↔C7):     {c7_transitions / total_time_ns:.2f} 次/ns")
        print(f"    平均停留时间:         {total_time_ns / (transitions_all + 1) * 1000:.1f} ps/态")

        # 估算自由能差
        if c7eq_frac > 0 and c7ax_frac > 0:
            kT = 0.593  # kcal/mol @ 300K
            dG = -kT * np.log(c7ax_frac / c7eq_frac)
            print(f"\n  自由能估算 (粗略):")
            print(f"    - ΔG(C7ax - C7eq) ≈ {dG:.2f} kcal/mol")
            print(f"    - 文献参考值: ~0.6-1.2 kcal/mol")

        # ========== 构象演化时间序列 ==========
        fig2, ax2 = plt.subplots(figsize=(14, 4))

        # 为每个构象分配数字编码（用于着色）
        state_code = np.zeros(len(phi_deg))
        state_code[state_labels == 'C7eq'] = 1
        state_code[state_labels == 'C7ax'] = 2
        state_code[state_labels == 'PII'] = 3
        state_code[state_labels == 'αR'] = 4
        state_code[state_labels == 'β'] = 5
        state_code[state_labels == 'αL'] = 6

        time_ns = np.arange(len(phi_deg)) * dt_ps / 1000.0

        scatter = ax2.scatter(time_ns, state_code, c=state_code, cmap='tab10', s=10, alpha=0.6)
        ax2.set_yticks([0, 1, 2, 3, 4, 5, 6])
        ax2.set_yticklabels(['其他', 'C7eq', 'C7ax', 'PII', 'αR', 'β', 'αL'])
        ax2.set_xlabel('时间 (ns)', fontsize=12)
        ax2.set_ylabel('构象状态', fontsize=12)
        ax2.set_title('构象演化时间序列 (300K)', fontsize=14)
        ax2.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('conformation_timeline_v2.png', dpi=300)
        print(f"\n[OK] 保存: conformation_timeline_v2.png")

        # ========== 构象转移矩阵（粗粒度 Markov 分析）==========
        from collections import defaultdict

        states = ['C7eq', 'C7ax', 'PII', 'αR', 'β', 'αL', 'other']
        transition_counts = defaultdict(lambda: defaultdict(int))

        for i in range(len(state_labels) - 1):
            from_state = state_labels[i]
            to_state = state_labels[i + 1]
            transition_counts[from_state][to_state] += 1

        # 转为概率
        transition_matrix = np.zeros((len(states), len(states)))
        for i, from_state in enumerate(states):
            total = sum(transition_counts[from_state].values())
            if total > 0:
                for j, to_state in enumerate(states):
                    transition_matrix[i, j] = transition_counts[from_state][to_state] / total

        print(f"\n  ══════════════════════════════════════")
        print(f"  🔀 构象转移概率矩阵")
        print(f"  ══════════════════════════════════════")
        print(f"         → ", end="")
        for s in states:
            print(f"{s:>8}", end="")
        print()
        for i, from_state in enumerate(states):
            print(f"  {from_state:>6} → ", end="")
            for j in range(len(states)):
                print(f"{transition_matrix[i, j]:8.3f}", end="")
            print()

# =====================================================================
# 6. HREMD 混合统计分析 (新增)
# =====================================================================
print("\n[6/8] HREMD 混合统计分析...")

# 混合统计分析函数定义
def compute_transition_matrix_from_trajectories(replica_states):
    """
    从副本状态轨迹计算转移矩阵

    参数:
        replica_states: shape (n_cycles, n_replicas)
                       replica_states[t, r] = 副本r在时刻t所处的状态索引

    返回:
        transition_matrix: shape (n_states, n_states)
                          T[i,j] = 从状态i转移到状态j的概率
        transition_counts: 原始转移次数矩阵
    """
    n_cycles, n_replicas = replica_states.shape
    n_states = n_replicas  # 状态数等于副本数

    # 统计转移次数
    transition_counts = np.zeros((n_states, n_states), dtype=int)

    # 遍历每个副本的轨迹
    for r in range(n_replicas):
        for t in range(n_cycles - 1):
            state_from = replica_states[t, r]
            state_to = replica_states[t + 1, r]
            transition_counts[state_from, state_to] += 1

    # 归一化为概率矩阵
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        transition_counts, row_sums,
        out=np.zeros_like(transition_counts, dtype=float),
        where=row_sums > 0
    )

    return transition_matrix, transition_counts


def compute_subdominant_eigenvalue(transition_matrix):
    """计算转移矩阵的次主导特征值和所有特征值"""
    # 确保矩阵是随机矩阵（每行和为1）
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    P = np.divide(
        transition_matrix, row_sums,
        out=np.zeros_like(transition_matrix),
        where=row_sums > 0
    )

    # 计算特征值（需要转置，因为我们要左特征向量）
    eigenvalues, _ = np.linalg.eig(P.T)

    # 按绝对值排序
    eigenvalues = np.sort(np.abs(eigenvalues.real))[::-1]

    # 第一特征值永远是1.0，返回第二个
    lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0

    return lambda_2, eigenvalues


def compute_mixing_time(lambda_2):
    """计算混合时间"""
    if lambda_2 >= 1.0 or lambda_2 <= 0:
        return float('inf')
    return -1.0 / np.log(lambda_2)


def count_roundtrips(replica_states, replica_idx=0):
    """
    计算指定副本完成的round-trip次数
    Round-trip: 从state 0 到 state n-1 再回到 state 0
    """
    n_states = replica_states.shape[1]
    states = replica_states[:, replica_idx]

    n_roundtrips = 0
    at_bottom = True
    reached_top = False

    for state in states:
        if at_bottom and state == n_states - 1:
            reached_top = True
            at_bottom = False
        elif reached_top and state == 0:
            n_roundtrips += 1
            at_bottom = True
            reached_top = False

    return n_roundtrips


# 执行混合统计分析
mixing_lambda_2 = None
mixing_time = None
mixing_transition_matrix = None

if 'replica_to_state_idx' in df.columns and n_replicas is not None:
    replica_states = np.array([np.array(x) for x in df['replica_to_state_idx']])
    n_cycles = replica_states.shape[0]

    print(f"  副本状态数据: {replica_states.shape}")

    # 6.1 计算转移矩阵
    mixing_transition_matrix, transition_counts = compute_transition_matrix_from_trajectories(replica_states)

    print(f"\n  ========================================")
    print(f"  6.1 转移状态矩阵分析")
    print(f"  ========================================")

    print(f"\n  转移次数矩阵:")
    print("         ", end="")
    for j in range(n_replicas):
        print(f"State{j:2d}", end="  ")
    print()
    for i in range(n_replicas):
        print(f"  State{i}", end=" ")
        for j in range(n_replicas):
            print(f"{transition_counts[i,j]:7d}", end="  ")
        print()

    print(f"\n  转移概率矩阵:")
    print("         ", end="")
    for j in range(n_replicas):
        print(f"State{j:2d}", end="  ")
    print()
    for i in range(n_replicas):
        print(f"  State{i}", end=" ")
        for j in range(n_replicas):
            print(f"{mixing_transition_matrix[i,j]:7.4f}", end="  ")
        print()

    # 分析对角线元素
    diag_elements = np.diag(mixing_transition_matrix)
    diag_min, diag_max = diag_elements.min(), diag_elements.max()

    # 分析相邻态交换概率
    neighbor_probs = []
    for i in range(n_replicas - 1):
        neighbor_probs.append(mixing_transition_matrix[i, i+1])
    neighbor_min, neighbor_max = min(neighbor_probs), max(neighbor_probs)

    print(f"\n  评估结果:")
    diag_status = "[OK]" if diag_max < 0.7 else "[WARN]"
    print(f"    - 对角线元素范围: [{diag_min:.4f}, {diag_max:.4f}] {diag_status}")
    neighbor_status = "[OK]" if neighbor_min > 0.2 else "[WARN]"
    print(f"    - 相邻态交换概率范围: [{neighbor_min:.4f}, {neighbor_max:.4f}] {neighbor_status}")

    # 6.2 计算次主导特征值
    mixing_lambda_2, all_eigenvalues = compute_subdominant_eigenvalue(mixing_transition_matrix)
    mixing_time = compute_mixing_time(mixing_lambda_2)

    print(f"\n  ========================================")
    print(f"  6.2 次主导特征值分析")
    print(f"  ========================================")

    print(f"\n  特征值列表: {np.round(all_eigenvalues, 4).tolist()}")
    print(f"  次主导特征值 lambda_2 = {mixing_lambda_2:.4f}")
    print(f"  混合时间 tau_mix = {mixing_time:.2f} 次迭代")

    # 评估
    if mixing_lambda_2 < 0.5:
        lambda_status = "[OK] 优秀"
        mixing_quality = "充分混合"
    elif mixing_lambda_2 < 0.8:
        lambda_status = "[OK] 可接受"
        mixing_quality = "中等混合"
    elif mixing_lambda_2 < 0.95:
        lambda_status = "[WARN] 警告"
        mixing_quality = "混合较慢"
    else:
        lambda_status = "[FAIL] 失败"
        mixing_quality = "混合不良"

    print(f"\n  评估结果:")
    print(f"    - lambda_2 判定: {lambda_status}")
    print(f"    - 混合质量: {mixing_quality}")

    # 6.3 副本轨迹诊断
    print(f"\n  ========================================")
    print(f"  6.3 副本轨迹诊断")
    print(f"  ========================================")

    # 状态覆盖度
    state_coverage = []
    for r in range(n_replicas):
        unique_states = len(np.unique(replica_states[:, r]))
        state_coverage.append(unique_states)
        coverage_pct = 100.0 * unique_states / n_replicas
        status = "[OK]" if coverage_pct == 100 else "[WARN]"
        print(f"    Replica {r}: {unique_states}/{n_replicas} states ({coverage_pct:.1f}%) {status}")

    # Round-trip 统计
    roundtrips = []
    for r in range(n_replicas):
        rt = count_roundtrips(replica_states, r)
        roundtrips.append(rt)

    total_roundtrips = sum(roundtrips)
    avg_roundtrips = total_roundtrips / n_replicas

    print(f"\n  Round-trip 统计:")
    for r in range(n_replicas):
        print(f"    Replica {r}: {roundtrips[r]} 次")
    print(f"    总计: {total_roundtrips} 次, 平均: {avg_roundtrips:.1f} 次/副本")

    rt_status = "[OK]" if avg_roundtrips >= 3 else "[WARN]"
    print(f"    评估 (平均 >= 3): {rt_status}")

    # 6.4 生成可视化

    # 热图: 转移矩阵
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im = axes[0].imshow(mixing_transition_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xlabel('目标状态 j')
    axes[0].set_ylabel('源状态 i')
    axes[0].set_title('状态转移概率矩阵 P[i,j]')
    axes[0].set_xticks(range(n_replicas))
    axes[0].set_yticks(range(n_replicas))
    axes[0].set_xticklabels([f'S{i}' for i in range(n_replicas)])
    axes[0].set_yticklabels([f'S{i}' for i in range(n_replicas)])

    # 添加数值标注
    for i in range(n_replicas):
        for j in range(n_replicas):
            val = mixing_transition_matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            axes[0].text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)

    plt.colorbar(im, ax=axes[0], label='转移概率')

    # 特征值柱状图
    x_pos = range(len(all_eigenvalues))
    colors = ['green' if i == 0 else ('orange' if all_eigenvalues[i] > 0.8 else 'blue')
              for i in range(len(all_eigenvalues))]
    axes[1].bar(x_pos, all_eigenvalues, color=colors)
    axes[1].axhline(0.8, color='red', linestyle='--', alpha=0.7, label='lambda_2 < 0.8 理想阈值')
    axes[1].set_xlabel('特征值序号')
    axes[1].set_ylabel('特征值 |lambda|')
    axes[1].set_title(f'转移矩阵特征值分布 (lambda_2 = {mixing_lambda_2:.4f})')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'lambda_{i}' for i in range(len(all_eigenvalues))])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('transition_matrix_heatmap.png', dpi=300)
    print(f"\n  [OK] 保存: transition_matrix_heatmap.png")

    # 副本轨迹图 (增强版)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 绘制副本轨迹（采样显示以提高性能）
    sample_interval = max(1, n_cycles // 5000)  # 最多显示5000个点
    sampled_indices = np.arange(0, n_cycles, sample_interval)

    for r in range(n_replicas):
        axes[0].plot(sampled_indices, replica_states[sampled_indices, r],
                     alpha=0.7, linewidth=0.5, label=f'Replica {r}')

    axes[0].set_xlabel('迭代周期')
    axes[0].set_ylabel('状态索引')
    axes[0].set_title(f'副本状态轨迹 (lambda_2 = {mixing_lambda_2:.4f}, tau_mix = {mixing_time:.1f})')
    axes[0].legend(ncol=n_replicas, fontsize=8, loc='upper right')
    axes[0].set_yticks(range(n_replicas))
    axes[0].set_yticklabels([f'State {i}' for i in range(n_replicas)])
    axes[0].grid(alpha=0.3)

    # 状态覆盖和round-trip柱状图
    x = np.arange(n_replicas)
    width = 0.35

    bars1 = axes[1].bar(x - width/2, state_coverage, width, label='状态覆盖数', color='steelblue')
    bars2 = axes[1].bar(x + width/2, roundtrips, width, label='Round-trip次数', color='coral')

    axes[1].axhline(n_replicas, color='green', linestyle='--', alpha=0.7, label=f'完全覆盖 ({n_replicas}个状态)')
    axes[1].axhline(3, color='orange', linestyle=':', alpha=0.7, label='Round-trip判据 (>=3)')

    axes[1].set_xlabel('副本索引')
    axes[1].set_ylabel('计数')
    axes[1].set_title('副本采样统计')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Replica {i}' for i in range(n_replicas)])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('replica_state_trajectory.png', dpi=300)
    print(f"  [OK] 保存: replica_state_trajectory.png")

    # 生成文本报告
    report_lines = [
        "=" * 60,
        "HREMD 采样质量评估报告",
        "=" * 60,
        "",
        "一、转移状态矩阵分析",
        "-" * 40,
        f"  矩阵维度: {n_replicas} x {n_replicas}",
        f"  对角线元素范围: [{diag_min:.4f}, {diag_max:.4f}]",
        f"  相邻态交换概率范围: [{neighbor_min:.4f}, {neighbor_max:.4f}]",
        "",
        "  评估结果:",
        f"    - 对角线元素 < 0.7: {diag_status}",
        f"    - 相邻态交换 > 0.2: {neighbor_status}",
        "",
        "二、次主导特征值分析",
        "-" * 40,
        f"  特征值列表: {np.round(all_eigenvalues, 4).tolist()}",
        f"  次主导特征值 lambda_2 = {mixing_lambda_2:.4f}",
        f"  混合时间 tau_mix = {mixing_time:.2f} 次迭代",
        "",
        "  评估结果:",
        f"    - lambda_2 判定: {lambda_status}",
        f"    - 混合质量: {mixing_quality}",
        "",
        "三、副本轨迹诊断",
        "-" * 40,
        f"  副本数: {n_replicas}",
        f"  总迭代数: {n_cycles}",
        "",
        "  各副本状态覆盖度:",
    ]

    for r in range(n_replicas):
        coverage_pct = 100.0 * state_coverage[r] / n_replicas
        status = "[OK]" if coverage_pct == 100 else "[WARN]"
        report_lines.append(f"    Replica {r}: {state_coverage[r]}/{n_replicas} states ({coverage_pct:.1f}%) {status}")

    report_lines.extend([
        "",
        f"  Round-trip次数: 总计 {total_roundtrips}, 平均 {avg_roundtrips:.1f} 次/副本 {rt_status}",
        "",
        "=" * 60,
    ])

    report_text = "\n".join(report_lines)

    with open('mixing_statistics_report.txt', 'w') as f:
        f.write(report_text)
    print(f"  [OK] 保存: mixing_statistics_report.txt")

# =====================================================================
# 7. 生成总结报告
# =====================================================================
print("\n[7/8] 生成总结报告...")

print("\n" + "="*60)
print("REST2 HREMD 优化版测试总结")
print("="*60)

print(f"\n系统配置:")
if n_replicas:
    print(f"  - 副本数: {n_replicas}")
    print(f"  - 采样循环: {len(df)}")
    print(f"  - 估算采样时间: {len(df) * 1 / 1000:.1f} ns (假设1ps/cycle)")

print(f"\n交换效率:")
if acceptance_rates is not None and len(acceptance_rates) > 0:
    avg_rate = np.mean(acceptance_rates)
    if 0.20 <= avg_rate <= 0.40:
        print(f"  [OK] 平均相邻态接受率: {100*avg_rate:.1f}% (理想范围)")
    else:
        print(f"  [WARN] 平均相邻态接受率: {100*avg_rate:.1f}%")
        if avg_rate > 0.40:
            print(f"     → 建议: 增大温度间隔")
        else:
            print(f"     → 建议: 减小温度间隔")

print(f"\n构象采样 (State 0):")
if phi_deg is not None:
    print(f"  - C7eq: {100*c7eq_frac:.1f}%")
    print(f"  - C7ax: {100*c7ax_frac:.1f}%")
    print(f"  - 转换次数: {transitions_all}")

    if transitions_all < 5:
        print(f"  [WARN] 转换次数太少，需要更长采样时间")
    elif transitions_all > 20:
        print(f"  [OK] 转换次数充足，采样较为可靠")

print(f"\n输出文件:")
print(f"  - acceptance_rates_v2.png")
print(f"  - replica_walk_v2.png")
print(f"  - energy_convergence_v2.png")
if phi_deg is not None:
    print(f"  - ramachandran_v2.png")
    print(f"  - conformation_timeline_v2.png")
if mixing_transition_matrix is not None:
    print(f"  - transition_matrix_heatmap.png")
    print(f"  - replica_state_trajectory.png")
    print(f"  - mixing_statistics_report.txt")

print("="*60)
print("\n[OK] 分析完成！")
