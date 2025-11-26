#!/usr/bin/env python
"""
步骤 4: REST2 HREMD 的 MBAR 重加权分析

完整修正版本（基于技术评审）:
1. ✅ 对每个replica的完整轨迹单独子采样（修正1）
2. ✅ 正确的MBAR权重索引 W_kn[0, :]（修正2）
3. ✅ N_k在子采样后统计（修正3）
4. ✅ 验证cycle→frame映射（修正4）
5. ✅ 完整的Phase 2诊断检查（overlap, ESS, 能量）

理论基础:
- REST2 = 哈密顿REMD（固定温度，不同λ）
- MBAR重加权恢复State 0（300K, λ=1）的物理分布
- 利用所有6个副本的增强采样数据

输出:
- ramachandran_mbar_comparison.png: MBAR vs Replica 0对比
- mbar_diagnostics.png: overlap/ESS/能量分布
- conformation_populations.png: 构象占比统计
- mbar_weights.npz: 权重数据（可复用）
- mbar_analysis_report.txt: 详细报告
"""

import sys
import pathlib
import numpy as np
import pyarrow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mdtraj as md

try:
    import pymbar
    print(f"✅ pymbar 版本: {pymbar.__version__}")
except ImportError:
    print("❌ 错误: pymbar 未安装")
    print("安装命令: conda install -c conda-forge pymbar")
    sys.exit(1)

print("="*70)
print(" REST2 HREMD 数据的 MBAR 重加权分析")
print("="*70)

# =========================================================================
# Part 1: 数据加载与验证
# =========================================================================
print("\n[Part 1/8] 数据加载与验证...")

# 配置参数
DATA_DIR = pathlib.Path('outputs_v2_gpu')
SAMPLES_FILE = DATA_DIR / 'samples.arrow'
TRAJ_DIR = DATA_DIR / 'trajectories'
TOPOLOGY_FILE = pathlib.Path('system.pdb')
TRAJECTORY_INTERVAL = 20  # 从femto配置：每20 cycles保存1帧

# 验证文件存在
if not SAMPLES_FILE.exists():
    print(f"❌ 错误: {SAMPLES_FILE} 不存在")
    sys.exit(1)

if not TOPOLOGY_FILE.exists():
    print(f"❌ 错误: {TOPOLOGY_FILE} 不存在")
    sys.exit(1)

print(f"✓ 数据目录: {DATA_DIR}")
print(f"✓ 拓扑文件: {TOPOLOGY_FILE}")

# 加载samples.arrow
with pyarrow.OSFile(str(SAMPLES_FILE), 'rb') as file:
    reader = pyarrow.RecordBatchStreamReader(file)
    table = reader.read_all()
    df = table.to_pandas()

n_cycles = len(df)
print(f"✓ 采样数据: {n_cycles} cycles")

# 提取关键数据
u_kn_raw = df['u_kn'].values
replica_to_state_raw = df['replica_to_state_idx'].values

# 转换为numpy数组
n_replicas = len(np.asarray(replica_to_state_raw[0]))
print(f"✓ 副本数: {n_replicas}")

# 构建完整的u_kn数组 [cycle, replica, state]
print("  正在重组u_kn数据...")
u_kn_array = np.zeros((n_cycles, n_replicas, n_replicas))
replica_to_state_array = np.zeros((n_cycles, n_replicas), dtype=int)

for cycle in range(n_cycles):
    # replica_to_state
    replica_to_state_array[cycle, :] = np.asarray(replica_to_state_raw[cycle])

    # u_kn矩阵
    u_kn_cycle = np.asarray(u_kn_raw[cycle])
    for replica in range(n_replicas):
        u_kn_array[cycle, replica, :] = np.asarray(u_kn_cycle[replica])

print(f"✓ u_kn数组: {u_kn_array.shape} (cycle, replica, state)")

# 验证轨迹文件
print("\n验证轨迹文件...")
traj_frames = []
for replica in range(n_replicas):
    traj_file = TRAJ_DIR / f'r{replica}.dcd'
    if not traj_file.exists():
        print(f"❌ 错误: {traj_file} 不存在")
        sys.exit(1)

    traj = md.load(str(traj_file), top=str(TOPOLOGY_FILE))
    traj_frames.append(len(traj))
    if replica == 0:
        print(f"  Replica 0: {len(traj)} 帧, {traj.n_atoms} 原子, timestep={traj.timestep} ps")

# 验证cycle→frame映射
expected_frames = n_cycles // TRAJECTORY_INTERVAL
if traj_frames[0] != expected_frames:
    print(f"⚠️ 警告: DCD帧数({traj_frames[0]}) ≠ 预期({expected_frames})")
    print(f"  cycles={n_cycles}, interval={TRAJECTORY_INTERVAL}")
    print(f"  将使用实际帧数: {traj_frames[0]}")
else:
    print(f"✅ Cycle→Frame映射验证通过: {n_cycles} cycles ÷ {TRAJECTORY_INTERVAL} = {expected_frames} frames")

# 能量单位检查
sample_energy = u_kn_array[0, 0, 0]
print(f"\n能量单位检查:")
print(f"  样本能量值: {sample_energy:.2f}")
if abs(sample_energy) > 1e6:
    print(f"  ⚠️ 数量级过大，可能是J/mol，需要转换为kT")
elif abs(sample_energy) > 1e3:
    print(f"  ✓ 数量级合理（可能是kJ/mol或约化能量）")
else:
    print(f"  ✓ 可能已是约化能量(kT)")

# =========================================================================
# Part 2: 逐Replica子采样（核心修正）
# =========================================================================
print("\n" + "="*70)
print("[Part 2/8] 逐Replica子采样（保持时间连续性）")
print("="*70)

subsampled_frames = []  # 存储所有子采样帧的元数据

print("\n对每个replica单独进行时间序列分析:")

for replica in range(n_replicas):
    print(f"\n--- Replica {replica} ---")

    # 提取该replica的完整能量时间序列
    # 使用该replica在其当前state下的能量（对角元素）
    replica_energies = []
    for cycle in range(n_cycles):
        state = replica_to_state_array[cycle, replica]
        u_self = u_kn_array[cycle, replica, state]
        replica_energies.append(u_self)

    replica_energies = np.array(replica_energies)

    # 平衡化检测
    try:
        t0, g, Neff_raw = pymbar.timeseries.detect_equilibration(replica_energies)
        print(f"  平衡化时间: cycle {t0} ({100*t0/n_cycles:.1f}%)")
        print(f"  统计不相关时间 g: {g:.2f}")
        print(f"  有效样本数 Neff: {Neff_raw:.0f} / {n_cycles - t0}")
    except Exception as e:
        print(f"  ⚠️ 平衡化检测失败: {e}")
        print(f"  使用默认: t0=0, g=50")
        t0 = 0
        g = 50.0
        Neff_raw = (n_cycles - t0) / g

    # 从t0开始子采样
    equilibrated = replica_energies[t0:]

    try:
        indices = pymbar.timeseries.subsample_correlated_data(equilibrated, g=g)
        print(f"  子采样: {len(indices)} 独立样本 (子采样率 1/{int(g)})")
    except Exception as e:
        print(f"  ⚠️ 子采样失败: {e}")
        # 使用简单的stride子采样
        stride = max(1, int(g))
        indices = np.arange(0, len(equilibrated), stride)
        print(f"  使用stride={stride}子采样: {len(indices)} 样本")

    # 保存子采样帧的完整信息
    for idx in indices:
        global_cycle = t0 + idx
        state = replica_to_state_array[global_cycle, replica]
        u_all_states = u_kn_array[global_cycle, replica, :]

        subsampled_frames.append({
            'cycle': global_cycle,
            'replica': replica,
            'state': state,          # 采样该帧时所处的state
            'u_kn': u_all_states     # 该构象在所有state下的能量
        })

print(f"\n✅ 子采样完成: 总共 {len(subsampled_frames)} 个独立样本")
print(f"  原始数据: {n_cycles * n_replicas} 帧")
print(f"  子采样后: {len(subsampled_frames)} 帧 (压缩率 {100*len(subsampled_frames)/(n_cycles*n_replicas):.1f}%)")

# =========================================================================
# Part 3: 构建MBAR输入
# =========================================================================
print("\n" + "="*70)
print("[Part 3/8] 构建MBAR输入矩阵")
print("="*70)

# 统计每个state的子采样样本数
N_k = np.zeros(n_replicas, dtype=int)
for frame in subsampled_frames:
    N_k[frame['state']] += 1

print("\n各State的子采样样本数:")
for k in range(n_replicas):
    print(f"  State {k}: {N_k[k]:4d} 样本 ({100*N_k[k]/len(subsampled_frames):5.1f}%)")

# 构建u_kn矩阵 [K, N]
N_total = len(subsampled_frames)
u_kn_mbar = np.zeros((n_replicas, N_total))

for n, frame in enumerate(subsampled_frames):
    u_kn_mbar[:, n] = frame['u_kn']

# 一致性检查
assert N_k.sum() == N_total, f"N_k.sum()={N_k.sum()} != N_total={N_total}"
assert u_kn_mbar.shape == (n_replicas, N_total), f"u_kn_mbar.shape={u_kn_mbar.shape}"

print(f"\n✅ MBAR输入矩阵构建完成:")
print(f"  u_kn shape: {u_kn_mbar.shape} (K={n_replicas} states, N={N_total} samples)")
print(f"  N_k: {N_k}")
print(f"  总样本数验证: {N_k.sum()} == {N_total} ✓")

# =========================================================================
# Part 4: MBAR计算
# =========================================================================
print("\n" + "="*70)
print("[Part 4/8] MBAR求解")
print("="*70)

print("\n初始化MBAR求解器...")
try:
    mbar = pymbar.MBAR(
        u_kn_mbar,
        N_k,
        verbose=True,
        maximum_iterations=10000,
        relative_tolerance=1e-7
    )
    print(f"\n✅ MBAR收敛成功!")
    if hasattr(mbar, 'iterations'):
        print(f"  迭代次数: {mbar.iterations}")
except Exception as e:
    print(f"\n❌ MBAR求解失败: {e}")
    print("\n可能原因:")
    print("  1. State间overlap太小")
    print("  2. 能量值包含NaN或Inf")
    print("  3. 样本数太少")
    sys.exit(1)

# 获取State 0的权重（pymbar 4.x API）
print("\n获取State 0权重...")
try:
    # pymbar 4.x: W_nk 是 [N_samples, K_states] 格式
    # W_nk[n, k] = 样本n在目标state k下的权重
    weights_state0 = mbar.W_nk[:, 0]  # ✅ 取第一列（State 0）
    print(f"✓ 权重矩阵: {mbar.W_nk.shape} [N_samples, K_states]")
    print(f"✓ State 0权重: {weights_state0.shape}")
    print(f"✓ 权重和: {weights_state0.sum():.6f} (应为1.0)")
except Exception as e:
    print(f"❌ 获取权重失败: {e}")
    sys.exit(1)

# =========================================================================
# Part 5: 诊断检查（Phase 2）
# =========================================================================
print("\n" + "="*70)
print("[Part 5/8] MBAR诊断检查")
print("="*70)

# -------------------------------------------------------------------------
# 检查 1: State Overlap矩阵
# -------------------------------------------------------------------------
print("\n【检查 1】State Overlap矩阵")
print("-" * 70)

try:
    overlap_result = mbar.compute_overlap()
    overlap_matrix = overlap_result['matrix']

    print("\nState Overlap Matrix:")
    print("       ", end="")
    for j in range(n_replicas):
        print(f"   S{j}  ", end="")
    print()

    for i in range(n_replicas):
        print(f"S{i}:  ", end="")
        for j in range(n_replicas):
            val = overlap_matrix[i, j]
            print(f" {val:6.3f}", end="")
        print()

    # 检查相邻state的overlap
    print("\n相邻State Overlap检查:")
    all_good = True
    for i in range(n_replicas - 1):
        overlap = overlap_matrix[i, i+1]
        if overlap > 0.05:
            status = "✅ 良好"
        elif overlap > 0.03:
            status = "⚠️ 偏低"
            all_good = False
        else:
            status = "❌ 太低"
            all_good = False
        print(f"  State {i} ↔ {i+1}: {overlap:.4f}  {status}")

    if all_good:
        print("\n✅ Overlap检查通过: 所有相邻state overlap > 0.05")
    else:
        print("\n⚠️ 部分state overlap偏低，MBAR结果需谨慎解释")

except Exception as e:
    print(f"⚠️ Overlap计算失败: {e}")
    overlap_matrix = None

# -------------------------------------------------------------------------
# 检查 2: 权重有效性（ESS）
# -------------------------------------------------------------------------
print("\n【检查 2】权重有效性分析")
print("-" * 70)

# 计算有效样本数
ESS = (weights_state0.sum())**2 / (weights_state0**2).sum()
efficiency = ESS / len(weights_state0)

print(f"\nState 0 权重统计:")
print(f"  总样本数: {len(weights_state0)}")
print(f"  有效样本数 (ESS): {ESS:.0f}")
print(f"  统计效率: {100*efficiency:.2f}%")

# 分析权重集中度
sorted_weights = np.sort(weights_state0)[::-1]
cumsum = np.cumsum(sorted_weights)
n_50 = np.searchsorted(cumsum, 0.5 * cumsum[-1]) + 1
n_90 = np.searchsorted(cumsum, 0.9 * cumsum[-1]) + 1

print(f"\n权重集中度:")
print(f"  前 {n_50} 个样本贡献 50% 权重 ({100*n_50/len(weights_state0):.2f}%)")
print(f"  前 {n_90} 个样本贡献 90% 权重 ({100*n_90/len(weights_state0):.2f}%)")

# 健康判断
if efficiency > 0.1:
    print(f"\n✅ 权重分布健康 (效率 > 10%)")
elif efficiency > 0.05:
    print(f"\n⚠️ 权重略集中 (5% < 效率 < 10%)，结果可用但需谨慎")
else:
    print(f"\n❌ 警告: 权重严重集中 (效率 < 5%)，MBAR结果可能不可靠")

# -------------------------------------------------------------------------
# 检查 3: 能量分布
# -------------------------------------------------------------------------
print("\n【检查 3】能量分布合理性")
print("-" * 70)

mean_energies = []
for k in range(n_replicas):
    mask = np.array([f['state'] == k for f in subsampled_frames])
    if mask.sum() > 0:
        energies_k = u_kn_mbar[k, mask]
        mean_e = energies_k.mean()
        std_e = energies_k.std()
        mean_energies.append(mean_e)
        print(f"  State {k}: 平均={mean_e:8.1f}, 标准差={std_e:7.1f}, 样本数={mask.sum()}")
    else:
        mean_energies.append(np.nan)
        print(f"  State {k}: 无样本")

# 检查能量趋势（REST2预期：State 0能量最高）
if len(mean_energies) > 1 and not np.isnan(mean_energies[0]):
    if mean_energies[0] > mean_energies[-1]:
        print(f"\n✅ 能量趋势正常: State 0 ({mean_energies[0]:.1f}) > State {n_replicas-1} ({mean_energies[-1]:.1f})")
    else:
        print(f"\n⚠️ 能量趋势异常: State 0应该能量最高（λ=1，完整能垒）")

# 绘制能量分布图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for k in range(n_replicas):
    ax = axes.flat[k]
    mask = np.array([f['state'] == k for f in subsampled_frames])
    if mask.sum() > 0:
        energies_k = u_kn_mbar[k, mask]
        ax.hist(energies_k, bins=50, alpha=0.7, color=f'C{k}', edgecolor='black')
        ax.axvline(energies_k.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={energies_k.mean():.0f}')
        ax.set_xlabel('Reduced Energy', fontsize=10)
        ax.set_ylabel('Sample Count', fontsize=10)
        ax.set_title(f'State {k} (N={mask.sum()})', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No samples', ha='center', va='center', fontsize=14)
        ax.set_title(f'State {k}', fontsize=12)

plt.tight_layout()
plt.savefig('mbar_energy_distributions.png', dpi=300)
print(f"\n✅ 保存: mbar_energy_distributions.png")

# =========================================================================
# Part 6: 轨迹分析与重加权
# =========================================================================
print("\n" + "="*70)
print("[Part 6/8] 轨迹分析：读取构象并计算二面角")
print("="*70)

print(f"\n将读取 {len(subsampled_frames)} 个子采样帧...")
print(f"  每 {TRAJECTORY_INTERVAL} cycles 对应 1 帧DCD")

phi_all = []
psi_all = []
weights_all = []

# 进度报告间隔
report_interval = max(1, len(subsampled_frames) // 20)

for n, frame_info in enumerate(subsampled_frames):
    cycle = frame_info['cycle']
    replica = frame_info['replica']
    weight = weights_state0[n]

    # Cycle到DCD帧号的映射
    frame_idx = cycle // TRAJECTORY_INTERVAL

    # 读取该帧
    try:
        traj = md.load_frame(
            str(TRAJ_DIR / f'r{replica}.dcd'),
            index=frame_idx,
            top=str(TOPOLOGY_FILE)
        )
    except Exception as e:
        print(f"\n⚠️ 读取失败: replica={replica}, cycle={cycle}, frame={frame_idx}")
        print(f"   错误: {e}")
        continue

    # 计算丙氨酸二肽的φ/ψ角
    try:
        phi_indices, phi_rad = md.compute_phi(traj)
        psi_indices, psi_rad = md.compute_psi(traj)

        # 转换为角度（取第一个残基，第一帧）
        phi_deg = np.rad2deg(phi_rad[0, 0])
        psi_deg = np.rad2deg(psi_rad[0, 0])

        phi_all.append(phi_deg)
        psi_all.append(psi_deg)
        weights_all.append(weight)

    except Exception as e:
        print(f"\n⚠️ 二面角计算失败: replica={replica}, cycle={cycle}")
        print(f"   错误: {e}")
        continue

    # 进度报告
    if (n + 1) % report_interval == 0:
        print(f"  进度: {n+1}/{len(subsampled_frames)} ({100*(n+1)/len(subsampled_frames):.1f}%)")

phi_all = np.array(phi_all)
psi_all = np.array(psi_all)
weights_all = np.array(weights_all)

# 重新归一化权重（因为可能有读取失败的帧）
if len(weights_all) < len(subsampled_frames):
    print(f"\n⚠️ 成功读取 {len(weights_all)}/{len(subsampled_frames)} 帧")
    weights_all = weights_all / weights_all.sum()

print(f"\n✅ 完成: {len(phi_all)} 个构象的二面角计算")
print(f"  φ 范围: [{phi_all.min():.1f}°, {phi_all.max():.1f}°]")
print(f"  ψ 范围: [{psi_all.min():.1f}°, {psi_all.max():.1f}°]")

# =========================================================================
# Part 7: 结果对比与可视化
# =========================================================================
print("\n" + "="*70)
print("[Part 7/8] 生成对比图表")
print("="*70)

# -------------------------------------------------------------------------
# 图1: MBAR vs Replica 0 Ramachandran对比
# -------------------------------------------------------------------------
print("\n生成 Ramachandran 对比图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：MBAR重加权
ax = axes[0]
hist_mbar, xedges, yedges = np.histogram2d(
    phi_all, psi_all,
    bins=50,
    range=[[-180, 180], [-180, 180]],
    weights=weights_all,
    density=True
)

im = ax.imshow(
    hist_mbar.T,
    origin='lower',
    extent=[-180, 180, -180, 180],
    cmap='Blues',
    aspect='auto',
    interpolation='bilinear'
)
ax.set_xlabel('φ (degrees)', fontsize=12)
ax.set_ylabel('ψ (degrees)', fontsize=12)
ax.set_title('MBAR Reweighted (State 0, 300K)', fontsize=14, fontweight='bold')
ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)

# 标注主要构象区域
ax.add_patch(Rectangle((-110, 60), 60, 40, fill=False,
             edgecolor='darkred', linewidth=2, linestyle='--', alpha=0.8))
ax.text(-80, 80, 'C7eq', fontsize=11, color='darkred', weight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='darkred'))

ax.add_patch(Rectangle((50, -100), 50, 60, fill=False,
             edgecolor='darkblue', linewidth=2, linestyle='--', alpha=0.8))
ax.text(75, -70, 'C7ax', fontsize=11, color='darkblue', weight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='darkblue'))

plt.colorbar(im, ax=ax, label='Probability Density')

# 右图：原始Replica 0（未重加权）
ax = axes[1]
print("  加载Replica 0轨迹作为对比...")
traj_r0 = md.load(str(TRAJ_DIR / 'r0.dcd'), top=str(TOPOLOGY_FILE))
phi_r0 = np.rad2deg(md.compute_phi(traj_r0)[1][:, 0])
psi_r0 = np.rad2deg(md.compute_psi(traj_r0)[1][:, 0])

hist_r0, _, _ = np.histogram2d(
    phi_r0, psi_r0,
    bins=50,
    range=[[-180, 180], [-180, 180]],
    density=True
)

im2 = ax.imshow(
    hist_r0.T,
    origin='lower',
    extent=[-180, 180, -180, 180],
    cmap='Oranges',
    aspect='auto',
    interpolation='bilinear'
)
ax.set_xlabel('φ (degrees)', fontsize=12)
ax.set_ylabel('ψ (degrees)', fontsize=12)
ax.set_title('Original Replica 0 (未重加权)', fontsize=14, fontweight='bold')
ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)

plt.colorbar(im2, ax=ax, label='Probability Density')

plt.tight_layout()
plt.savefig('ramachandran_mbar_comparison.png', dpi=300)
print("✅ 保存: ramachandran_mbar_comparison.png")

# -------------------------------------------------------------------------
# 图2: 构象占比统计
# -------------------------------------------------------------------------
print("\n计算构象占比...")

def classify_conformation(phi, psi):
    """分类构象（基于文献定义）"""
    if -110 < phi < -50 and 60 < psi < 100:
        return 'C7eq'
    elif 50 < phi < 100 and -100 < psi < -40:
        return 'C7ax'
    elif -90 < phi < -50 and 120 < psi < 160:
        return 'PII'
    elif -70 < phi < -40 and -60 < psi < -20:
        return 'alphaR'
    elif -180 < phi < -120 and 120 < psi < 180:
        return 'beta'
    else:
        return 'other'

# MBAR重加权统计
conf_counts_mbar = {'C7eq': 0, 'C7ax': 0, 'PII': 0, 'alphaR': 0, 'beta': 0, 'other': 0}
for phi, psi, weight in zip(phi_all, psi_all, weights_all):
    conf = classify_conformation(phi, psi)
    conf_counts_mbar[conf] += weight

total_mbar = sum(conf_counts_mbar.values())
conf_fractions_mbar = {k: v/total_mbar for k, v in conf_counts_mbar.items()}

# Replica 0统计
conf_counts_r0 = {'C7eq': 0, 'C7ax': 0, 'PII': 0, 'alphaR': 0, 'beta': 0, 'other': 0}
for phi, psi in zip(phi_r0, psi_r0):
    conf = classify_conformation(phi, psi)
    conf_counts_r0[conf] += 1

total_r0 = len(phi_r0)
conf_fractions_r0 = {k: v/total_r0 for k, v in conf_counts_r0.items()}

print("\n构象占比对比:")
print(f"{'Conformation':<10s}  {'MBAR':>8s}  {'Replica 0':>8s}  {'差异':>8s}")
print("-" * 45)
for conf in ['C7eq', 'C7ax', 'PII', 'alphaR', 'beta', 'other']:
    mbar_pct = 100 * conf_fractions_mbar[conf]
    r0_pct = 100 * conf_fractions_r0[conf]
    diff = mbar_pct - r0_pct
    print(f"{conf:<10s}  {mbar_pct:7.2f}%  {r0_pct:7.2f}%  {diff:+7.2f}%")

# 计算自由能差
kT = 0.593  # kcal/mol @ 300K
if conf_fractions_mbar['C7eq'] > 0 and conf_fractions_mbar['C7ax'] > 0:
    dG_mbar = -kT * np.log(conf_fractions_mbar['C7ax'] / conf_fractions_mbar['C7eq'])
    print(f"\n自由能差 (MBAR):")
    print(f"  ΔG(C7ax - C7eq) = {dG_mbar:.2f} kcal/mol")
    print(f"  文献参考值: 0.6-1.2 kcal/mol")

if conf_fractions_r0['C7eq'] > 0 and conf_fractions_r0['C7ax'] > 0:
    dG_r0 = -kT * np.log(conf_fractions_r0['C7ax'] / conf_fractions_r0['C7eq'])
    print(f"\n自由能差 (Replica 0):")
    print(f"  ΔG(C7ax - C7eq) = {dG_r0:.2f} kcal/mol")

# 绘制构象占比对比图
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(conf_fractions_mbar))
width = 0.35

mbar_values = [100 * conf_fractions_mbar[k] for k in ['C7eq', 'C7ax', 'PII', 'alphaR', 'beta', 'other']]
r0_values = [100 * conf_fractions_r0[k] for k in ['C7eq', 'C7ax', 'PII', 'alphaR', 'beta', 'other']]

bars1 = ax.bar(x - width/2, mbar_values, width, label='MBAR Reweighted',
               color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2, r0_values, width, label='Replica 0 (未重加权)',
               color='coral', edgecolor='black')

ax.set_xlabel('Conformation', fontsize=12)
ax.set_ylabel('Population (%)', fontsize=12)
ax.set_title('构象占比对比：MBAR vs Replica 0', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['C7eq', 'C7ax', 'PII', 'αR', 'β', 'Other'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('conformation_populations.png', dpi=300)
print("\n✅ 保存: conformation_populations.png")

# -------------------------------------------------------------------------
# 图3: MBAR诊断图（Overlap + 权重分布）
# -------------------------------------------------------------------------
print("\n生成 MBAR 诊断图...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 子图1: Overlap矩阵热图
if overlap_matrix is not None:
    ax = axes[0]
    im = ax.imshow(overlap_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(n_replicas))
    ax.set_yticks(range(n_replicas))
    ax.set_xlabel('State j', fontsize=11)
    ax.set_ylabel('State i', fontsize=11)
    ax.set_title('State Overlap Matrix', fontsize=12, fontweight='bold')

    # 在格子中显示数值
    for i in range(n_replicas):
        for j in range(n_replicas):
            text = ax.text(j, i, f'{overlap_matrix[i, j]:.2f}',
                          ha="center", va="center", color="white", fontsize=9)

    plt.colorbar(im, ax=ax, label='Overlap')

# 子图2: 权重分布直方图
ax = axes[1]
ax.hist(weights_state0, bins=50, edgecolor='black', color='steelblue', alpha=0.7)
ax.set_xlabel('Weight', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Weight Distribution (State 0)\nESS={ESS:.0f}, Efficiency={100*efficiency:.1f}%',
             fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.grid(alpha=0.3)

# 子图3: 各State的有效样本数
ax = axes[2]
Neff_k = []
for k in range(n_replicas):
    w_k = mbar.W_nk[:, k]  # pymbar 4.x: W_nk[n, k]
    ess_k = (w_k.sum())**2 / (w_k**2).sum()
    Neff_k.append(ess_k)

ax.bar(range(n_replicas), Neff_k, color='seagreen', edgecolor='black')
ax.axhline(N_total * 0.1, color='red', linestyle='--', label='10% threshold')
ax.set_xlabel('State', fontsize=11)
ax.set_ylabel('Effective Sample Size', fontsize=11)
ax.set_title('Effective Samples per State', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('mbar_diagnostics.png', dpi=300)
print("✅ 保存: mbar_diagnostics.png")

# =========================================================================
# Part 8: 保存结果
# =========================================================================
print("\n" + "="*70)
print("[Part 8/8] 保存结果数据")
print("="*70)

# 保存权重和二面角数据
np.savez_compressed(
    'mbar_weights.npz',
    weights_state0=weights_state0,
    phi=phi_all,
    psi=psi_all,
    N_k=N_k,
    u_kn=u_kn_mbar,
    conf_fractions_mbar=conf_fractions_mbar,
    conf_fractions_r0=conf_fractions_r0,
    ESS=ESS,
    efficiency=efficiency
)
print("✅ 保存: mbar_weights.npz")

# 生成文本报告
report_lines = []
report_lines.append("="*70)
report_lines.append(" REST2 HREMD MBAR重加权分析报告")
report_lines.append("="*70)
report_lines.append("")

report_lines.append("【数据概览】")
report_lines.append(f"  原始数据: {n_cycles} cycles × {n_replicas} replicas = {n_cycles * n_replicas} 总帧")
report_lines.append(f"  子采样后: {len(subsampled_frames)} 独立样本")
report_lines.append(f"  压缩率: {100*len(subsampled_frames)/(n_cycles*n_replicas):.1f}%")
report_lines.append("")

report_lines.append("【子采样统计】")
for k in range(n_replicas):
    report_lines.append(f"  State {k}: {N_k[k]:4d} 样本 ({100*N_k[k]/len(subsampled_frames):5.1f}%)")
report_lines.append("")

report_lines.append("【MBAR收敛】")
if hasattr(mbar, 'iterations'):
    report_lines.append(f"  迭代次数: {mbar.iterations}")
report_lines.append(f"  总样本数: {N_total}")
report_lines.append("")

report_lines.append("【State Overlap】")
if overlap_matrix is not None:
    report_lines.append("  相邻State Overlap:")
    for i in range(n_replicas - 1):
        overlap = overlap_matrix[i, i+1]
        status = "✅" if overlap > 0.05 else "⚠️" if overlap > 0.03 else "❌"
        report_lines.append(f"    State {i} ↔ {i+1}: {overlap:.4f}  {status}")
report_lines.append("")

report_lines.append("【权重统计 (State 0)】")
report_lines.append(f"  有效样本数 (ESS): {ESS:.0f}")
report_lines.append(f"  统计效率: {100*efficiency:.2f}%")
report_lines.append(f"  前{n_50}个样本贡献50%权重 ({100*n_50/len(weights_state0):.1f}%)")
report_lines.append(f"  前{n_90}个样本贡献90%权重 ({100*n_90/len(weights_state0):.1f}%)")
report_lines.append("")

report_lines.append("【构象占比 (MBAR)】")
for conf in ['C7eq', 'C7ax', 'PII', 'alphaR', 'beta', 'other']:
    pct = 100 * conf_fractions_mbar[conf]
    report_lines.append(f"  {conf:<10s}: {pct:6.2f}%")
report_lines.append("")

report_lines.append("【构象占比 (Replica 0)】")
for conf in ['C7eq', 'C7ax', 'PII', 'alphaR', 'beta', 'other']:
    pct = 100 * conf_fractions_r0[conf]
    report_lines.append(f"  {conf:<10s}: {pct:6.2f}%")
report_lines.append("")

if conf_fractions_mbar['C7eq'] > 0 and conf_fractions_mbar['C7ax'] > 0:
    report_lines.append("【自由能差】")
    report_lines.append(f"  MBAR:      ΔG(C7ax - C7eq) = {dG_mbar:.2f} kcal/mol")
    if conf_fractions_r0['C7eq'] > 0 and conf_fractions_r0['C7ax'] > 0:
        report_lines.append(f"  Replica 0: ΔG(C7ax - C7eq) = {dG_r0:.2f} kcal/mol")
    report_lines.append(f"  文献参考: 0.6 - 1.2 kcal/mol")
    report_lines.append("")

report_lines.append("【输出文件】")
report_lines.append("  1. ramachandran_mbar_comparison.png   - Ramachandran对比图")
report_lines.append("  2. conformation_populations.png       - 构象占比柱状图")
report_lines.append("  3. mbar_diagnostics.png               - MBAR诊断图")
report_lines.append("  4. mbar_energy_distributions.png      - 能量分布图")
report_lines.append("  5. mbar_weights.npz                   - 权重数据（可复用）")
report_lines.append("  6. mbar_analysis_report.txt           - 本报告")
report_lines.append("")

report_lines.append("="*70)
report_lines.append("分析完成！")
report_lines.append("="*70)

report_text = "\n".join(report_lines)
pathlib.Path('mbar_analysis_report.txt').write_text(report_text)
print("✅ 保存: mbar_analysis_report.txt")

# 在终端也打印报告
print("\n" + report_text)

print("\n" + "="*70)
print("✅ MBAR重加权分析全部完成！")
print("="*70)
print("\n请检查以下输出文件:")
print("  • ramachandran_mbar_comparison.png")
print("  • conformation_populations.png")
print("  • mbar_diagnostics.png")
print("  • mbar_energy_distributions.png")
print("  • mbar_weights.npz")
print("  • mbar_analysis_report.txt")
print("="*70)
