#!/usr/bin/env python
"""
步骤 4: 分析所有温度副本的采样分布

功能:
1. 加载所有温度副本的轨迹
2. 生成多温度 Ramachandran 对比图
3. 分析每个温度的构象分布
4. 评估 REST2 采样效率
"""

import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    import mdtraj as md
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False
    print("❌ mdtraj 未安装，无法运行此脚本")
    sys.exit(1)

print("=" * 70)
print("分析所有温度副本的采样分布")
print("=" * 70)

# =====================================================================
# 1. 计算温度梯度
# =====================================================================
print("\n[1/5] 计算温度梯度...")

T_min = 300.0  # K
T_max = 1000.0  # K
n_replicas = 6

# 几何分布
temperatures = [
    T_min * (T_max / T_min) ** (i / (n_replicas - 1))
    for i in range(n_replicas)
]

print(f"\n✓ 温度梯度 ({n_replicas} 个副本):")
for i, T in enumerate(temperatures):
    print(f"  State {i}: {T:.1f} K")

# =====================================================================
# 2. 加载所有副本轨迹
# =====================================================================
print("\n[2/5] 加载所有副本轨迹...")

traj_dir = pathlib.Path('outputs_v2_gpu/trajectories')
if not traj_dir.exists():
    traj_dir = pathlib.Path('outputs/trajectories')
    if not traj_dir.exists():
        print(f"❌ 未找到轨迹目录")
        sys.exit(1)

topology_file = pathlib.Path('system.pdb')
if not topology_file.exists():
    topology_file = pathlib.Path('test_alanine_dipeptide/system.pdb')
    if not topology_file.exists():
        print(f"❌ 未找到拓扑文件 system.pdb")
        sys.exit(1)

trajectories = []
phi_angles = []
psi_angles = []

for i in range(n_replicas):
    traj_file = traj_dir / f"r{i}.dcd"
    if not traj_file.exists():
        print(f"⚠️ 未找到轨迹文件: {traj_file}")
        continue

    print(f"  加载 {traj_file.name} (T={temperatures[i]:.1f}K)...", end=" ")
    traj = md.load(str(traj_file), top=str(topology_file))
    trajectories.append(traj)

    # 计算扭转角
    phi_indices, phi = md.compute_phi(traj)
    psi_indices, psi = md.compute_psi(traj)

    phi_deg = np.rad2deg(phi[:, 0])
    psi_deg = np.rad2deg(psi[:, 0])

    phi_angles.append(phi_deg)
    psi_angles.append(psi_deg)

    print(f"{len(traj)} 帧 ✓")

print(f"\n✓ 成功加载 {len(trajectories)} 个副本")

# =====================================================================
# 3. 生成多温度 Ramachandran 对比图
# =====================================================================
print("\n[3/5] 生成 Ramachandran 对比图...")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

for i in range(len(trajectories)):
    ax = fig.add_subplot(gs[i // 3, i % 3])

    phi = phi_angles[i]
    psi = psi_angles[i]

    # 绘制 2D 直方图
    h = ax.hist2d(phi, psi, bins=50, cmap='Blues', density=True,
                  range=[[-180, 180], [-180, 180]], vmin=0, vmax=0.0003)

    ax.set_xlabel('φ (degrees)', fontsize=11)
    ax.set_ylabel('ψ (degrees)', fontsize=11)
    ax.set_title(f'State {i}: {temperatures[i]:.0f} K', fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlim([-180, 180])
    ax.set_ylim([-180, 180])

    # 标注主要构象区域
    ax.text(-80, 80, 'C7eq', fontsize=10, color='red', weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.text(60, -60, 'C7ax', fontsize=10, color='blue', weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # 添加色标
    if i % 3 == 2:  # 每行最右边的图
        plt.colorbar(h[3], ax=ax, label='Probability Density')

plt.savefig('ramachandran_all_temps.png', dpi=300, bbox_inches='tight')
print(f"✓ 保存: ramachandran_all_temps.png")

# =====================================================================
# 4. 分析构象分布
# =====================================================================
print("\n[4/5] 分析构象分布...")

# 定义构象区域（基于 Ramachandran plot）
def classify_conformation(phi, psi):
    """分类构象"""
    c7eq_mask = (phi < -30) & (psi > 30)
    c7ax_mask = (phi > 30) & (psi < -30)
    alpha_mask = (phi > -100) & (phi < -30) & (psi > -70) & (psi < -20)

    c7eq_frac = c7eq_mask.sum() / len(phi)
    c7ax_frac = c7ax_mask.sum() / len(phi)
    alpha_frac = alpha_mask.sum() / len(phi)
    other_frac = 1.0 - c7eq_frac - c7ax_frac - alpha_frac

    return {
        'C7eq': c7eq_frac,
        'C7ax': c7ax_frac,
        'α-helix': alpha_frac,
        'Other': other_frac
    }

# 统计每个温度的构象分布
conformation_stats = []
for i in range(len(trajectories)):
    stats = classify_conformation(phi_angles[i], psi_angles[i])
    stats['temperature'] = temperatures[i]
    stats['replica'] = i
    conformation_stats.append(stats)

# 打印统计表格
print(f"\n构象分布统计:")
print(f"{'State':<8} {'Temp (K)':<10} {'C7eq':<10} {'C7ax':<10} {'α-helix':<10} {'Other':<10}")
print("-" * 58)

for stats in conformation_stats:
    print(f"{stats['replica']:<8} {stats['temperature']:<10.0f} "
          f"{stats['C7eq']*100:<10.1f} {stats['C7ax']*100:<10.1f} "
          f"{stats['α-helix']*100:<10.1f} {stats['Other']*100:<10.1f}")

# 绘制构象分布柱状图
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# 子图1: 堆叠柱状图
x_labels = [f"State {i}\n{temperatures[i]:.0f}K" for i in range(len(trajectories))]
c7eq_vals = [s['C7eq'] * 100 for s in conformation_stats]
c7ax_vals = [s['C7ax'] * 100 for s in conformation_stats]
alpha_vals = [s['α-helix'] * 100 for s in conformation_stats]
other_vals = [s['Other'] * 100 for s in conformation_stats]

x = np.arange(len(trajectories))
width = 0.6

axes[0].bar(x, c7eq_vals, width, label='C7eq', color='#1f77b4')
axes[0].bar(x, c7ax_vals, width, bottom=c7eq_vals, label='C7ax', color='#ff7f0e')
axes[0].bar(x, alpha_vals, width,
            bottom=np.array(c7eq_vals) + np.array(c7ax_vals),
            label='α-helix', color='#2ca02c')
axes[0].bar(x, other_vals, width,
            bottom=np.array(c7eq_vals) + np.array(c7ax_vals) + np.array(alpha_vals),
            label='Other', color='#d62728')

axes[0].set_xlabel('Temperature State', fontsize=12)
axes[0].set_ylabel('Population (%)', fontsize=12)
axes[0].set_title('Conformation Distribution by Temperature', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(x_labels)
axes[0].legend(loc='upper right')
axes[0].grid(axis='y', alpha=0.3)

# 子图2: C7eq vs C7ax 比例
c7eq_ratio = np.array(c7eq_vals) / (np.array(c7eq_vals) + np.array(c7ax_vals) + 1e-10)
c7ax_ratio = np.array(c7ax_vals) / (np.array(c7eq_vals) + np.array(c7ax_vals) + 1e-10)

axes[1].plot(temperatures[:len(trajectories)], c7eq_ratio, 'o-',
             label='C7eq', linewidth=2, markersize=8)
axes[1].plot(temperatures[:len(trajectories)], c7ax_ratio, 's-',
             label='C7ax', linewidth=2, markersize=8)
axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal population')
axes[1].set_xlabel('Temperature (K)', fontsize=12)
axes[1].set_ylabel('Relative Population (C7eq+C7ax=1)', fontsize=12)
axes[1].set_title('C7eq ↔ C7ax Balance vs Temperature', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('conformation_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ 保存: conformation_distribution.png")

# =====================================================================
# 5. 生成文本报告
# =====================================================================
print("\n[5/5] 生成分析报告...")

report = []
report.append("=" * 70)
report.append("REST2 多温度采样分析报告")
report.append("=" * 70)
report.append("")
report.append(f"副本数量: {n_replicas}")
report.append(f"温度范围: {T_min:.1f} - {T_max:.1f} K")
report.append("")
report.append("构象分布统计:")
report.append("-" * 70)
report.append(f"{'State':<8} {'Temp (K)':<10} {'C7eq (%)':<12} {'C7ax (%)':<12} {'Ratio':<10}")
report.append("-" * 70)

for stats in conformation_stats:
    c7eq_pct = stats['C7eq'] * 100
    c7ax_pct = stats['C7ax'] * 100
    ratio = c7eq_pct / (c7ax_pct + 1e-10)
    report.append(f"{stats['replica']:<8} {stats['temperature']:<10.0f} "
                  f"{c7eq_pct:<12.1f} {c7ax_pct:<12.1f} {ratio:<10.2f}")

report.append("")
report.append("采样评估:")
report.append("-" * 70)

# 检查高温副本是否能均匀采样
high_temp_idx = -1
high_temp_stats = conformation_stats[high_temp_idx]
high_c7eq = high_temp_stats['C7eq'] * 100
high_c7ax = high_temp_stats['C7ax'] * 100
high_ratio = high_c7eq / (high_c7ax + 1e-10)

report.append(f"最高温度 ({high_temp_stats['temperature']:.0f} K):")
report.append(f"  - C7eq: {high_c7eq:.1f}%")
report.append(f"  - C7ax: {high_c7ax:.1f}%")
report.append(f"  - 比例: {high_ratio:.2f}")

if 2.0 < high_ratio < 5.0:
    report.append(f"  ✅ 高温采样较为均衡（接近热力学比例 2:1 到 3:1）")
elif high_ratio > 10.0:
    report.append(f"  ⚠️ 高温仍偏向 C7eq，建议增加温度或延长模拟时间")
elif high_ratio < 1.5:
    report.append(f"  ⚠️ 高温采样异常均匀，可能温度过高")
else:
    report.append(f"  ✅ 采样基本合理")

# 检查低温副本
low_temp_stats = conformation_stats[0]
low_c7eq = low_temp_stats['C7eq'] * 100
low_c7ax = low_temp_stats['C7ax'] * 100
low_ratio = low_c7eq / (low_c7ax + 1e-10)

report.append(f"\n最低温度 ({low_temp_stats['temperature']:.0f} K):")
report.append(f"  - C7eq: {low_c7eq:.1f}%")
report.append(f"  - C7ax: {low_c7ax:.1f}%")
report.append(f"  - 比例: {low_ratio:.2f}")

if low_c7ax > 5.0:
    report.append(f"  ✅ 低温副本有 C7ax 采样，REST2 交换有效")
else:
    report.append(f"  ⚠️ 低温副本几乎无 C7ax 采样，依赖高温副本的信息交换")

report.append("")
report.append("总体结论:")
report.append("-" * 70)

# 计算所有温度的平均比例
avg_c7eq = np.mean([s['C7eq'] for s in conformation_stats]) * 100
avg_c7ax = np.mean([s['C7ax'] for s in conformation_stats]) * 100
overall_ratio = avg_c7eq / (avg_c7ax + 1e-10)

report.append(f"所有副本平均:")
report.append(f"  - C7eq: {avg_c7eq:.1f}%")
report.append(f"  - C7ax: {avg_c7ax:.1f}%")
report.append(f"  - 比例: {overall_ratio:.2f}")

if high_c7ax > 20.0 and low_c7ax > 1.0:
    report.append(f"\n✅ REST2 采样充分：")
    report.append(f"   - 高温副本能有效采样 C7ax 构象")
    report.append(f"   - 低温副本通过交换获得了构象多样性")
elif high_c7ax > 30.0:
    report.append(f"\n⚠️ REST2 采样基本合理，但低温副本采样不足：")
    report.append(f"   - 高温副本采样充分")
    report.append(f"   - 建议增加模拟时间或提高交换频率")
else:
    report.append(f"\n❌ REST2 采样不足：")
    report.append(f"   - 所有温度都未能充分采样 C7ax 构象")
    report.append(f"   - 建议增加最高温度（如 1200K）或延长模拟时间")

report.append("")
report.append("=" * 70)

# 保存报告
report_text = "\n".join(report)
with open('temperature_analysis_report.txt', 'w') as f:
    f.write(report_text)

print(report_text)
print(f"\n✓ 保存报告: temperature_analysis_report.txt")

print("\n" + "=" * 70)
print("分析完成！")
print("=" * 70)
print("\n生成的文件:")
print("  1. ramachandran_all_temps.png - 所有温度的 Ramachandran 对比图")
print("  2. conformation_distribution.png - 构象分布统计图")
print("  3. temperature_analysis_report.txt - 详细文本报告")
