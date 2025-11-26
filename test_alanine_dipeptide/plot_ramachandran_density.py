#!/usr/bin/env python
"""
单独生成高质量的 2D (φ,ψ) 密度图

用途：
1. 可视化实际的构象分布热区
2. 检查定义的掩码边界是否合理
3. 根据实际分布微调掩码范围
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['font.sans-serif'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False

try:
    import mdtraj as md
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False
    print("❌ 错误: 需要安装 mdtraj")
    print("   请运行: pip install mdtraj")
    exit(1)

print("="*60)
print("生成 Ramachandran 2D 密度图")
print("="*60)

# =====================================================================
# 1. 加载轨迹
# =====================================================================
print("\n[1/3] 加载轨迹文件...")

traj_path = pathlib.Path('outputs_v2_gpu/trajectories/r0.dcd')
if not traj_path.exists():
    traj_path = pathlib.Path('outputs/trajectories/r0.dcd')
    if not traj_path.exists():
        print(f"❌ 错误: 未找到轨迹文件")
        exit(1)

print(f"✓ 加载轨迹: {traj_path}")
traj = md.load(str(traj_path), top='system.pdb')
print(f"  - 总帧数: {len(traj)}")

# =====================================================================
# 2. 计算二面角
# =====================================================================
print("\n[2/3] 计算二面角...")

phi_indices, phi = md.compute_phi(traj)
psi_indices, psi = md.compute_psi(traj)

print(f"  - φ 索引 (原子编号): {phi_indices}")
print(f"  - ψ 索引 (原子编号): {psi_indices}")

phi_deg = np.rad2deg(phi[:, 0])
psi_deg = np.rad2deg(psi[:, 0])

print(f"  - φ 范围: [{phi_deg.min():.1f}°, {phi_deg.max():.1f}°]")
print(f"  - ψ 范围: [{psi_deg.min():.1f}°, {psi_deg.max():.1f}°]")

# =====================================================================
# 3. 生成高质量 2D 密度图
# =====================================================================
print("\n[3/3] 生成 2D 密度图...")

# 定义构象区域掩码（用于标注）
regions = {
    'C7eq': {'bounds': (-110, -50, 60, 100), 'color': 'darkred', 'label_pos': (-80, 80)},
    'C7ax': {'bounds': (50, 100, -100, -40), 'color': 'darkblue', 'label_pos': (75, -70)},
    'PII':  {'bounds': (-90, -50, 120, 160), 'color': 'darkgreen', 'label_pos': (-70, 140)},
    'αR':   {'bounds': (-70, -40, -60, -20), 'color': 'darkorange', 'label_pos': (-55, -40)},
    'β':    {'bounds': (-180, -120, 120, 180), 'color': 'purple', 'label_pos': (-150, 150)},
    'αL':   {'bounds': (40, 80, 20, 60), 'color': 'brown', 'label_pos': (60, 40)},
}

# 创建图形
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.25)

# ========== 左图：2D 直方图（高分辨率）==========
ax1 = fig.add_subplot(gs[0])

# 使用更高的分辨率
bins = 100
h = ax1.hist2d(phi_deg, psi_deg, bins=bins, cmap='viridis', density=True,
               range=[[-180, 180], [-180, 180]])

ax1.set_xlabel('φ (degrees)', fontsize=13, weight='bold')
ax1.set_ylabel('ψ (degrees)', fontsize=13, weight='bold')
ax1.set_title('Ramachandran 2D density (State 0, 300K)', fontsize=14, weight='bold')
ax1.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.3)
ax1.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.3)
ax1.set_xlim([-180, 180])
ax1.set_ylim([-180, 180])
ax1.set_aspect('equal')

# 标注构象区域 + 边界框
for name, props in regions.items():
    phi_min, phi_max, psi_min, psi_max = props['bounds']
    color = props['color']
    label_x, label_y = props['label_pos']

    # 绘制边界矩形
    width = phi_max - phi_min
    height = psi_max - psi_min
    rect = Rectangle((phi_min, psi_min), width, height,
                     fill=False, edgecolor=color, linewidth=2.5,
                     linestyle='--', alpha=0.9)
    ax1.add_patch(rect)

    # 标注标签
    ax1.text(label_x, label_y, name, fontsize=12, color=color, weight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=color, linewidth=2, alpha=0.9))

cbar1 = plt.colorbar(h[3], ax=ax1, label='概率密度', pad=0.02)
cbar1.ax.tick_params(labelsize=10)

# ========== 右图：2D 等高线图 ==========
ax2 = fig.add_subplot(gs[1])

# 生成等高线
H, xedges, yedges = np.histogram2d(phi_deg, psi_deg, bins=bins,
                                   range=[[-180, 180], [-180, 180]],
                                   density=True)
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

# 绘制填充等高线
levels = 15
contourf = ax2.contourf(X, Y, H.T, levels=levels, cmap='coolwarm', alpha=0.8)
contour = ax2.contour(X, Y, H.T, levels=levels, colors='black',
                      linewidths=0.5, alpha=0.4)

ax2.set_xlabel('φ (degrees)', fontsize=13, weight='bold')
ax2.set_ylabel('ψ (degrees)', fontsize=13, weight='bold')
ax2.set_title('Ramachandran map', fontsize=14, weight='bold')  # 英文标题
ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax2.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax2.set_xlim([-180, 180])
ax2.set_ylim([-180, 180])
ax2.set_aspect('equal')

# 标注主要构象区（简化版，只标注名称）
for name, props in regions.items():
    label_x, label_y = props['label_pos']
    color = props['color']
    ax2.text(label_x, label_y, name, fontsize=11, color=color, weight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, linewidth=1.5, alpha=0.85))

cbar2 = plt.colorbar(contourf, ax=ax2, label='概率密度', pad=0.02)
cbar2.ax.tick_params(labelsize=10)

# ========== 保存图形 ==========
plt.tight_layout()
output_file = 'ramachandran_density_2d.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ 保存: {output_file}")

# =====================================================================
# 4. 统计各区域的实际占比
# =====================================================================
print("\n" + "="*60)
print("各构象区域占比（基于当前掩码定义）")
print("="*60)

total_frames = len(phi_deg)

for name, props in regions.items():
    phi_min, phi_max, psi_min, psi_max = props['bounds']
    mask = (phi_deg >= phi_min) & (phi_deg <= phi_max) & \
           (psi_deg >= psi_min) & (psi_deg <= psi_max)
    fraction = mask.sum() / total_frames
    print(f"  {name:6s}: {100.0 * fraction:6.2f}%  (φ ∈ [{phi_min:4.0f}, {phi_max:4.0f}], ψ ∈ [{psi_min:4.0f}, {psi_max:4.0f}])")

# 计算未覆盖的区域
all_masks = np.zeros(total_frames, dtype=bool)
for name, props in regions.items():
    phi_min, phi_max, psi_min, psi_max = props['bounds']
    mask = (phi_deg >= phi_min) & (phi_deg <= phi_max) & \
           (psi_deg >= psi_min) & (psi_deg <= psi_max)
    all_masks |= mask

other_fraction = (~all_masks).sum() / total_frames
print(f"  {'其他':6s}: {100.0 * other_fraction:6.2f}%  (未定义区域)")
print(f"\n  总计: {100.0 * (all_masks.sum() / total_frames + other_fraction):.2f}%")

# =====================================================================
# 5. 寻找密度峰值位置（帮助微调掩码）
# =====================================================================
print("\n" + "="*60)
print("密度峰值位置（前 5 个）")
print("="*60)

# 找出密度最高的 5 个网格点
H_flat = H.flatten()
peak_indices = np.argsort(H_flat)[-5:][::-1]  # 降序

for i, idx in enumerate(peak_indices, 1):
    row, col = np.unravel_index(idx, H.shape)
    phi_peak = xedges[row] + (xedges[1] - xedges[0]) / 2
    psi_peak = yedges[col] + (yedges[1] - yedges[0]) / 2
    density = H_flat[idx]
    print(f"  峰值 {i}: φ = {phi_peak:6.1f}°, ψ = {psi_peak:6.1f}°  (密度 = {density:.4f})")

print("\n💡 提示:")
print("  1. 查看 ramachandran_density_2d.png 检查掩码边界是否覆盖了热区")
print("  2. 根据实际密度分布，在 03_analyze_results_v2.py 中调整掩码范围")
print("  3. 峰值位置可以作为掩码中心的参考")

print("\n✅ 完成！")
