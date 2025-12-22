#!/usr/bin/env python
"""
HREMD 采样质量评估脚本

评估指标:
1. 转移状态矩阵 - 副本在不同状态间的交换频率
2. 次主导特征值 λ₂ - 状态间的流动速度/混合时间
3. 副本状态轨迹图 - 每个副本访问状态的轨迹
"""

import argparse
import pathlib
import sys

import numpy as np
import pyarrow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_transition_matrix(replica_states):
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
    n_states = n_replicas

    # 统计转移次数
    transition_counts = np.zeros((n_states, n_states), dtype=int)

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
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    P = np.divide(
        transition_matrix, row_sums,
        out=np.zeros_like(transition_matrix),
        where=row_sums > 0
    )

    # 计算特征值
    eigenvalues, _ = np.linalg.eig(P.T)
    eigenvalues = np.sort(np.abs(eigenvalues.real))[::-1]

    lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0

    return lambda_2, eigenvalues


def compute_mixing_time(lambda_2):
    """计算混合时间"""
    if lambda_2 >= 1.0 or lambda_2 <= 0:
        return float('inf')
    return -1.0 / np.log(lambda_2)


def count_roundtrips(replica_states, replica_idx=0):
    """计算指定副本完成的round-trip次数"""
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


def evaluate_mixing(samples_file, output_dir=None):
    """执行混合统计评估"""

    samples_path = pathlib.Path(samples_file)
    if not samples_path.exists():
        print(f"[FAIL] 文件不存在: {samples_path}")
        sys.exit(1)

    if output_dir is None:
        output_dir = samples_path.parent
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HREMD 采样质量评估")
    print("=" * 60)

    # 加载数据
    print(f"\n[1/4] 加载采样数据: {samples_path}")
    with pyarrow.OSFile(str(samples_path), 'rb') as f:
        reader = pyarrow.RecordBatchStreamReader(f)
        table = reader.read_all()
        df = table.to_pandas()

    print(f"  总周期数: {len(df)}")

    # 提取副本状态轨迹
    replica_states = np.array([np.array(x) for x in df['replica_to_state_idx']])
    n_cycles, n_replicas = replica_states.shape
    print(f"  副本数: {n_replicas}")
    print(f"  数据形状: {replica_states.shape}")

    # =========================================================================
    # 评估指标 1: 转移状态矩阵
    # =========================================================================
    print(f"\n[2/4] 计算转移状态矩阵...")

    transition_matrix, transition_counts = compute_transition_matrix(replica_states)

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
            print(f"{transition_matrix[i,j]:7.4f}", end="  ")
        print()

    # 分析对角线元素
    diag_elements = np.diag(transition_matrix)
    diag_min, diag_max = diag_elements.min(), diag_elements.max()

    # 分析相邻态交换概率
    neighbor_probs = [transition_matrix[i, i+1] for i in range(n_replicas - 1)]
    neighbor_min, neighbor_max = min(neighbor_probs), max(neighbor_probs)

    print(f"\n  评估结果:")
    diag_status = "[OK]" if diag_max < 0.7 else "[WARN]"
    print(f"    - 对角线元素范围: [{diag_min:.4f}, {diag_max:.4f}] {diag_status}")
    neighbor_status = "[OK]" if neighbor_min > 0.2 else "[WARN]"
    print(f"    - 相邻态交换概率范围: [{neighbor_min:.4f}, {neighbor_max:.4f}] {neighbor_status}")

    # =========================================================================
    # 评估指标 2: 次主导特征值
    # =========================================================================
    print(f"\n[3/4] 计算次主导特征值...")

    lambda_2, all_eigenvalues = compute_subdominant_eigenvalue(transition_matrix)
    mixing_time = compute_mixing_time(lambda_2)

    print(f"\n  特征值列表: {np.round(all_eigenvalues, 4).tolist()}")
    print(f"  次主导特征值 lambda_2 = {lambda_2:.4f}")
    print(f"  混合时间 tau_mix = {mixing_time:.2f} 次迭代")

    if lambda_2 < 0.5:
        lambda_status = "[OK] 优秀"
        mixing_quality = "充分混合"
    elif lambda_2 < 0.8:
        lambda_status = "[OK] 可接受"
        mixing_quality = "中等混合"
    elif lambda_2 < 0.95:
        lambda_status = "[WARN] 警告"
        mixing_quality = "混合较慢"
    else:
        lambda_status = "[FAIL] 失败"
        mixing_quality = "混合不良"

    print(f"\n  评估结果:")
    print(f"    - lambda_2 判定: {lambda_status}")
    print(f"    - 混合质量: {mixing_quality}")

    # =========================================================================
    # 评估指标 3: 副本轨迹诊断
    # =========================================================================
    print(f"\n[4/4] 副本轨迹诊断...")

    # 状态覆盖度
    state_coverage = []
    for r in range(n_replicas):
        unique_states = len(np.unique(replica_states[:, r]))
        state_coverage.append(unique_states)
        coverage_pct = 100.0 * unique_states / n_replicas
        status = "[OK]" if coverage_pct == 100 else "[WARN]"
        print(f"    Replica {r}: {unique_states}/{n_replicas} states ({coverage_pct:.1f}%) {status}")

    # Round-trip 统计
    roundtrips = [count_roundtrips(replica_states, r) for r in range(n_replicas)]
    total_roundtrips = sum(roundtrips)
    avg_roundtrips = total_roundtrips / n_replicas

    print(f"\n  Round-trip 统计:")
    for r in range(n_replicas):
        print(f"    Replica {r}: {roundtrips[r]} 次")
    print(f"    总计: {total_roundtrips} 次, 平均: {avg_roundtrips:.1f} 次/副本")

    rt_status = "[OK]" if avg_roundtrips >= 3 else "[WARN]"
    print(f"    评估 (平均 >= 3): {rt_status}")

    # =========================================================================
    # 生成可视化
    # =========================================================================
    print(f"\n生成可视化图表...")

    # 图1: 转移矩阵热图 + 特征值分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im = axes[0].imshow(transition_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xlabel('Target State j')
    axes[0].set_ylabel('Source State i')
    axes[0].set_title('State Transition Probability Matrix P[i,j]')
    axes[0].set_xticks(range(n_replicas))
    axes[0].set_yticks(range(n_replicas))
    axes[0].set_xticklabels([f'S{i}' for i in range(n_replicas)])
    axes[0].set_yticklabels([f'S{i}' for i in range(n_replicas)])

    for i in range(n_replicas):
        for j in range(n_replicas):
            val = transition_matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            axes[0].text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)

    plt.colorbar(im, ax=axes[0], label='Transition Probability')

    x_pos = range(len(all_eigenvalues))
    colors = ['green' if i == 0 else ('orange' if all_eigenvalues[i] > 0.8 else 'blue')
              for i in range(len(all_eigenvalues))]
    axes[1].bar(x_pos, all_eigenvalues, color=colors)
    axes[1].axhline(0.8, color='red', linestyle='--', alpha=0.7, label='lambda_2 < 0.8 threshold')
    axes[1].set_xlabel('Eigenvalue Index')
    axes[1].set_ylabel('Eigenvalue |lambda|')
    axes[1].set_title(f'Eigenvalue Distribution (lambda_2 = {lambda_2:.4f})')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'l_{i}' for i in range(len(all_eigenvalues))])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig1_path = output_dir / 'transition_matrix_heatmap.png'
    plt.savefig(fig1_path, dpi=300)
    print(f"  [OK] 保存: {fig1_path}")
    plt.close()

    # 图2: 副本轨迹 + 采样统计 (改进版)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 左上: 副本轨迹的热图表示（更清晰的可视化）
    # 将轨迹分成时间窗口，统计每个窗口内副本在各状态的停留比例
    n_windows = 100  # 分成100个时间窗口
    window_size = n_cycles // n_windows
    trajectory_heatmap = np.zeros((n_replicas, n_windows))

    for r in range(n_replicas):
        for w in range(n_windows):
            start = w * window_size
            end = (w + 1) * window_size
            # 计算该窗口内的平均状态
            trajectory_heatmap[r, w] = np.mean(replica_states[start:end, r])

    im1 = axes[0, 0].imshow(trajectory_heatmap, cmap='viridis', aspect='auto',
                            extent=[0, n_cycles, n_replicas-0.5, -0.5])
    axes[0, 0].set_xlabel('Iteration Cycle')
    axes[0, 0].set_ylabel('Replica Index')
    axes[0, 0].set_title(f'Replica State Evolution (lambda_2={lambda_2:.3f}, tau_mix={mixing_time:.1f})')
    axes[0, 0].set_yticks(range(n_replicas))
    axes[0, 0].set_yticklabels([f'R{i}' for i in range(n_replicas)])
    cbar1 = plt.colorbar(im1, ax=axes[0, 0])
    cbar1.set_label('Average State')

    # 右上: 状态占用时间分布（每个副本在各状态停留的比例）
    state_occupancy = np.zeros((n_replicas, n_replicas))
    for r in range(n_replicas):
        for s in range(n_replicas):
            state_occupancy[r, s] = np.sum(replica_states[:, r] == s) / n_cycles

    im2 = axes[0, 1].imshow(state_occupancy, cmap='YlOrRd', aspect='auto', vmin=0)
    axes[0, 1].set_xlabel('State Index')
    axes[0, 1].set_ylabel('Replica Index')
    axes[0, 1].set_title('State Occupancy Fraction (ideal: ~0.17 uniform)')
    axes[0, 1].set_xticks(range(n_replicas))
    axes[0, 1].set_yticks(range(n_replicas))
    axes[0, 1].set_xticklabels([f'S{i}' for i in range(n_replicas)])
    axes[0, 1].set_yticklabels([f'R{i}' for i in range(n_replicas)])

    for r in range(n_replicas):
        for s in range(n_replicas):
            val = state_occupancy[r, s]
            color = 'white' if val > 0.3 else 'black'
            axes[0, 1].text(s, r, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im2, ax=axes[0, 1], label='Fraction')

    # 左下: 状态覆盖度柱状图
    colors_coverage = ['green' if c == n_replicas else 'orange' for c in state_coverage]
    axes[1, 0].bar(range(n_replicas), state_coverage, color=colors_coverage, edgecolor='black')
    axes[1, 0].axhline(n_replicas, color='green', linestyle='--', alpha=0.7,
                       label=f'Full Coverage ({n_replicas})')
    axes[1, 0].set_xlabel('Replica Index')
    axes[1, 0].set_ylabel('Unique States Visited')
    axes[1, 0].set_title('State Coverage per Replica')
    axes[1, 0].set_xticks(range(n_replicas))
    axes[1, 0].set_xticklabels([f'R{i}' for i in range(n_replicas)])
    axes[1, 0].set_ylim(0, n_replicas + 1)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 右下: Round-trip次数柱状图
    colors_rt = ['green' if rt >= 3 else 'orange' for rt in roundtrips]
    axes[1, 1].bar(range(n_replicas), roundtrips, color=colors_rt, edgecolor='black')
    axes[1, 1].axhline(3, color='red', linestyle='--', alpha=0.7, label='Minimum (3)')
    axes[1, 1].axhline(avg_roundtrips, color='blue', linestyle='-', alpha=0.7,
                       label=f'Average ({avg_roundtrips:.1f})')
    axes[1, 1].set_xlabel('Replica Index')
    axes[1, 1].set_ylabel('Round-trip Count')
    axes[1, 1].set_title(f'Round-trips per Replica (Total: {total_roundtrips})')
    axes[1, 1].set_xticks(range(n_replicas))
    axes[1, 1].set_xticklabels([f'R{i}' for i in range(n_replicas)])
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig2_path = output_dir / 'replica_state_trajectory.png'
    plt.savefig(fig2_path, dpi=300)
    print(f"  [OK] 保存: {fig2_path}")
    plt.close()

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
        f"    - 对角线 < 0.7: {diag_status}",
        f"    - 相邻态交换 > 0.2: {neighbor_status}",
        "",
        "二、次主导特征值分析",
        "-" * 40,
        f"  特征值列表: {np.round(all_eigenvalues, 4).tolist()}",
        f"  次主导特征值 lambda_2 = {lambda_2:.4f}",
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
        report_lines.append(f"    副本 {r}: {state_coverage[r]}/{n_replicas} ({coverage_pct:.1f}%) {status}")

    report_lines.extend([
        "",
        f"  Round-trip次数: 总计 {total_roundtrips}, 平均 {avg_roundtrips:.1f} 次/副本 {rt_status}",
        "",
        "=" * 60,
        "",
        "评估判据说明:",
        "  - 对角线元素: < 0.5 (优秀), < 0.7 (可接受), >= 0.7 (警告)",
        "  - 相邻态交换: > 0.3 (优秀), > 0.2 (可接受), <= 0.2 (警告)",
        "  - lambda_2: < 0.5 (优秀), < 0.8 (可接受), < 0.95 (警告), >= 0.95 (失败)",
        "  - tau_mix: < 2 (优秀), < 5 (可接受), < 20 (警告), >= 20 (失败)",
        "  - Round-trip: >= 5 (优秀), >= 3 (可接受), >= 1 (警告), 0 (失败)",
    ])

    report_text = "\n".join(report_lines)
    report_path = output_dir / 'mixing_statistics_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"  [OK] 保存: {report_path}")

    # 总结
    print("\n" + "=" * 60)
    print("评估总结")
    print("=" * 60)
    print(f"  次主导特征值 lambda_2 = {lambda_2:.4f} {lambda_status}")
    print(f"  混合时间 tau_mix = {mixing_time:.2f} 次迭代")
    print(f"  平均 Round-trip = {avg_roundtrips:.1f} 次/副本 {rt_status}")
    print(f"\n  输出文件:")
    print(f"    - {fig1_path}")
    print(f"    - {fig2_path}")
    print(f"    - {report_path}")
    print("=" * 60)

    return {
        'lambda_2': lambda_2,
        'mixing_time': mixing_time,
        'transition_matrix': transition_matrix,
        'eigenvalues': all_eigenvalues,
        'roundtrips': roundtrips,
        'state_coverage': state_coverage,
    }


def main():
    parser = argparse.ArgumentParser(description='HREMD 采样质量评估')
    parser.add_argument('samples_file', type=str, help='samples.arrow 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出目录')

    args = parser.parse_args()

    evaluate_mixing(args.samples_file, args.output)


if __name__ == '__main__':
    main()
