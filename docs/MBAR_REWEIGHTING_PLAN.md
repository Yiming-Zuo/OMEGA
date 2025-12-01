# REST2 HREMD 数据的 MBAR 重加权完整方案

## **项目概述**

### 目标
对 REST2 HREMD 模拟的所有副本数据进行严格的统计力学重加权，恢复目标体系（State 0, 300K, λ=1）的物理分布。

### 输入数据
```
outputs_v2_gpu/
├── samples.arrow          # 50000 cycles 的能量和交换记录
│   ├── u_kn[cycle][replica][state]: 每帧在所有state下的约化能量
│   ├── replica_to_state_idx[cycle][replica]: 副本-状态映射
│   ├── n_proposed_swaps, n_accepted_swaps: 交换统计
│   └── step: OpenMM step编号
├── trajectories/          # 6个副本的轨迹文件
│   ├── r0.dcd ~ r5.dcd   # 每个2500帧，timestep=1ps
│   └── 保存间隔: 每20 cycles = 1帧
└── checkpoint.pkl         # 检查点文件
```

### 数据对齐验证
- **Cycle与Frame对应关系**：`frame_index = cycle // 20`
- **总帧数一致性**：50000 cycles ÷ 20 = 2500 frames
- **时间跨度**：2500 frames × 1 ps = 2.5 ns（每个replica）
- **总采样量**：6 replicas × 2500 frames = 15000 帧

---

## **方法论：MBAR重加权理论**

### REST2的物理图景

**REST2 = Hamiltonian Replica Exchange（非温度交换）**

1. **固定温度**：所有副本在相同温度β₀（300K）运行
2. **不同哈密顿量**：通过λ缩放溶质相互作用
   ```
   State k: U_k(x) = U_rest(x; λ_k)
   λ_k = β₀/β_k（有效缩放因子）
   ```
3. **副本交换**：
   - 交换的是**构象和速度**（不是温度）
   - 遵循Metropolis准则确保联合分布平衡
   - `replica_to_state_idx`记录动态映射

### 采样分布与目标分布

**各副本的采样分布**：
```
p_k(x) = exp[-β₀ U_k(x)] / Z_k
```
- State 0 (λ=1): 物理体系，完整能垒
- State k (λ<1): 削弱能垒，增强采样

**目标分布**：State 0的平衡分布
```
p_0(x) = exp[-β₀ U_0(x)] / Z_0
```

**MBAR核心思想**：
利用所有state的采样数据，通过自洽求解配分函数比值{Z_k}，计算任意样本在目标state下的权重。

### MBAR权重公式

对于来自state k的样本x_n，其在目标state 0下的权重为：

```
w_n^(0) = exp[-β₀ U_0(x_n)] / Σ_k N_k exp[f_k - β₀ U_k(x_n)]
```

其中：
- `f_k = -ln(Z_k)`：各state的无量纲自由能（MBAR自洽求解）
- `N_k`：state k采样的样本数
- `β₀ U_k(x_n)`：femto记录的`u_kn[n, k]`（约化能量）

**重要**：
- **不涉及温度比值** β_k/β₀（这是温度REMD的公式）
- **只有能量差** U_0(x) - U_k(x)，都在β₀温度下

---

## **实现方案：逐步流程**

### Phase 1: 数据准备与子采样

#### 步骤1.1：加载原始数据
```python
import pyarrow
import numpy as np

# 加载samples.arrow
with pyarrow.OSFile('outputs_v2_gpu/samples.arrow', 'rb') as f:
    table = pyarrow.RecordBatchStreamReader(f).read_all()
    df = table.to_pandas()

# 提取关键数据
u_kn = df['u_kn']              # [cycle][replica][state]
replica_to_state = df['replica_to_state_idx']  # [cycle][replica]
```

#### 步骤1.2：逐Replica子采样（关键修正）

**[FAIL] 错误做法**：按state分组后做时间序列分析
```python
# 这会破坏时间连续性！
for state_k in range(6):
    energies = [u for samples in state_k if ...]
    g = pymbar.timeseries.statistical_inefficiency(energies)  # 错误！
```

**[OK] 正确做法**：对每个replica的完整轨迹单独子采样
```python
subsampled_frames = []

for replica in range(6):
    # 提取该replica的完整能量时间序列
    replica_energies = []
    for cycle in range(n_cycles):
        state = replica_to_state_idx[cycle, replica]
        u_self = u_kn[cycle, replica, state]  # 该帧在自己state下的能量
        replica_energies.append(u_self)

    replica_energies = np.array(replica_energies)

    # 平衡化检测
    t0, g, Neff = pymbar.timeseries.detect_equilibration(replica_energies)
    print(f"Replica {replica}: 平衡时间={t0}, 不相关时间g={g:.1f}, Neff={Neff:.0f}")

    # 从t0开始子采样
    equilibrated = replica_energies[t0:]
    indices = pymbar.timeseries.subsample_correlated_data(equilibrated, g=g)

    # 保存子采样帧的完整信息
    for idx in indices:
        global_cycle = t0 + idx
        state = replica_to_state_idx[global_cycle, replica]
        u_all_states = u_kn[global_cycle, replica, :]

        subsampled_frames.append({
            'cycle': global_cycle,
            'replica': replica,
            'state': state,        # 采样该帧时所处的state
            'u_kn': u_all_states   # 该构象在所有state下的能量
        })
```

**理论依据**：
- 单个replica的轨迹是连续的马尔可夫过程
- 自相关函数的计算要求时间连续性
- 副本交换不影响单个replica的时间演化（只改变哈密顿量）

#### 步骤1.3：构建MBAR输入

```python
# 统计每个state的子采样样本数
N_k = np.zeros(6, dtype=int)
for frame in subsampled_frames:
    N_k[frame['state']] += 1

print("\n子采样后各State的样本数:")
for k in range(6):
    print(f"  State {k}: {N_k[k]} 样本")

# 构建u_kn矩阵 [K, N]
N_total = len(subsampled_frames)
u_kn_mbar = np.zeros((6, N_total))

for n, frame in enumerate(subsampled_frames):
    u_kn_mbar[:, n] = frame['u_kn']

# 一致性检查
assert N_k.sum() == N_total
assert u_kn_mbar.shape == (6, N_total)
print(f"\n[OK] MBAR输入矩阵: {u_kn_mbar.shape}, 总样本数: {N_total}")
```

---

### Phase 2: MBAR计算与诊断

#### 步骤2.1：初始化MBAR

```python
import pymbar

# 初始化MBAR求解器
mbar = pymbar.MBAR(u_kn_mbar, N_k, verbose=True, maximum_iterations=10000)

print("\n[OK] MBAR收敛完成")
print(f"  迭代次数: {mbar.iterations}")
```

#### 步骤2.2：诊断检查1 - State Overlap

```python
# 计算state间的构象重叠
overlap_matrix = mbar.compute_overlap()

print("\n" + "="*60)
print("State Overlap Matrix")
print("="*60)
print("      ", end="")
for j in range(6):
    print(f"   S{j}  ", end="")
print()
for i in range(6):
    print(f"S{i}: ", end="")
    for j in range(6):
        val = overlap_matrix['matrix'][i, j]
        print(f" {val:6.3f}", end="")
    print()

# 检查相邻state的overlap（关键指标）
print("\n相邻State Overlap:")
for i in range(5):
    overlap = overlap_matrix['matrix'][i, i+1]
    if overlap > 0.05:
        status = "[OK] 良好"
    elif overlap > 0.03:
        status = "[WARN] 偏低"
    else:
        status = "[FAIL] 太低（MBAR可能不可靠）"
    print(f"  State {i} ↔ {i+1}: {overlap:.3f} {status}")
```

**健康阈值**：
- Overlap > 0.05：良好
- 0.03 < Overlap < 0.05：勉强可用
- Overlap < 0.03：不可靠，需要重新设计λ梯度

#### 步骤2.3：诊断检查2 - 权重有效性

```python
# 获取State 0的权重（pymbar 4.x API）
# W_nk 是 [N_samples, K_states] 格式
# W_nk[n, k] = 样本n在目标state k下的权重
weights_state0 = mbar.W_nk[:, 0]  # [OK] 取第一列（State 0）

# 计算有效样本数（ESS）
ESS = (weights_state0.sum())**2 / (weights_state0**2).sum()
efficiency = ESS / len(weights_state0)

print("\n" + "="*60)
print("权重统计 (State 0)")
print("="*60)
print(f"  总样本数: {len(weights_state0)}")
print(f"  有效样本数 (ESS): {ESS:.0f}")
print(f"  统计效率: {100*efficiency:.1f}%")

# 分析权重集中度
sorted_weights = np.sort(weights_state0)[::-1]
cumsum = np.cumsum(sorted_weights)
n_50 = np.searchsorted(cumsum, 0.5 * cumsum[-1]) + 1
n_90 = np.searchsorted(cumsum, 0.9 * cumsum[-1]) + 1

print(f"  前{n_50}个样本贡献50%权重 ({100*n_50/len(weights_state0):.1f}%)")
print(f"  前{n_90}个样本贡献90%权重 ({100*n_90/len(weights_state0):.1f}%)")

# 健康判断
if efficiency > 0.1:
    print("  [OK] 权重分布健康")
elif efficiency > 0.05:
    print("  [WARN] 权重略集中，可用但需谨慎")
else:
    print("  [FAIL] 权重严重集中，结果可能不可靠")
```

#### 步骤2.4：诊断检查3 - 能量分布

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for k in range(6):
    ax = axes.flat[k]

    # 该state采样的构象的能量分布
    mask = np.array([f['state'] == k for f in subsampled_frames])
    energies_k = u_kn_mbar[k, mask]

    ax.hist(energies_k, bins=50, alpha=0.7, color=f'C{k}')
    ax.axvline(energies_k.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean={energies_k.mean():.1f}')
    ax.set_xlabel('Reduced Energy (kT)', fontsize=10)
    ax.set_ylabel('Sample Count', fontsize=10)
    ax.set_title(f'State {k}', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mbar_energy_distributions.png', dpi=300)

# 验证能量趋势
mean_energies = [u_kn_mbar[k, :].mean() for k in range(6)]
print("\n各State平均能量:")
for k in range(6):
    print(f"  State {k}: {mean_energies[k]:.2f}")

# REST2预期：λ越小，有效温度越高，平均能量越低（削弱能垒）
print("\n能量趋势检查:")
if mean_energies[0] > mean_energies[-1]:
    print("  [OK] State 0能量高于State 5（符合REST2预期）")
else:
    print("  [WARN] 能量趋势异常")
```

---

### Phase 3: 轨迹分析与重加权

#### 步骤3.1：读取轨迹并计算二面角

```python
import mdtraj as md

# 验证cycle→frame映射
TRAJECTORY_INTERVAL = 20  # 从femto配置读取

print("\n" + "="*60)
print("读取轨迹并计算二面角")
print("="*60)

phi_all = []
psi_all  = []
weights_all = []

for n, frame_info in enumerate(subsampled_frames):
    cycle = frame_info['cycle']
    replica = frame_info['replica']
    weight = weights_state0[n]

    # Cycle到DCD帧号的映射
    frame_idx = cycle // TRAJECTORY_INTERVAL

    # 读取该帧
    traj = md.load_frame(
        f'outputs_v2_gpu/trajectories/r{replica}.dcd',
        index=frame_idx,
        top='system.pdb'
    )

    # 计算丙氨酸二肽的φ/ψ角
    phi_indices, phi_rad = md.compute_phi(traj)
    psi_indices, psi_rad = md.compute_psi(traj)

    # 转换为角度
    phi_deg = np.rad2deg(phi_rad[0, 0])
    psi_deg = np.rad2deg(psi_rad[0, 0])

    phi_all.append(phi_deg)
    psi_all.append(psi_deg)
    weights_all.append(weight)

    if (n+1) % 500 == 0:
        print(f"  处理进度: {n+1}/{len(subsampled_frames)} 帧")

phi_all = np.array(phi_all)
psi_all = np.array(psi_all)
weights_all = np.array(weights_all)

print(f"\n[OK] 完成：{len(phi_all)} 个构象的二面角计算")
```

#### 步骤3.2：生成重加权Ramachandran图

```python
# 重加权直方图
hist_mbar, xedges, yedges = np.histogram2d(
    phi_all, psi_all,
    bins=50,
    range=[[-180, 180], [-180, 180]],
    weights=weights_all,
    density=True
)

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MBAR重加权结果
ax = axes[0]
im = ax.imshow(
    hist_mbar.T,
    origin='lower',
    extent=[-180, 180, -180, 180],
    cmap='Blues',
    aspect='auto'
)
ax.set_xlabel('φ (degrees)', fontsize=12)
ax.set_ylabel('ψ (degrees)', fontsize=12)
ax.set_title('MBAR Reweighted (State 0, 300K)', fontsize=14, fontweight='bold')
ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)

# 标注主要构象区域
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((-110, 60), 60, 40, fill=False,
             edgecolor='darkred', linewidth=2, linestyle='--'))
ax.text(-80, 80, 'C7eq', fontsize=11, color='darkred', weight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.add_patch(Rectangle((50, -100), 50, 60, fill=False,
             edgecolor='darkblue', linewidth=2, linestyle='--'))
ax.text(75, -70, 'C7ax', fontsize=11, color='darkblue', weight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.colorbar(im, ax=ax, label='Probability Density')

# 原始Replica 0对比（未重加权）
ax = axes[1]
traj_r0 = md.load('outputs_v2_gpu/trajectories/r0.dcd', top='system.pdb')
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
    aspect='auto'
)
ax.set_xlabel('φ (degrees)', fontsize=12)
ax.set_ylabel('ψ (degrees)', fontsize=12)
ax.set_title('Original Replica 0 (未重加权)', fontsize=14)
ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)

plt.colorbar(im2, ax=ax, label='Probability Density')

plt.tight_layout()
plt.savefig('ramachandran_mbar_comparison.png', dpi=300)
print("\n[OK] 保存: ramachandran_mbar_comparison.png")
```

#### 步骤3.3：计算构象占比

```python
# 构象区域定义（基于文献）
def classify_conformation(phi, psi):
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

# 分类并重加权统计
conf_counts = {'C7eq': 0, 'C7ax': 0, 'PII': 0, 'alphaR': 0, 'beta': 0, 'other': 0}

for phi, psi, weight in zip(phi_all, psi_all, weights_all):
    conf = classify_conformation(phi, psi)
    conf_counts[conf] += weight

# 归一化
total = sum(conf_counts.values())
conf_fractions = {k: v/total for k, v in conf_counts.items()}

print("\n" + "="*60)
print("构象占比 (MBAR重加权)")
print("="*60)
for conf, frac in conf_fractions.items():
    print(f"  {conf:8s}: {100*frac:6.2f}%")

# 计算自由能差
kT = 0.593  # kcal/mol @ 300K
if conf_fractions['C7eq'] > 0 and conf_fractions['C7ax'] > 0:
    dG = -kT * np.log(conf_fractions['C7ax'] / conf_fractions['C7eq'])
    print(f"\n自由能差:")
    print(f"  ΔG(C7ax - C7eq) = {dG:.2f} kcal/mol")
    print(f"  文献参考值: 0.6-1.2 kcal/mol")
```

---

## **输出结果**

### 文件列表

#### 图表
1. **mbar_energy_distributions.png**
   - 各state采样的能量分布
   - 验证能量趋势合理性

2. **ramachandran_mbar_comparison.png**
   - 左：MBAR重加权后的Ramachandran图
   - 右：原始Replica 0（未重加权）
   - 对比显示重加权的效果

3. **mbar_diagnostics.png**
   - 子图1：State overlap矩阵热图
   - 子图2：权重分布直方图
   - 子图3：有效样本数统计

4. **conformation_populations.png**
   - 柱状图：各构象的占比（MBAR vs Replica 0）

#### 数据文件
1. **mbar_weights.npz**
   ```python
   # 保存内容：
   np.savez('mbar_weights.npz',
            weights=weights_state0,
            phi=phi_all,
            psi=psi_all,
            subsampled_frames=subsampled_frames,  # 元数据
            N_k=N_k,
            u_kn=u_kn_mbar)
   ```

2. **mbar_analysis_report.txt**
   - 子采样统计（每个replica的g值、Neff）
   - MBAR收敛信息
   - State overlap矩阵
   - 权重ESS
   - 构象占比
   - 自由能估算

---

## **成功指标**

### 必须满足的条件

1. **子采样合理性**
   - 每个replica的统计不相关时间 g < 100
   - 总有效样本数 N_eff > 1000
   - 各state至少有50个独立样本

2. **MBAR收敛性**
   - 迭代收敛（< 10000次）
   - 相邻state overlap > 0.03
   - 权重ESS > 10% × 总样本数

3. **物理合理性**
   - C7eq占比 > C7ax（文献共识）
   - ΔG(C7ax-C7eq) ≈ 0.6-1.2 kcal/mol
   - Ramachandran图更平滑、覆盖更全面

### 预期改进

相比只使用Replica 0：
-  **有效样本数增加** 2-5倍（即使子采样后）
-  **构象空间覆盖更全面**（高λ态采样了稀有区域）
-  **自由能精度提高**（更好的统计）
-  **不确定性降低**（更多独立样本）

---

## 🚨 **潜在问题与应对**

### 问题1：MBAR不收敛

**症状**：
```
RuntimeError: MBAR did not converge after 10000 iterations
```

**原因**：State间overlap太小

**解决方案**：
1. 只使用相邻的几个state（如0,1,2,3）
2. 增加初始化迭代次数
3. 使用更保守的初始猜测

```python
# 只用部分states
u_kn_subset = u_kn_mbar[:4, :]  # 只用前4个state
N_k_subset = N_k[:4]
mbar = pymbar.MBAR(u_kn_subset, N_k_subset)
```

### 问题2：权重极度集中

**症状**：
```
ESS = 23 / 5000 (效率0.5%)
前1%样本贡献95%权重
```

**原因**：各state采样区域重叠太少

**解决方案**：
1. 检查λ梯度设计（可能间隔太大）
2. 使用更多cycles增加采样
3. 考虑使用更多副本

### 问题3：能量单位不明确

**症状**：能量值数量级异常

**解决方案**：
```python
# 在MBAR输入前检查并转换
if np.abs(u_kn_mbar).max() > 1e6:
    # 可能是kJ/mol，需要转换为kT
    kT = 8.314 * 300 / 1000  # kJ/mol @ 300K
    u_kn_mbar = u_kn_mbar / kT
```

---

## 📖 **理论背景文献**

### 关键引用

1. **MBAR方法**
   - Shirts & Chodera (2008) J. Chem. Phys. 129, 124105
   - "Statistically optimal analysis of samples from multiple equilibrium states"

2. **REST2方法**
   - Wang, Arrar, et al. (2011) J. Phys. Chem. B 115, 9431
   - "Replica Exchange with Solute Scaling: A More Efficient Version of Replica Exchange with Solute Tempering"

3. **丙氨酸二肽基准**
   - Beauchamp et al. (2012) J. Chem. Theory Comput. 8, 1409
   - "Are Protein Force Fields Getting Better?"

4. **子采样理论**
   - Chodera (2016) J. Chem. Theory Comput. 12, 1799
   - "A Simple Method for Automated Equilibration Detection in Molecular Simulations"

---

## **与现有分析的对比**

### 当前分析（03_analyze_results_v2.py）

**方法**：
- 只分析Replica 0的DCD轨迹（2500帧）
- 假设Replica 0采样的就是State 0的分布
- 没有利用其他副本的数据

**问题**：
- 浪费了5个副本（12500帧）的数据
- 忽略了副本交换带来的增强采样
- Replica 0不一定总在State 0（动态交换）
- 统计精度受限于单个副本

### MBAR方案

**方法**：
- 使用全部6个副本的数据（15000帧）
- 正确处理副本在不同state间的交换
- 通过MBAR严格重加权恢复State 0分布

**优势**：
- 利用高λ态采样的稀有构象（通过重加权映射到State 0）
- 有效样本数增加（即使子采样后）
- 理论上严格（满足统计力学平衡条件）
- 可以计算任意state的期望值

---

## **实现文件说明**

### 主脚本：`04_mbar_reweighting.py`

**结构**：
```python
#!/usr/bin/env python
"""
步骤 4: REST2 HREMD 的 MBAR 重加权分析

基于评审意见的完整修正版本：
1. [OK] 对每个replica的完整轨迹单独子采样
2. [OK] 正确的MBAR权重索引 mbar.W_nk[:, 0] (pymbar 4.x)
3. [OK] N_k在子采样后统计
4. [OK] 验证cycle→frame映射
5. [OK] 完整的Phase 2诊断检查
"""

# Part 1: 数据加载与验证
# Part 2: 逐Replica子采样（核心修正）
# Part 3: 构建MBAR输入
# Part 4: MBAR计算
# Part 5: 诊断检查（overlap, ESS, 能量）
# Part 6: 轨迹分析与重加权
# Part 7: 结果对比与可视化
# Part 8: 保存结果
```


---


## **后续扩展方向**

### Phase 2+ 可选功能

1. **不确定性估计**
   ```python
   # Bootstrap重采样
   for bootstrap in range(100):
       indices = np.random.choice(N_total, N_total, replace=True)
       mbar_boot = pymbar.MBAR(u_kn[:, indices], N_k_boot)
       # 计算置信区间
   ```

2. **其他结构参数重加权**
   - RMSD分布
   - 回旋半径
   - 氢键占据率
   - 端到端距离

3. **温度依赖性分析**
   - 计算所有6个state的自由能面
   - 研究温度对构象平衡的影响

4. **动力学性质**（需要时间加权MBAR）
   - 扩散系数
   - 转移速率常数

---

