#!/usr/bin/env python
"""
步骤 2: 运行 REST2 HREMD 模拟

任务:
1. 加载溶剂化后的系统
2. 设置温度梯度（300K - 500K，6个副本）
3. 先进行平衡化（最小化 + 升温 + NPT 平衡）
4. 运行 REST2 HREMD（500 cycles 测试）
5. 保存采样数据和轨迹
"""

import pickle
import pathlib
import openmm
import openmm.app
import openmm.unit
import numpy as np
import mdtop
import femto.md.config
import femto.md.hremd
import femto.md.rest
import femto.md.utils.openmm
import femto.md.simulate

print("="*60)
print("Step 2: 运行 REST2 HREMD 模拟")
print("="*60)

# 1. 加载溶剂化系统（使用 mdtop.Topology）
print("\n[1/4] 加载系统...")
system_xml = pathlib.Path('system.xml').read_text()
system = openmm.XmlSerializer.deserialize(system_xml)

# 加载 mdtop.Topology（而不是 openmm.app.PDBFile）
topology = pickle.loads(pathlib.Path('topology.pkl').read_bytes())
print(f"✓ 系统加载: {len(topology.atoms)} 原子")
print(f"  - 类型: mdtop.Topology（支持 .select() 方法）")

# 检查是否有 femto 模块
try:
    import femto
    print(f"✓ femto 版本: {femto.__version__}")
except:
    print("✓ femto 已加载（无版本信息）")

# =====================================================================
# 第 0 步：系统平衡化（重要！）
# =====================================================================
print("\n" + "="*60)
print("第 0 步：系统平衡化")
print("="*60)

print("\n设置平衡化协议...")
equilibration_stages = [
    # 最小化
    femto.md.config.Minimization(
        tolerance=10.0 * openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom,
        max_iterations=0  # 直到收敛
    ),
    # NVT 升温（50K -> 300K）
    femto.md.config.Anneal(
        integrator=femto.md.config.LangevinIntegrator(
            timestep=1.0 * openmm.unit.femtosecond,
            friction=1.0 / openmm.unit.picosecond
        ),
        temperature_initial=50.0 * openmm.unit.kelvin,
        temperature_final=300.0 * openmm.unit.kelvin,
        n_steps=25000,  # 25 ps
        frequency=100
    ),
    # NPT 平衡（300K, 1 bar）
    femto.md.config.Simulation(
        integrator=femto.md.config.LangevinIntegrator(
            timestep=2.0 * openmm.unit.femtosecond,
            friction=1.0 / openmm.unit.picosecond
        ),
        temperature=300.0 * openmm.unit.kelvin,
        pressure=1.0 * openmm.unit.bar,
        n_steps=50000  # 100 ps
    )
]

print("✓ 平衡化阶段:")
print("  1. 最小化（能量极小化）")
print("  2. NVT 升温（50K → 300K, 25 ps）")
print("  3. NPT 平衡（300K, 1 bar, 100 ps）")

print("\n开始平衡化（总计 ~125 ps）...")
print("（这可能需要几分钟，请耐心等待...）\n")

# 运行平衡化（REST 参数设为 1.0 = 无缩放）
equilibrated_coords = femto.md.simulate.simulate_state(
    system,
    topology,  # mdtop.Topology
    state={femto.md.rest.REST_CTX_PARAM: 1.0},
    stages=equilibration_stages,
    platform='CPU'  # 改为 'CUDA' 如果有 GPU
)

print("\n✅ 平衡化完成！")

# =====================================================================
# 第 1 步：设置 REST2 温度梯度
# =====================================================================
print("\n" + "="*60)
print("第 1 步：设置 REST2 温度梯度")
print("="*60)

T_min = 300.0 * openmm.unit.kelvin
T_max = 500.0 * openmm.unit.kelvin
n_replicas = 6  # 6 个副本（显式溶剂用较少副本）

# 指数分布温度梯度
temperatures = [
    T_min + (T_max - T_min) * (np.exp(i/(n_replicas-1)) - 1) / (np.e - 1)
    for i in range(n_replicas)
]

print(f"\n温度梯度 ({n_replicas} 个副本):")
for i, T in enumerate(temperatures):
    print(f"  State {i}: {T.value_in_unit(openmm.unit.kelvin):.1f} K")

# 计算 beta_m / beta_0
betas = [1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * T) for T in temperatures]
states = [{femto.md.rest.REST_CTX_PARAM: beta / betas[0]} for beta in betas]
states = [femto.md.utils.openmm.evaluate_ctx_parameters(s, system) for s in states]

print(f"\nBeta 缩放因子 (β_m/β_0):")
for i, s in enumerate(states):
    print(f"  State {i}: {s[femto.md.rest.REST_CTX_PARAM]:.4f}")

# =====================================================================
# 第 2 步：创建模拟对象
# =====================================================================
print("\n" + "="*60)
print("第 2 步：创建 OpenMM 模拟对象")
print("="*60)

integrator_config = femto.md.config.LangevinIntegrator(
    timestep=2.0 * openmm.unit.femtosecond,
    friction=1.0 / openmm.unit.picosecond
)
integrator = femto.md.utils.openmm.create_integrator(integrator_config, T_min)

simulation = femto.md.utils.openmm.create_simulation(
    system,
    topology,  # mdtop.Topology
    equilibrated_coords,  # 使用平衡化后的坐标！
    integrator=integrator,
    state=states[0],
    platform='CPU'  # 改为 'CUDA' 如果有 GPU
)

print("✓ 模拟对象创建完成")
print(f"  - 积分器: Langevin")
print(f"  - 时间步长: 2.0 fs")
print(f"  - 摩擦系数: 1.0 ps⁻¹")
print(f"  - 平台: CPU")

# =====================================================================
# 第 3 步：配置 HREMD
# =====================================================================
print("\n" + "="*60)
print("第 3 步：配置 HREMD 参数")
print("="*60)

hremd_config = femto.md.config.HREMD(
    temperature=T_min,
    n_warmup_steps=5000,         # 10 ps warmup
    n_steps_per_cycle=250,       # 0.5 ps per cycle（显式溶剂）
    n_cycles=500,                # 500 cycles = 250 ps 采样
    swap_mode='all',             # 全对交换
    max_swaps=None,              # 不限制交换次数
    trajectory_interval=10,      # 每 10 cycles 保存
    checkpoint_interval=50,      # 每 50 cycles 保存检查点
    trajectory_enforce_pbc=True  # 显式溶剂应用 PBC
)

warmup_time_ps = hremd_config.n_warmup_steps * 2 / 1000
sampling_time_ps = hremd_config.n_cycles * hremd_config.n_steps_per_cycle * 2 / 1000
total_time_ps = warmup_time_ps + sampling_time_ps

print(f"✓ HREMD 配置:")
print(f"  - Warmup: {hremd_config.n_warmup_steps} 步 = {warmup_time_ps:.1f} ps")
print(f"  - 每轮步数: {hremd_config.n_steps_per_cycle} 步 = {hremd_config.n_steps_per_cycle * 2 / 1000:.2f} ps")
print(f"  - 总轮数: {hremd_config.n_cycles}")
print(f"  - 采样时间: {sampling_time_ps:.1f} ps")
print(f"  - 总模拟时间: {total_time_ps:.1f} ps")
print(f"  - 交换模式: all (全对，{n_replicas*(n_replicas-1)//2} 对)")
print(f"  - 轨迹保存: 每 {hremd_config.trajectory_interval} 轮")

# =====================================================================
# 第 4 步：运行 REST2 HREMD
# =====================================================================
print("\n" + "="*60)
print("第 4 步：运行 REST2 HREMD")
print("="*60)

output_dir = pathlib.Path('outputs')
output_dir.mkdir(exist_ok=True)
print(f"✓ 输出目录: {output_dir.absolute()}")

print("\n开始 HREMD 模拟...")
print("（这可能需要 10-20 分钟，请耐心等待...）")
print("\n提示:")
print("  - 你会看到进度条显示采样进度")
print("  - 如果遇到 NaN，femto 会自动重试（REST2 特性）")
print("  - 轨迹文件会保存到 outputs/trajectories/")
print("")

try:
    femto.md.hremd.run_hremd(
        simulation,
        states,
        hremd_config,
        output_dir=output_dir
    )

    print("\n" + "="*60)
    print("✅ HREMD 完成！")
    print("="*60)
    print(f"输出文件:")
    print(f"  - {output_dir}/samples.arrow (采样数据)")
    print(f"  - {output_dir}/trajectories/r*.dcd (轨迹)")
    print(f"  - {output_dir}/checkpoint.pkl (检查点)")
    print("="*60)
    print("\n下一步: 运行 python 03_analyze_results.py")

except Exception as e:
    print("\n" + "="*60)
    print("❌ HREMD 运行失败")
    print("="*60)
    print(f"错误信息: {e}")
    import traceback
    traceback.print_exc()
    print("\n可能的原因:")
    print("  1. conda 环境未激活 (需要 conda activate fm)")
    print("  2. 缺少依赖包 (需要 openmm, femto, mpi4py 等)")
    print("  3. 系统文件损坏")
    print("\n请检查后重试。")
