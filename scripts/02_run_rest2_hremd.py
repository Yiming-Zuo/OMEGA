#!/usr/bin/env python
"""
步骤 2 改进版: 运行 REST2 HREMD 模拟

改进:
1. 使用相邻态交换 (swap_mode='neighbors')
2. 重新设计温度梯度（目标接受率 25-35%）
3. 增加采样时间（500 ps → 10 ns）
4. 扩大温度范围（300-600K）
5. 增加副本数（6 → 8）
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
print("Step 2 改进版: REST2 HREMD 优化配置")
print("="*60)

# 1. 加载系统
print("\n[1/4] 加载系统...")
system_xml = pathlib.Path('system.xml').read_text()
system = openmm.XmlSerializer.deserialize(system_xml)
topology = pickle.loads(pathlib.Path('topology.pkl').read_bytes())
print(f"✓ 系统加载: {len(topology.atoms)} 原子")

# =====================================================================
# 第 0 步：系统平衡化
# =====================================================================
print("\n" + "="*60)
print("第 0 步：系统平衡化")
print("="*60)

print("\n设置平衡化协议...")
equilibration_stages = [
    femto.md.config.Minimization(
        tolerance=10.0 * openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom,
        max_iterations=0
    ),
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
print("  1. 最小化")
print("  2. NVT 升温（50K → 300K, 25 ps）")
print("  3. NPT 平衡（300K, 1 bar, 100 ps）")

print("\n开始平衡化...")
equilibrated_coords = femto.md.simulate.simulate_state(
    system,
    topology,
    state={femto.md.rest.REST_CTX_PARAM: 1.0},
    stages=equilibration_stages,
    platform='CPU'
)
print("[OK] 平衡化完成！")

# =====================================================================
# 第 1 步：优化温度梯度
# =====================================================================
print("\n" + "="*60)
print("第 1 步：优化温度梯度设计")
print("="*60)

T_min = 300.0 * openmm.unit.kelvin
T_max = 600.0 * openmm.unit.kelvin  # 扩大到600K
n_replicas = 8  # 增加副本数

# 几何分布：T_i = T_min * (T_max/T_min)^(i/(n-1))
# 对于丙氨酸二肽，几何分布通常比指数分布更优
temperatures = [
    T_min * (T_max / T_min) ** (i / (n_replicas - 1))
    for i in range(n_replicas)
]

print(f"\n[OK] 新温度梯度 ({n_replicas} 副本，几何分布):")
for i, T in enumerate(temperatures):
    print(f"  State {i}: {T.value_in_unit(openmm.unit.kelvin):.1f} K")

# 预测相邻态接受率（粗略估计）
print(f"\n预测相邻态接受率:")
betas = [1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * T) for T in temperatures]
for i in range(n_replicas - 1):
    # 简化公式：P ≈ exp(-Δβ * ⟨E⟩)
    # 假设 ⟨E⟩ ≈ -10000 kT @ 300K (粗略)
    delta_beta = (betas[i+1] - betas[i]).value_in_unit(openmm.unit.mole / openmm.unit.kilojoule)
    # 这只是示意，真实值需要运行后调整
    print(f"  State {i} ↔ {i+1}: ΔT = {(temperatures[i+1] - temperatures[i]).value_in_unit(openmm.unit.kelvin):.1f} K")

states = [{femto.md.rest.REST_CTX_PARAM: beta / betas[0]} for beta in betas]
states = [femto.md.utils.openmm.evaluate_ctx_parameters(s, system) for s in states]

print(f"\nBeta 缩放因子 (β_m/β_0):")
for i, s in enumerate(states):
    print(f"  State {i}: {s[femto.md.rest.REST_CTX_PARAM]:.4f}")

# =====================================================================
# 第 2 步：创建模拟对象
# =====================================================================
print("\n" + "="*60)
print("第 2 步：创建模拟对象")
print("="*60)

integrator_config = femto.md.config.LangevinIntegrator(
    timestep=2.0 * openmm.unit.femtosecond,
    friction=1.0 / openmm.unit.picosecond
)
integrator = femto.md.utils.openmm.create_integrator(integrator_config, T_min)

simulation = femto.md.utils.openmm.create_simulation(
    system,
    topology,
    equilibrated_coords,
    integrator=integrator,
    state=states[0],
    platform='CPU'
)

print("✓ 模拟对象创建完成")

# =====================================================================
# 第 3 步：配置 HREMD（优化参数）
# =====================================================================
print("\n" + "="*60)
print("第 3 步：优化 HREMD 配置")
print("="*60)

hremd_config = femto.md.config.HREMD(
    temperature=T_min,
    n_warmup_steps=5000,          # 10 ps warmup
    n_steps_per_cycle=500,        # 1 ps per cycle（增加到1ps）
    n_cycles=2000,               # 10000 cycles = 10 ns 采样
    swap_mode='neighbours',       # 改为相邻态交换！（注意英式拼写）
    max_swaps=None,
    trajectory_interval=20,       # 每 20 cycles = 20 ps 保存一次
    checkpoint_interval=100,
    trajectory_enforce_pbc=True
)

warmup_time_ps = hremd_config.n_warmup_steps * 2 / 1000
sampling_time_ps = hremd_config.n_cycles * hremd_config.n_steps_per_cycle * 2 / 1000
total_time_ps = warmup_time_ps + sampling_time_ps

print(f"[OK] 优化后的 HREMD 配置:")
print(f"  - Warmup: {warmup_time_ps:.1f} ps")
print(f"  - 每轮步数: {hremd_config.n_steps_per_cycle} 步 = {hremd_config.n_steps_per_cycle * 2 / 1000:.2f} ps")
print(f"  - 总轮数: {hremd_config.n_cycles}")
print(f"  - 采样时间: {sampling_time_ps:.1f} ps = {sampling_time_ps/1000:.1f} ns")
print(f"  - 总模拟时间: {total_time_ps:.1f} ps = {total_time_ps/1000:.1f} ns")
print(f"  - 交换模式: neighbours (相邻态，{n_replicas-1} 对)")
print(f"  - 轨迹保存: 每 {hremd_config.trajectory_interval} 轮 = {hremd_config.trajectory_interval * hremd_config.n_steps_per_cycle * 2 / 1000:.1f} ps")

print(f"\n[WARN] 预计运行时间（CPU）: ~2-4 小时")
print(f"   （如有GPU可改为 platform='CUDA'，速度提升10-20倍）")

# =====================================================================
# 第 4 步：运行 REST2 HREMD
# =====================================================================
print("\n" + "="*60)
print("第 4 步：运行优化后的 REST2 HREMD")
print("="*60)

output_dir = pathlib.Path('outputs_v2')
output_dir.mkdir(exist_ok=True)
print(f"✓ 输出目录: {output_dir.absolute()}")

print("\n开始 HREMD 模拟...")
print("（10 ns采样，需要较长时间，建议挂后台运行）")
print("")

try:
    femto.md.hremd.run_hremd(
        simulation,
        states,
        hremd_config,
        output_dir=output_dir
    )

    print("\n" + "="*60)
    print("[OK] HREMD 完成！")
    print("="*60)
    print(f"输出文件:")
    print(f"  - {output_dir}/samples.arrow")
    print(f"  - {output_dir}/trajectories/r*.dcd")
    print(f"  - {output_dir}/checkpoint.pkl")
    print("="*60)
    print("\n下一步: 运行 python 03_analyze_results_v2.py")

except Exception as e:
    print("\n" + "="*60)
    print("[FAIL] HREMD 运行失败")
    print("="*60)
    print(f"错误信息: {e}")
    import traceback
    traceback.print_exc()
