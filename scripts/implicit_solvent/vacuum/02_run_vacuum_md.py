#!/usr/bin/env python
"""
气相（真空）MD 采样脚本

对丙氨酸二肽（ACE-ALA-NME）进行气相常规分子动力学采样。

采样方法: 常规 MD（Langevin 恒温，NVT 系综）
温度: 300 K
步长: 2 fs
采样时间: 10 ns（可调整）

输入：outputs/implicit_solvent/vacuum/alanine_dipeptide/system.xml
输出：outputs/implicit_solvent/vacuum/alanine_dipeptide/md/
    - trajectory.dcd    DCD 轨迹文件
    - state_data.csv    能量/温度数据
    - checkpoint.chk    检查点文件
"""

import pathlib
import sys
from openmm.app import (
    PDBFile, Simulation, DCDReporter, StateDataReporter,
    CheckpointReporter
)
from openmm import XmlSerializer, LangevinMiddleIntegrator
from openmm.unit import kelvin, picosecond, femtoseconds, nanoseconds

# =============================================================================
# 配置
# =============================================================================
# 路径配置
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SYSTEM_DIR = PROJECT_ROOT / "outputs" / "implicit_solvent" / "vacuum" / "alanine_dipeptide"
OUTPUT_DIR = SYSTEM_DIR / "md"

# 模拟参数
TEMPERATURE = 300 * kelvin           # 模拟温度
FRICTION = 1.0 / picosecond          # Langevin 摩擦系数
TIMESTEP = 2.0 * femtoseconds        # 积分步长

# 采样配置
EQUILIBRATION_STEPS = 50000          # 平衡化步数 (100 ps)
PRODUCTION_STEPS = 5000000           # 生产 MD 步数 (10 ns)
REPORT_INTERVAL = 1000               # 报告间隔 (2 ps)
TRAJECTORY_INTERVAL = 1000           # 轨迹保存间隔 (2 ps)
CHECKPOINT_INTERVAL = 50000          # 检查点间隔 (100 ps)

# 平台选择 (None 表示自动选择)
PLATFORM_NAME = None  # 可选: 'CPU', 'CUDA', 'OpenCL'


# =============================================================================
# 主程序
# =============================================================================
def main():
    print("=" * 60)
    print("气相 MD 采样: 丙氨酸二肽 (ACE-ALA-NME)")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. 加载系统
    # -------------------------------------------------------------------------
    print("\n[1/5] 加载系统...")

    system_xml_path = SYSTEM_DIR / "system.xml"
    system_pdb_path = SYSTEM_DIR / "system.pdb"

    if not system_xml_path.exists():
        print(f"[FAIL] 系统文件不存在: {system_xml_path}")
        print("请先运行 01_prepare_vacuum_system.py")
        sys.exit(1)

    # 加载 OpenMM System
    with open(system_xml_path, 'r') as f:
        system = XmlSerializer.deserialize(f.read())
    print(f"  - 加载: {system_xml_path}")

    # 加载拓扑和坐标
    pdb = PDBFile(str(system_pdb_path))
    topology = pdb.topology
    positions = pdb.positions
    print(f"  - 加载: {system_pdb_path}")
    print(f"  - 原子数: {topology.getNumAtoms()}")

    # -------------------------------------------------------------------------
    # 2. 创建积分器和模拟
    # -------------------------------------------------------------------------
    print("\n[2/5] 创建模拟...")

    # Langevin 积分器（NVT 系综）
    integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP)

    # 创建模拟对象
    if PLATFORM_NAME:
        from openmm import Platform
        platform = Platform.getPlatformByName(PLATFORM_NAME)
        simulation = Simulation(topology, system, integrator, platform)
    else:
        simulation = Simulation(topology, system, integrator)

    # 设置初始坐标
    simulation.context.setPositions(positions)

    # 打印平台信息
    platform = simulation.context.getPlatform()
    print(f"  - 平台: {platform.getName()}")
    print(f"  - 温度: {TEMPERATURE}")
    print(f"  - 步长: {TIMESTEP}")
    print(f"  - 摩擦系数: {FRICTION}")

    # -------------------------------------------------------------------------
    # 3. 能量最小化
    # -------------------------------------------------------------------------
    print("\n[3/5] 能量最小化...")

    # 获取初始能量
    state = simulation.context.getState(getEnergy=True)
    initial_energy = state.getPotentialEnergy()
    print(f"  - 初始势能: {initial_energy}")

    # 最小化
    simulation.minimizeEnergy()

    # 获取最小化后能量
    state = simulation.context.getState(getEnergy=True)
    minimized_energy = state.getPotentialEnergy()
    print(f"  - 最小化后势能: {minimized_energy}")

    # -------------------------------------------------------------------------
    # 4. 平衡化
    # -------------------------------------------------------------------------
    print("\n[4/5] 平衡化...")

    # 初始化速度
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)

    # 升温阶段（从 50K 到 300K）
    n_heating_steps = 25000  # 50 ps
    n_heating_stages = 5
    steps_per_stage = n_heating_steps // n_heating_stages

    print(f"  升温阶段 (50K -> {TEMPERATURE}):")
    for i in range(n_heating_stages):
        # 线性升温
        current_temp = 50 * kelvin + (TEMPERATURE - 50 * kelvin) * (i + 1) / n_heating_stages
        integrator.setTemperature(current_temp)
        simulation.step(steps_per_stage)
        print(f"    阶段 {i+1}/{n_heating_stages}: {current_temp}")

    # NVT 平衡
    print(f"  NVT 平衡 ({EQUILIBRATION_STEPS - n_heating_steps} 步)...")
    integrator.setTemperature(TEMPERATURE)
    simulation.step(EQUILIBRATION_STEPS - n_heating_steps)

    # 获取平衡后状态
    state = simulation.context.getState(getEnergy=True)
    equilibrated_energy = state.getPotentialEnergy()
    print(f"  - 平衡后势能: {equilibrated_energy}")
    print(f"  [OK] 平衡化完成")

    # -------------------------------------------------------------------------
    # 5. 生产 MD
    # -------------------------------------------------------------------------
    print("\n[5/5] 生产 MD...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 添加报告器
    trajectory_path = OUTPUT_DIR / "trajectory.dcd"
    state_data_path = OUTPUT_DIR / "state_data.csv"
    checkpoint_path = OUTPUT_DIR / "checkpoint.chk"

    simulation.reporters.append(
        DCDReporter(str(trajectory_path), TRAJECTORY_INTERVAL)
    )
    simulation.reporters.append(
        StateDataReporter(
            str(state_data_path), REPORT_INTERVAL,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            speed=True
        )
    )
    simulation.reporters.append(
        CheckpointReporter(str(checkpoint_path), CHECKPOINT_INTERVAL)
    )

    # 同时输出到控制台
    simulation.reporters.append(
        StateDataReporter(
            sys.stdout, REPORT_INTERVAL * 10,  # 每 20 ps 输出一次
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            remainingTime=True,
            totalSteps=PRODUCTION_STEPS
        )
    )

    # 计算预计时间
    total_time_ns = PRODUCTION_STEPS * TIMESTEP.value_in_unit(nanoseconds)
    n_frames = PRODUCTION_STEPS // TRAJECTORY_INTERVAL

    print(f"  - 生产步数: {PRODUCTION_STEPS:,} 步")
    print(f"  - 模拟时间: {total_time_ns:.1f} ns")
    print(f"  - 轨迹帧数: {n_frames:,} 帧")
    print(f"  - 输出文件:")
    print(f"    - 轨迹: {trajectory_path}")
    print(f"    - 状态数据: {state_data_path}")
    print(f"    - 检查点: {checkpoint_path}")
    print()

    # 运行生产 MD
    simulation.step(PRODUCTION_STEPS)

    # -------------------------------------------------------------------------
    # 完成
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[OK] 气相 MD 采样完成")
    print("=" * 60)
    print(f"\n下一步: 运行 03_analyze_vacuum_results.py 分析结果")


if __name__ == "__main__":
    main()
