#!/usr/bin/env python
"""
隐式溶剂 MD 采样脚本（通用版）

支持气相（vacuum）和隐式溶剂（GBSA）两种模式。

使用方式:
    python 02_run_md.py --solvent vacuum
    python 02_run_md.py --solvent gbsa

输入: 由配置文件指定的 system.xml
输出: trajectory.dcd, state_data.csv, checkpoint.chk
"""

import argparse
import pathlib
import sys
import yaml
from openmm.app import (
    PDBFile, Simulation, DCDReporter, StateDataReporter,
    CheckpointReporter
)
from openmm import XmlSerializer, LangevinMiddleIntegrator
from openmm.unit import kelvin, picosecond, femtoseconds, nanoseconds
from tqdm import tqdm


def load_config(solvent: str) -> dict:
    """加载溶剂配置文件"""
    config_path = pathlib.Path(__file__).parent / f"{solvent}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_with_progress(simulation, total_steps, desc, chunk_size=1000):
    """带进度条的模拟运行

    Args:
        simulation: OpenMM Simulation 对象
        total_steps: 总步数
        desc: 进度条描述
        chunk_size: 每次运行的步数（用于更新进度条）
    """
    with tqdm(total=total_steps, desc=desc, unit="step",
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        remaining = total_steps
        while remaining > 0:
            steps = min(chunk_size, remaining)
            simulation.step(steps)
            pbar.update(steps)
            remaining -= steps


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='隐式溶剂 MD 采样')
    parser.add_argument(
        '--solvent',
        choices=['vacuum', 'gbsa'],
        required=True,
        help='溶剂模型: vacuum (气相) 或 gbsa (隐式溶剂)'
    )
    parser.add_argument(
        '--platform',
        choices=['CPU', 'CUDA', 'OpenCL'],
        default=None,
        help='计算平台 (默认自动选择)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.solvent)

    solvent_config = config['solvent']
    sim_config = config['simulation']
    output_config = config['output']

    # 项目根目录
    project_root = pathlib.Path(__file__).resolve().parents[2]

    # 模拟参数
    temperature = sim_config['temperature'] * kelvin
    friction = sim_config['friction'] / picosecond
    timestep = sim_config['timestep'] * femtoseconds
    equilibration_steps = sim_config['equilibration_steps']
    production_steps = sim_config['production_steps']
    report_interval = sim_config['report_interval']
    trajectory_interval = sim_config['trajectory_interval']
    checkpoint_interval = sim_config['checkpoint_interval']

    print("=" * 60)
    print(f"MD 采样: {solvent_config['name'].upper()} 模式")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. 加载系统
    # -------------------------------------------------------------------------
    print("\n[1/5] 加载系统...")

    system_dir = project_root / output_config['system_dir']
    system_xml_path = system_dir / "system.xml"
    system_pdb_path = system_dir / "system.pdb"

    if not system_xml_path.exists():
        print(f"[FAIL] 系统文件不存在: {system_xml_path}")
        print(f"请先运行: python 01_prepare_system.py --solvent {args.solvent}")
        sys.exit(1)

    with open(system_xml_path, 'r') as f:
        system = XmlSerializer.deserialize(f.read())
    print(f"  - 加载: {system_xml_path}")

    pdb = PDBFile(str(system_pdb_path))
    topology = pdb.topology
    positions = pdb.positions
    print(f"  - 加载: {system_pdb_path}")
    print(f"  - 原子数: {topology.getNumAtoms()}")

    # -------------------------------------------------------------------------
    # 2. 创建积分器和模拟
    # -------------------------------------------------------------------------
    print("\n[2/5] 创建模拟...")

    integrator = LangevinMiddleIntegrator(temperature, friction, timestep)

    if args.platform:
        from openmm import Platform
        platform = Platform.getPlatformByName(args.platform)
        simulation = Simulation(topology, system, integrator, platform)
    else:
        simulation = Simulation(topology, system, integrator)

    simulation.context.setPositions(positions)

    platform = simulation.context.getPlatform()
    print(f"  - 平台: {platform.getName()}")
    print(f"  - 温度: {temperature}")
    print(f"  - 步长: {timestep}")
    print(f"  - 摩擦系数: {friction}")

    # -------------------------------------------------------------------------
    # 3. 能量最小化
    # -------------------------------------------------------------------------
    print("\n[3/5] 能量最小化...")

    state = simulation.context.getState(getEnergy=True)
    initial_energy = state.getPotentialEnergy()
    print(f"  - 初始势能: {initial_energy}")

    simulation.minimizeEnergy()

    state = simulation.context.getState(getEnergy=True)
    minimized_energy = state.getPotentialEnergy()
    print(f"  - 最小化后势能: {minimized_energy}")

    # -------------------------------------------------------------------------
    # 4. 平衡化
    # -------------------------------------------------------------------------
    print("\n[4/5] 平衡化...")

    simulation.context.setVelocitiesToTemperature(temperature)

    # 升温阶段（从 50K 到目标温度）
    n_heating_steps = min(25000, equilibration_steps // 2)
    n_heating_stages = 5
    steps_per_stage = n_heating_steps // n_heating_stages

    print(f"  升温阶段 (50K -> {temperature}):")
    with tqdm(range(n_heating_stages), desc="  升温",
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]') as pbar:
        for i in pbar:
            current_temp = 50 * kelvin + (temperature - 50 * kelvin) * (i + 1) / n_heating_stages
            integrator.setTemperature(current_temp)
            simulation.step(steps_per_stage)
            pbar.set_postfix_str(f"{current_temp}")

    # NVT 平衡
    remaining_steps = equilibration_steps - n_heating_steps
    print(f"  NVT 平衡 ({remaining_steps} 步):")
    integrator.setTemperature(temperature)
    run_with_progress(simulation, remaining_steps, "  NVT平衡")

    state = simulation.context.getState(getEnergy=True)
    equilibrated_energy = state.getPotentialEnergy()
    print(f"  - 平衡后势能: {equilibrated_energy}")
    print(f"  [OK] 平衡化完成")

    # -------------------------------------------------------------------------
    # 5. 生产 MD
    # -------------------------------------------------------------------------
    print("\n[5/5] 生产 MD...")

    md_output_dir = system_dir / "md"
    md_output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = md_output_dir / "trajectory.dcd"
    state_data_path = md_output_dir / "state_data.csv"
    checkpoint_path = md_output_dir / "checkpoint.chk"

    simulation.reporters.append(
        DCDReporter(str(trajectory_path), trajectory_interval)
    )
    simulation.reporters.append(
        StateDataReporter(
            str(state_data_path), report_interval,
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
        CheckpointReporter(str(checkpoint_path), checkpoint_interval)
    )

    total_time_ns = production_steps * timestep.value_in_unit(nanoseconds)
    n_frames = production_steps // trajectory_interval

    print(f"  - 生产步数: {production_steps:,} 步")
    print(f"  - 模拟时间: {total_time_ns:.1f} ns")
    print(f"  - 轨迹帧数: {n_frames:,} 帧")
    print(f"  - 输出文件:")
    print(f"    - 轨迹: {trajectory_path}")
    print(f"    - 状态数据: {state_data_path}")
    print(f"    - 检查点: {checkpoint_path}")
    print()

    run_with_progress(simulation, production_steps, "  生产MD", chunk_size=5000)

    # -------------------------------------------------------------------------
    # 完成
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"[OK] {solvent_config['name'].upper()} MD 采样完成")
    print("=" * 60)
    print(f"\n下一步: python 03_analyze_results.py --solvent {args.solvent}")


if __name__ == "__main__":
    main()
