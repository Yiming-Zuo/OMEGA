#!/usr/bin/env python
"""
为 REST2 HREMD 预优化温度梯度。

该脚本读取目标温度范围与副本数，调用 femto.md.hremd.optimize_ladder()
（如果可用）生成在目标接受率范围内的 β / 温度 / REST 缩放因子。

用法示例:
    python 00_optimize_ladder.py --t-min 300 --t-max 500 --replicas 6

生成的梯度会写入 optimized_ladder.json，可直接在 02_run_rest2_hremd.py 中使用。
"""

from __future__ import annotations

import argparse
import inspect
import json
import pathlib
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import openmm.unit


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def _ensure_femto_modules():
    try:
        from femto.md import hremd  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "无法导入 femto.md.hremd，请确认已安装 femto 并满足它的依赖 "
            "(例如 mdtraj)。可以在 femto/ 目录下执行 `pip install -e .[md]`。"
        ) from exc


def _fetch_optimize_ladder():
    from femto.md import hremd  # noqa: F401

    optimize = getattr(hremd, "optimize_ladder", None)
    if optimize is None:
        raise SystemExit(
            "当前 femto 版本尚未包含 femto.md.hremd.optimize_ladder。\n"
            "请更新到包含该功能的版本（>=0.3.1 及以上），或按照文档手动构造温度梯度。"
        )
    return optimize


def _to_kelvin(values: Any) -> np.ndarray:
    """将温度序列转换为 numpy ndarray，单位 K。"""
    if isinstance(values, openmm.unit.Quantity):
        return values.value_in_unit(openmm.unit.kelvin)
    arr = np.asarray(values, dtype=float)
    return arr


def _to_beta(values: Any) -> np.ndarray:
    """将 β 序列转换为 numpy ndarray，单位 1/(kJ/mol)。"""
    unit = 1.0 / openmm.unit.kilojoules_per_mole
    if isinstance(values, openmm.unit.Quantity):
        return values.value_in_unit(unit)
    arr = np.asarray(values, dtype=float)
    return arr


def _as_jsonable(data: Any) -> Any:
    if is_dataclass(data):
        return _as_jsonable(asdict(data))
    if isinstance(data, dict):
        return {k: _as_jsonable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_as_jsonable(v) for v in data]
    if isinstance(data, openmm.unit.Quantity):
        return {
            "value": data.value_in_unit(data.unit),
            "unit": str(data.unit),
        }
    if isinstance(data, (np.ndarray, np.generic)):
        return np.asarray(data).tolist()
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 femto.md.hremd.optimize_ladder 预优化 REST2 温度梯度"
    )
    parser.add_argument("--t-min", type=float, default=300.0, help="最低温度 (K)")
    parser.add_argument("--t-max", type=float, default=500.0, help="最高温度 (K)")
    parser.add_argument("--replicas", type=int, default=6, help="副本数量")
    parser.add_argument(
        "--target-min",
        type=float,
        default=0.15,
        help="目标接受率下限（0-1 之间）",
    )
    parser.add_argument(
        "--target-max",
        type=float,
        default=0.35,
        help="目标接受率上限（0-1 之间）",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=100,
        help="内部迭代上限（传递给 optimize_ladder）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机数种子（可选）",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=SCRIPT_DIR / "optimized_ladder.json",
        help="保存输出的 JSON 文件路径",
    )
    return parser.parse_args()


def call_optimize_ladder(optimize, args: argparse.Namespace):
    """根据 optimize_ladder 的签名动态构造参数。"""
    sig = inspect.signature(optimize)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    beta_cold = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * (args.t_min * openmm.unit.kelvin))
    beta_hot = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * (args.t_max * openmm.unit.kelvin))

    def _set(names, value) -> bool:
        for name in names:
            if name in params:
                kwargs[name] = value
                return True
        return False

    if not _set(("n_states", "n_replicas", "replicas"), args.replicas):
        raise SystemExit("optimize_ladder 的签名中找不到副本数量参数（n_states / n_replicas）。")

    _set(("beta_min", "min_beta"), beta_hot)
    _set(("beta_max", "max_beta"), beta_cold)
    _set(("target_acceptance", "target"), (args.target_min, args.target_max))
    _set(("max_iterations", "max_iters"), args.max_iters)

    if args.seed is not None:
        _set(("seed", "random_seed"), args.seed)

    # REST2 缩放函数有些版本需要显式传入
    if _set(("scaling", "scale_fn", "rest_scaling"), "rest2"):
        pass

    print("调用 femto.md.hremd.optimize_ladder 参数:")
    for key, val in kwargs.items():
        print(f"  {key}: {val}")

    ladder = optimize(**kwargs)
    return ladder


def extract_results(ladder) -> dict[str, Any]:
    """从 optimize_ladder 返回值中提取常用信息。"""
    result: dict[str, Any] = {}

    for attr in ("temperatures", "temperature"):
        if hasattr(ladder, attr):
            result["temperatures"] = _to_kelvin(getattr(ladder, attr))
            break
    else:
        result["temperatures"] = None

    for attr in ("betas", "beta_values", "beta"):
        if hasattr(ladder, attr):
            result["betas"] = _to_beta(getattr(ladder, attr))
            break
    else:
        result["betas"] = None

    for attr in ("ctx_params", "states"):
        if hasattr(ladder, attr):
            result["states"] = getattr(ladder, attr)
            break
    else:
        result["states"] = None

    if result["temperatures"] is None and result["betas"] is not None:
        beta_vals = np.asarray(result["betas"], dtype=float)
        beta_quantity = beta_vals * (1.0 / openmm.unit.kilojoules_per_mole)
        temps = 1.0 / (beta_quantity * openmm.unit.MOLAR_GAS_CONSTANT_R)
        result["temperatures"] = temps.value_in_unit(openmm.unit.kelvin)

    if result["betas"] is None and result["temperatures"] is not None:
        temp_vals = np.asarray(result["temperatures"], dtype=float)
        temp_quantity = temp_vals * openmm.unit.kelvin
        betas = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temp_quantity)
        result["betas"] = betas.value_in_unit(1.0 / openmm.unit.kilojoules_per_mole)

    # 确保 numpy->list，Quantity->dict
    result["temperatures"] = _as_jsonable(result["temperatures"])
    result["betas"] = _as_jsonable(result["betas"])
    result["states"] = _as_jsonable(result["states"])
    result["raw"] = _as_jsonable(ladder)

    return result


def pretty_print(result: dict[str, Any]):
    print("\n建议的温度梯度 (K):")
    temps = result["temperatures"]
    if temps is None:
        print("  (optimize_ladder 未返回温度信息)")
        return
    if isinstance(temps, dict):  # Quantity path
        values = temps["value"]
    else:
        values = temps

    for i, t in enumerate(values):
        print(f"  State {i}: {t:.2f} K")

    if result["betas"]:
        print("\nβ (1/(kJ/mol)):")
        betas = result["betas"]
        if isinstance(betas, dict):
            betas = betas["value"]
        for i, b in enumerate(betas):
            print(f"  State {i}: {b:.6f}")


def main():
    _ensure_femto_modules()
    args = parse_args()
    optimize = _fetch_optimize_ladder()
    ladder = call_optimize_ladder(optimize, args)
    result = extract_results(ladder)
    pretty_print(result)

    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n已将详细结果写入: {args.output}\n")
    print(
        "下一步：\n"
        "1. 在 02_run_rest2_hremd.py 中替换温度/β 配置为上面的输出。\n"
        "2. 将 HREMD 配置中的 n_cycles 暂时调小（例如 50），跑一次短试算。\n"
        "3. 使用 03_analyze_results.py 检查接受率是否落在目标区间。\n"
        "4. 如有偏差，可调整 target-min / target-max 重新优化。"
    )


if __name__ == "__main__":
    main()
