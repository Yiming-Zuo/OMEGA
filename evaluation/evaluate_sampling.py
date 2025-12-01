#!/usr/bin/env python
"""
通用小分子溶剂化增强采样评估脚本

评估目标：为概率流自由能差估计方法评估端点玻尔兹曼分布的采样质量

核心评估维度：
1. 平衡收敛性（最关键）
2. 构象空间覆盖度
3. Replica Exchange效率
4. 统计质量（ESS）
"""

import argparse
import pathlib
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import jensenshannon

# 可选依赖
try:
    import mdtraj as md
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False
    warnings.warn("mdtraj 未安装，部分功能不可用")

try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    warnings.warn("pyarrow 未安装，无法读取samples.arrow")

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn 未安装，聚类和PCA功能不可用")


@dataclass
class EvaluationMetrics:
    """评估指标汇总"""
    # 收敛性指标
    js_divergence: Optional[float] = None  # 块间JS散度
    half_similarity: Optional[float] = None  # 前后半分布相似度
    convergence_reached: Optional[bool] = None  # 是否达到平台期

    # 统计质量
    n_frames: int = 0
    n_eff_autocorr: Optional[float] = None  # 基于自相关的有效样本数
    autocorr_time: Optional[float] = None  # 积分自相关时间

    # 构象覆盖度
    n_clusters: Optional[int] = None  # 构象聚类数
    n_transitions: Optional[int] = None  # 状态间转变次数
    dihedral_coverage: dict = field(default_factory=dict)  # 各二面角覆盖情况

    # Exchange效率
    acceptance_rate: Optional[float] = None
    n_roundtrips: Optional[int] = None

    def summary(self) -> str:
        """生成评估报告摘要"""
        lines = [
            "=" * 60,
            "采样质量评估报告",
            "=" * 60,
            "",
            "一、收敛性评估",
            "-" * 40,
        ]

        if self.js_divergence is not None:
            status = "✅ 通过" if self.js_divergence < 0.1 else "⚠️ 需关注"
            lines.append(f"  块间JS散度: {self.js_divergence:.4f} (判据 < 0.1) {status}")

        if self.half_similarity is not None:
            status = "✅ 通过" if self.half_similarity > 0.9 else "⚠️ 需关注"
            lines.append(f"  前后半分布相似度: {self.half_similarity:.4f} (判据 > 0.9) {status}")

        if self.convergence_reached is not None:
            status = "✅ 已收敛" if self.convergence_reached else "⚠️ 可能未收敛"
            lines.append(f"  累积指标收敛: {status}")

        lines.extend([
            "",
            "二、统计质量",
            "-" * 40,
            f"  总帧数: {self.n_frames}",
        ])

        if self.autocorr_time is not None:
            lines.append(f"  积分自相关时间: {self.autocorr_time:.1f} 帧")

        if self.n_eff_autocorr is not None:
            ratio = self.n_eff_autocorr / self.n_frames * 100 if self.n_frames > 0 else 0
            status = "✅ 充足" if self.n_eff_autocorr > 100 else "⚠️ 不足"
            lines.append(f"  有效样本数: {self.n_eff_autocorr:.0f} ({ratio:.1f}%) {status}")

        lines.extend([
            "",
            "三、构象空间覆盖",
            "-" * 40,
        ])

        if self.n_clusters is not None:
            lines.append(f"  发现构象聚类数: {self.n_clusters}")

        if self.n_transitions is not None:
            status = "✅ 充足" if self.n_transitions >= 10 else "⚠️ 较少"
            lines.append(f"  状态转变次数: {self.n_transitions} {status}")

        if self.dihedral_coverage:
            lines.append(f"  可旋转二面角: {len(self.dihedral_coverage)} 个")

        lines.extend([
            "",
            "四、Exchange效率",
            "-" * 40,
        ])

        if self.acceptance_rate is not None:
            status = "✅ 理想" if 0.2 <= self.acceptance_rate <= 0.4 else "⚠️ 需调整"
            lines.append(f"  交换接受率: {self.acceptance_rate*100:.1f}% (理想20-40%) {status}")

        if self.n_roundtrips is not None:
            status = "✅ 充足" if self.n_roundtrips >= 3 else "⚠️ 不足"
            lines.append(f"  Round-trip次数: {self.n_roundtrips} (判据 ≥ 3) {status}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)


class SamplingEvaluator:
    """通用小分子采样评估器"""

    def __init__(self,
                 traj_dir: pathlib.Path,
                 topology_file: pathlib.Path,
                 samples_file: Optional[pathlib.Path] = None,
                 output_dir: Optional[pathlib.Path] = None,
                 ligand_selection: str = "not water"):
        """
        参数:
            traj_dir: 轨迹文件目录
            topology_file: 拓扑文件(PDB)
            samples_file: samples.arrow文件（可选，用于exchange分析）
            output_dir: 输出目录
            ligand_selection: MDTraj选择语法，用于选择小分子
        """
        self.traj_dir = pathlib.Path(traj_dir)
        self.topology_file = pathlib.Path(topology_file)
        self.samples_file = pathlib.Path(samples_file) if samples_file else None
        self.output_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ligand_selection = ligand_selection

        self.metrics = EvaluationMetrics()
        self.traj = None
        self.ligand_traj = None
        self.dihedrals = None
        self.dihedral_indices = None

    def load_trajectory(self, replica_idx: int = 0) -> bool:
        """加载指定replica的轨迹"""
        if not HAS_MDTRAJ:
            print("❌ 需要安装mdtraj")
            return False

        traj_file = self.traj_dir / f"r{replica_idx}.dcd"
        if not traj_file.exists():
            print(f"❌ 轨迹文件不存在: {traj_file}")
            return False

        print(f"加载轨迹: {traj_file}")
        self.traj = md.load(str(traj_file), top=str(self.topology_file))
        self.metrics.n_frames = len(self.traj)
        print(f"  总帧数: {self.metrics.n_frames}")

        # 提取小分子轨迹
        ligand_atoms = self.traj.topology.select(self.ligand_selection)
        if len(ligand_atoms) == 0:
            print(f"⚠️ 选择 '{self.ligand_selection}' 未找到原子")
            self.ligand_traj = self.traj
        else:
            self.ligand_traj = self.traj.atom_slice(ligand_atoms)
            print(f"  小分子原子数: {len(ligand_atoms)}")

        return True

    def detect_rotatable_dihedrals(self) -> list:
        """自动检测可旋转二面角"""
        if self.ligand_traj is None:
            return []

        # 对于通用分子，尝试获取所有非环内的连续4原子
        # 这里使用MDTraj的内置函数
        topology = self.ligand_traj.topology

        # 获取重原子（非氢）
        heavy_atoms = [a.index for a in topology.atoms if a.element.symbol != 'H']

        # 简化方法：对于小分子，直接使用phi/psi（如果是蛋白/肽）
        # 或者使用chi角
        dihedral_indices = []

        # 尝试计算phi角
        try:
            phi_indices, _ = md.compute_phi(self.ligand_traj)
            dihedral_indices.extend(phi_indices.tolist())
        except Exception:
            pass

        # 尝试计算psi角
        try:
            psi_indices, _ = md.compute_psi(self.ligand_traj)
            dihedral_indices.extend(psi_indices.tolist())
        except Exception:
            pass

        # 尝试计算chi角
        try:
            chi_indices, _ = md.compute_chi1(self.ligand_traj)
            dihedral_indices.extend(chi_indices.tolist())
        except Exception:
            pass

        # 如果以上都失败，尝试手动构建
        if not dihedral_indices:
            # 从bonds构建连续的4原子序列
            bonds = [(b[0].index, b[1].index) for b in topology.bonds]
            # 简单方法：找重原子链
            if len(heavy_atoms) >= 4:
                # 使用前4个重原子作为示例
                dihedral_indices = [heavy_atoms[:4]]

        self.dihedral_indices = np.array(dihedral_indices) if dihedral_indices else None

        if self.dihedral_indices is not None and len(self.dihedral_indices) > 0:
            print(f"  检测到 {len(self.dihedral_indices)} 个可旋转二面角")
        else:
            print("  ⚠️ 未检测到可旋转二面角")

        return dihedral_indices

    def compute_dihedrals(self) -> np.ndarray:
        """计算所有检测到的二面角"""
        if self.ligand_traj is None or self.dihedral_indices is None:
            return None

        if len(self.dihedral_indices) == 0:
            return None

        # 使用完整轨迹计算二面角（需要正确的原子索引）
        self.dihedrals = md.compute_dihedrals(self.traj, self.dihedral_indices)
        self.dihedrals = np.rad2deg(self.dihedrals)  # 转换为度

        return self.dihedrals

    def analyze_convergence(self, observable: np.ndarray, name: str = "observable") -> dict:
        """
        分析收敛性

        参数:
            observable: 时间序列数据 (n_frames,) 或 (n_frames, n_features)
            name: 用于标签

        返回:
            包含收敛性指标的字典
        """
        if observable is None or len(observable) == 0:
            return {}

        # 确保是2D
        if observable.ndim == 1:
            observable = observable.reshape(-1, 1)

        n_frames = len(observable)
        results = {}

        # 1. 块平均分析 - 分4块
        n_blocks = 4
        block_size = n_frames // n_blocks

        if block_size > 10:
            # 对第一个特征计算直方图分布
            first_feature = observable[:, 0]

            # 获取全局范围
            global_min, global_max = first_feature.min(), first_feature.max()
            bins = np.linspace(global_min, global_max, 50)

            block_hists = []
            for i in range(n_blocks):
                start = i * block_size
                end = (i + 1) * block_size
                hist, _ = np.histogram(first_feature[start:end], bins=bins, density=True)
                hist = hist + 1e-10  # 避免零值
                hist = hist / hist.sum()  # 归一化
                block_hists.append(hist)

            # 计算相邻块之间的JS散度
            js_divs = []
            for i in range(n_blocks - 1):
                js = jensenshannon(block_hists[i], block_hists[i+1])
                js_divs.append(js)

            results['js_divergence'] = np.mean(js_divs)
            self.metrics.js_divergence = results['js_divergence']

            # 前后半分布相似度
            half = n_frames // 2
            hist_first, _ = np.histogram(first_feature[:half], bins=bins, density=True)
            hist_second, _ = np.histogram(first_feature[half:], bins=bins, density=True)
            hist_first = hist_first + 1e-10
            hist_second = hist_second + 1e-10
            hist_first = hist_first / hist_first.sum()
            hist_second = hist_second / hist_second.sum()

            # 相似度 = 1 - JS散度
            js_half = jensenshannon(hist_first, hist_second)
            results['half_similarity'] = 1 - js_half
            self.metrics.half_similarity = results['half_similarity']

        # 2. 自相关时间分析
        autocorr_time = self._compute_autocorr_time(observable[:, 0])
        results['autocorr_time'] = autocorr_time
        self.metrics.autocorr_time = autocorr_time

        if autocorr_time > 0:
            n_eff = n_frames / (2 * autocorr_time)
            results['n_eff'] = n_eff
            self.metrics.n_eff_autocorr = n_eff

        # 3. 累积均值收敛检验
        cumulative_mean = np.cumsum(observable[:, 0]) / np.arange(1, n_frames + 1)

        # 检查后25%是否稳定（标准差小于均值的5%）
        last_quarter = cumulative_mean[int(0.75 * n_frames):]
        if len(last_quarter) > 10:
            std_last = np.std(last_quarter)
            mean_last = np.mean(last_quarter)
            if mean_last != 0:
                relative_std = std_last / abs(mean_last)
                results['convergence_reached'] = relative_std < 0.05
                self.metrics.convergence_reached = results['convergence_reached']

        return results

    def _compute_autocorr_time(self, x: np.ndarray, max_lag: int = None) -> float:
        """计算积分自相关时间"""
        n = len(x)
        if max_lag is None:
            max_lag = min(n // 4, 1000)

        x = x - np.mean(x)
        var = np.var(x)

        if var == 0:
            return 1.0

        # 计算自相关函数
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[n-1:n-1+max_lag+1]
        autocorr = autocorr / (var * n)

        # 积分自相关时间 = 0.5 + sum(autocorr[1:])
        # 使用首次穿越零点截断
        zero_crossings = np.where(autocorr < 0)[0]
        if len(zero_crossings) > 0:
            cutoff = zero_crossings[0]
        else:
            cutoff = max_lag

        tau_int = 0.5 + np.sum(autocorr[1:cutoff])

        return max(tau_int, 1.0)

    def analyze_conformational_space(self) -> dict:
        """分析构象空间覆盖"""
        results = {}

        if self.dihedrals is None or len(self.dihedrals) == 0:
            return results

        # 1. 二面角分布分析
        for i in range(self.dihedrals.shape[1]):
            angles = self.dihedrals[:, i]

            # 检测峰的数量
            hist, bin_edges = np.histogram(angles, bins=72, range=(-180, 180))

            # 简单峰检测：高于平均值1.5倍的bin
            threshold = np.mean(hist) * 1.5
            peaks = np.sum(hist > threshold)

            self.metrics.dihedral_coverage[f"dihedral_{i}"] = {
                'n_peaks': peaks,
                'mean': np.mean(angles),
                'std': np.std(angles),
                'range': (angles.min(), angles.max())
            }

        results['n_dihedrals'] = self.dihedrals.shape[1]

        # 2. 构象聚类（如果有sklearn）
        if HAS_SKLEARN and self.ligand_traj is not None:
            # 使用RMSD进行聚类
            # 首先计算RMSD矩阵（采样以减少计算量）
            n_samples = min(1000, len(self.ligand_traj))
            indices = np.linspace(0, len(self.ligand_traj)-1, n_samples, dtype=int)

            # 使用二面角作为特征进行聚类
            if self.dihedrals is not None:
                features = self.dihedrals[indices]

                # 使用层次聚类
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=30,  # 30度阈值
                    linkage='average'
                )
                labels = clustering.fit_predict(features)

                n_clusters = len(np.unique(labels))
                results['n_clusters'] = n_clusters
                self.metrics.n_clusters = n_clusters

        # 3. 状态转变检测
        if self.dihedrals is not None and self.dihedrals.shape[1] > 0:
            # 基于主要二面角的状态转变
            main_dihedral = self.dihedrals[:, 0]

            # 定义状态：将角度分成3个区域
            states = np.digitize(main_dihedral, bins=[-60, 60])

            # 计算转变次数
            transitions = np.sum(states[:-1] != states[1:])
            results['n_transitions'] = transitions
            self.metrics.n_transitions = transitions

        return results

    def analyze_exchange_efficiency(self) -> dict:
        """分析Replica Exchange效率"""
        results = {}

        if self.samples_file is None or not self.samples_file.exists():
            return results

        if not HAS_PYARROW:
            return results

        print(f"加载采样数据: {self.samples_file}")

        with pyarrow.OSFile(str(self.samples_file), 'rb') as f:
            reader = pyarrow.RecordBatchStreamReader(f)
            table = reader.read_all()
            df = table.to_pandas()

        # 1. 交换接受率
        if 'n_proposed_swaps' in df.columns and 'n_accepted_swaps' in df.columns:
            n_prop = df['n_proposed_swaps'].iloc[-1]
            n_acc = df['n_accepted_swaps'].iloc[-1]

            # 处理各种数据格式
            n_prop_arr = np.asarray(n_prop)
            n_acc_arr = np.asarray(n_acc)

            # 检查是否是嵌套数组
            if n_prop_arr.dtype == object:
                # 全对交换矩阵格式
                try:
                    prop_matrix = np.array([np.array(row) for row in n_prop_arr])
                    acc_matrix = np.array([np.array(row) for row in n_acc_arr])
                    total_prop = np.sum(prop_matrix)
                    total_acc = np.sum(acc_matrix)
                except Exception:
                    total_prop = 0
                    total_acc = 0
            elif n_prop_arr.ndim >= 1:
                total_prop = np.sum(n_prop_arr)
                total_acc = np.sum(n_acc_arr)
            else:
                total_prop = float(n_prop_arr)
                total_acc = float(n_acc_arr)

            if total_prop > 0:
                acceptance_rate = total_acc / total_prop
                results['acceptance_rate'] = acceptance_rate
                self.metrics.acceptance_rate = acceptance_rate

        # 2. Replica游走分析
        if 'replica_to_state_idx' in df.columns:
            replica_states = np.array([np.array(x) for x in df['replica_to_state_idx']])
            n_replicas = replica_states.shape[1]

            # 计算round-trip
            # 对于replica 0，统计从state 0到state n-1再回到state 0的次数
            replica_0_states = replica_states[:, 0]

            n_roundtrips = 0
            at_bottom = True  # 开始在底部
            reached_top = False

            for state in replica_0_states:
                if at_bottom and state == n_replicas - 1:
                    reached_top = True
                    at_bottom = False
                elif reached_top and state == 0:
                    n_roundtrips += 1
                    at_bottom = True
                    reached_top = False

            results['n_roundtrips'] = n_roundtrips
            self.metrics.n_roundtrips = n_roundtrips
            results['replica_states'] = replica_states

        return results

    def generate_plots(self):
        """生成所有评估图表"""
        print("\n生成评估图表...")

        # 1. 收敛性诊断图
        if self.dihedrals is not None and len(self.dihedrals) > 0:
            self._plot_convergence()

        # 2. 构象空间图
        if self.dihedrals is not None and len(self.dihedrals) > 0:
            self._plot_conformational_space()

        # 3. Exchange效率图（如果有数据）
        if self.metrics.acceptance_rate is not None:
            self._plot_exchange_efficiency()

    def _plot_convergence(self):
        """绘制收敛性诊断图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        main_dihedral = self.dihedrals[:, 0]
        n_frames = len(main_dihedral)

        # 1. 时间序列
        axes[0, 0].plot(main_dihedral, alpha=0.7, linewidth=0.5)
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Dihedral Angle (deg)')
        axes[0, 0].set_title('Dihedral Angle Time Series')
        axes[0, 0].grid(alpha=0.3)

        # 2. 累积均值
        cumulative_mean = np.cumsum(main_dihedral) / np.arange(1, n_frames + 1)
        axes[0, 1].plot(cumulative_mean)
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Cumulative Mean (deg)')
        axes[0, 1].set_title('Cumulative Mean Convergence')
        axes[0, 1].grid(alpha=0.3)

        # 3. 块平均分布对比
        half = n_frames // 2
        axes[1, 0].hist(main_dihedral[:half], bins=72, range=(-180, 180),
                        alpha=0.5, density=True, label='First Half')
        axes[1, 0].hist(main_dihedral[half:], bins=72, range=(-180, 180),
                        alpha=0.5, density=True, label='Second Half')
        axes[1, 0].set_xlabel('Dihedral Angle (deg)')
        axes[1, 0].set_ylabel('Probability Density')
        axes[1, 0].set_title('Distribution Comparison (First vs Second Half)')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 4. 自相关函数
        max_lag = min(len(main_dihedral) // 4, 500)
        x = main_dihedral - np.mean(main_dihedral)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[len(x)-1:len(x)-1+max_lag+1]
        autocorr = autocorr / autocorr[0]

        axes[1, 1].plot(autocorr)
        axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(0.05, color='r', linestyle='--', alpha=0.5, label='5% threshold')
        axes[1, 1].set_xlabel('Lag (frames)')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].set_title('Autocorrelation Function')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'convergence_analysis.png'
        plt.savefig(output_path, dpi=300)
        print(f"  保存: {output_path}")
        plt.close()

    def _plot_conformational_space(self):
        """绘制构象空间图"""
        n_dihedrals = self.dihedrals.shape[1]

        # 如果只有1-2个二面角
        if n_dihedrals == 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(self.dihedrals[:, 0], bins=72, range=(-180, 180),
                    density=True, alpha=0.7)
            ax.set_xlabel('Dihedral Angle (deg)')
            ax.set_ylabel('Probability Density')
            ax.set_title('Dihedral Angle Distribution')
            ax.grid(alpha=0.3)

        elif n_dihedrals >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # 1. 2D联合分布（Ramachandran-like）
            h = axes[0].hist2d(self.dihedrals[:, 0], self.dihedrals[:, 1],
                              bins=50, range=[[-180, 180], [-180, 180]],
                              cmap='Blues', density=True)
            axes[0].set_xlabel('Dihedral 1 (deg)')
            axes[0].set_ylabel('Dihedral 2 (deg)')
            axes[0].set_title('Joint Distribution')
            plt.colorbar(h[3], ax=axes[0], label='Probability Density')

            # 2. 各二面角分布
            for i in range(min(n_dihedrals, 4)):
                axes[1].hist(self.dihedrals[:, i], bins=72, range=(-180, 180),
                            density=True, alpha=0.5, label=f'Dihedral {i+1}')
            axes[1].set_xlabel('Dihedral Angle (deg)')
            axes[1].set_ylabel('Probability Density')
            axes[1].set_title('Individual Dihedral Distributions')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'conformational_space.png'
        plt.savefig(output_path, dpi=300)
        print(f"  保存: {output_path}")
        plt.close()

        # PCA投影（如果有sklearn）
        if HAS_SKLEARN and n_dihedrals >= 2:
            fig, ax = plt.subplots(figsize=(8, 8))

            pca = PCA(n_components=2)
            projected = pca.fit_transform(self.dihedrals)

            ax.scatter(projected[:, 0], projected[:, 1], c=np.arange(len(projected)),
                      cmap='viridis', alpha=0.5, s=1)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_title('PCA Projection of Conformational Space')

            plt.tight_layout()
            output_path = self.output_dir / 'pca_projection.png'
            plt.savefig(output_path, dpi=300)
            print(f"  保存: {output_path}")
            plt.close()

    def _plot_exchange_efficiency(self):
        """绘制Exchange效率图"""
        # 这里需要从analyze_exchange_efficiency获取replica_states
        # 简化版本：只显示接受率

        fig, ax = plt.subplots(figsize=(8, 6))

        if self.metrics.acceptance_rate is not None:
            ax.bar(['Acceptance Rate'], [self.metrics.acceptance_rate * 100])
            ax.axhline(20, color='g', linestyle='--', label='Ideal range (20-40%)')
            ax.axhline(40, color='g', linestyle='--')
            ax.set_ylabel('Rate (%)')
            ax.set_title('Exchange Acceptance Rate')
            ax.legend()
            ax.set_ylim([0, 100])

        plt.tight_layout()
        output_path = self.output_dir / 'exchange_efficiency.png'
        plt.savefig(output_path, dpi=300)
        print(f"  保存: {output_path}")
        plt.close()

    def run_full_evaluation(self, replica_idx: int = 0) -> EvaluationMetrics:
        """运行完整评估流程"""
        print("=" * 60)
        print("小分子溶剂化增强采样质量评估")
        print("=" * 60)

        # Step 1: 加载轨迹
        print("\n[1/6] 加载轨迹...")
        if not self.load_trajectory(replica_idx):
            return self.metrics

        # Step 2: 检测可旋转二面角
        print("\n[2/6] 检测可旋转二面角...")
        self.detect_rotatable_dihedrals()
        self.compute_dihedrals()

        # Step 3: 收敛性分析
        print("\n[3/6] 收敛性分析...")
        if self.dihedrals is not None:
            convergence_results = self.analyze_convergence(self.dihedrals, "dihedrals")
            print(f"  JS散度: {self.metrics.js_divergence:.4f}" if self.metrics.js_divergence else "")
            print(f"  前后半相似度: {self.metrics.half_similarity:.4f}" if self.metrics.half_similarity else "")
            print(f"  自相关时间: {self.metrics.autocorr_time:.1f} 帧" if self.metrics.autocorr_time else "")
            print(f"  有效样本数: {self.metrics.n_eff_autocorr:.0f}" if self.metrics.n_eff_autocorr else "")

        # Step 4: 构象空间分析
        print("\n[4/6] 构象空间分析...")
        conf_results = self.analyze_conformational_space()
        if self.metrics.n_clusters:
            print(f"  聚类数: {self.metrics.n_clusters}")
        if self.metrics.n_transitions:
            print(f"  状态转变次数: {self.metrics.n_transitions}")

        # Step 5: Exchange效率分析
        print("\n[5/6] Exchange效率分析...")
        exchange_results = self.analyze_exchange_efficiency()
        if self.metrics.acceptance_rate:
            print(f"  交换接受率: {self.metrics.acceptance_rate*100:.1f}%")
        if self.metrics.n_roundtrips is not None:
            print(f"  Round-trip次数: {self.metrics.n_roundtrips}")

        # Step 6: 生成图表
        print("\n[6/6] 生成评估图表...")
        self.generate_plots()

        # 输出报告
        print("\n" + self.metrics.summary())

        # 保存报告
        report_path = self.output_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(self.metrics.summary())
        print(f"\n报告已保存: {report_path}")

        return self.metrics


def main():
    parser = argparse.ArgumentParser(description='评估小分子溶剂化增强采样质量')
    parser.add_argument('--traj-dir', type=str, required=True,
                        help='轨迹文件目录（包含r0.dcd, r1.dcd等）')
    parser.add_argument('--topology', type=str, required=True,
                        help='拓扑文件（PDB格式）')
    parser.add_argument('--samples', type=str, default=None,
                        help='samples.arrow文件路径（可选）')
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='输出目录')
    parser.add_argument('--replica', type=int, default=0,
                        help='要分析的replica索引（默认0，即最低温度）')
    parser.add_argument('--selection', type=str, default='not water',
                        help='小分子选择语法（MDTraj格式）')

    args = parser.parse_args()

    evaluator = SamplingEvaluator(
        traj_dir=args.traj_dir,
        topology_file=args.topology,
        samples_file=args.samples,
        output_dir=args.output,
        ligand_selection=args.selection
    )

    evaluator.run_full_evaluation(replica_idx=args.replica)


if __name__ == '__main__':
    main()
