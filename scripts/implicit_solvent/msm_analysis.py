#!/usr/bin/env python
"""
MSM 马尔科夫状态模型分析模块

基于 Deeptime 实现自动化亚稳态识别：
- TICA 降维
- k-means 聚类
- MSM 构建与验证
- PCCA+ 亚稳态分解
- 动力学分析

依赖: deeptime, mdtraj, numpy, pandas
"""

import pickle
from dataclasses import dataclass, field
from typing import Optional, List, Any
import numpy as np
import pandas as pd

try:
    from deeptime.decomposition import TICA
    from deeptime.clustering import KMeans
    from deeptime.markov.msm import MaximumLikelihoodMSM
    HAS_DEEPTIME = True
except ImportError:
    HAS_DEEPTIME = False

try:
    import mdtraj as md
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False


@dataclass
class ITSResult:
    """隐含时间尺度分析结果"""
    lags: np.ndarray  # lag times (帧数)
    timescales: np.ndarray  # shape: (n_lags, n_timescales)

    @property
    def n_lags(self):
        return len(self.lags)


@dataclass
class MSMAnalysisResult:
    """MSM 分析结果数据类"""
    # 原始数据
    phi_deg: np.ndarray
    psi_deg: np.ndarray

    # TICA
    tica_model: Any
    tica_output: np.ndarray
    tica_lag: int

    # 聚类
    clustering: Any
    dtrajs: List[np.ndarray]
    n_clusters: int

    # MSM
    msm_model: Any
    msm_lag: int
    its_result: ITSResult  # 隐含时间尺度结果
    cktest: Any = None

    # PCCA+
    n_metastable: int = 4
    pcca_model: Any = None
    pcca_assignments: np.ndarray = None
    metastable_probs: np.ndarray = None

    # 动力学
    kinetics: dict = field(default_factory=dict)
    timestep_ps: float = 20.0

    def save(self, path):
        """保存结果到 pickle 文件"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """从 pickle 文件加载"""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def export_kinetics_csv(self, path):
        """导出动力学量到 CSV"""
        rows = []

        # 基本信息
        rows.append({
            'quantity': 'tica_lag',
            'value': self.tica_lag,
            'unit': 'frames',
        })
        rows.append({
            'quantity': 'msm_lag',
            'value': self.msm_lag,
            'unit': 'frames',
        })
        rows.append({
            'quantity': 'n_clusters',
            'value': self.n_clusters,
            'unit': '',
        })
        rows.append({
            'quantity': 'n_metastable',
            'value': self.n_metastable,
            'unit': '',
        })

        # 隐含时间尺度
        timescales = self.kinetics.get('timescales_ps', [])
        for i, ts in enumerate(timescales[:5]):  # 前 5 个
            rows.append({
                'quantity': f'implied_timescale_{i+1}',
                'value': ts,
                'unit': 'ps',
            })

        # 亚稳态信息
        free_energies = self.kinetics.get('free_energies_kcal', [])
        metastable_probs = self.metastable_probs if self.metastable_probs is not None else []
        free_energies_list = list(free_energies) if len(free_energies) > 0 else [0] * self.n_metastable

        for i, (prob, fe) in enumerate(zip(metastable_probs, free_energies_list)):
            rows.append({
                'quantity': f'metastable_{i}_probability',
                'value': prob,
                'unit': '',
            })
            rows.append({
                'quantity': f'metastable_{i}_free_energy',
                'value': fe,
                'unit': 'kcal/mol',
            })

        # MFPT
        mfpt = self.kinetics.get('mfpt_matrix_ps')
        if mfpt is not None:
            for i in range(len(mfpt)):
                for j in range(len(mfpt)):
                    if i != j:
                        rows.append({
                            'quantity': f'mfpt_{i}_to_{j}',
                            'value': mfpt[i, j],
                            'unit': 'ps',
                        })

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    def get_metastable_labels(self) -> np.ndarray:
        """
        获取每帧的亚稳态标签

        返回: shape (n_frames,) 的整数数组
        """
        dtraj = self.dtrajs[0]
        labels = np.array([
            self.pcca_assignments[s] if s < len(self.pcca_assignments) else -1
            for s in dtraj
        ])
        return labels


class MSMAnalyzer:
    """MSM 分析器 (基于 Deeptime)"""

    def __init__(
        self,
        trajectory_path: str,
        topology_path: str,
        timestep_ps: float = 20.0,
    ):
        if not HAS_DEEPTIME:
            raise ImportError("需要安装 Deeptime: conda install -c conda-forge deeptime")
        if not HAS_MDTRAJ:
            raise ImportError("需要安装 MDTraj: conda install -c conda-forge mdtraj")

        self.trajectory_path = str(trajectory_path)
        self.topology_path = str(topology_path)
        self.timestep_ps = timestep_ps

        # 加载轨迹
        self.traj = md.load(self.trajectory_path, top=self.topology_path)
        print(f"  - 加载轨迹: {self.traj.n_frames} 帧")

        # 原始二面角（用于可视化）
        self.phi_deg = None
        self.psi_deg = None

    def compute_features(self, use_cossin: bool = True) -> np.ndarray:
        """
        计算二面角特征

        参数:
            use_cossin: 使用 sin/cos 编码消除周期性边界

        返回:
            特征数组 shape (n_frames, n_features)
        """
        # 使用 mdtraj 计算二面角
        phi_indices, phi = md.compute_phi(self.traj)
        psi_indices, psi = md.compute_psi(self.traj)

        # 保存原始角度（用于可视化）
        self.phi_deg = np.rad2deg(phi[:, 0])
        self.psi_deg = np.rad2deg(psi[:, 0])

        if use_cossin:
            # sin/cos 编码：2 个角度 x 2 = 4 维特征
            features = np.column_stack([
                np.sin(phi[:, 0]),
                np.cos(phi[:, 0]),
                np.sin(psi[:, 0]),
                np.cos(psi[:, 0]),
            ])
        else:
            features = np.column_stack([phi[:, 0], psi[:, 0]])

        print(f"  - 特征维度: {features.shape[1]}")
        return features

    def run_tica(self, data: np.ndarray, lag: int, dim: int = 2):
        """
        时间独立成分分析 (TICA)

        参数:
            data: 特征数据
            lag: 滞后时间（帧数）
            dim: 输出维度

        返回:
            tica_model: Deeptime TICA 模型
            tica_output: 降维后的数据
        """
        estimator = TICA(lagtime=lag, dim=dim, scaling='kinetic_map')
        model = estimator.fit(data).fetch_model()
        tica_output = model.transform(data)

        print(f"  - TICA lag: {lag} 帧 = {lag * self.timestep_ps:.1f} ps")
        eigenvalues = model.singular_values[:min(dim, len(model.singular_values))]
        print(f"  - TICA 奇异值: {eigenvalues}")

        return model, tica_output

    def run_clustering(self, data: np.ndarray, n_clusters: int):
        """
        k-means 聚类生成微观状态

        参数:
            data: 输入数据（TICA 输出或原始特征）
            n_clusters: 聚类数目

        返回:
            clustering: Deeptime 聚类模型
            dtrajs: 离散轨迹列表
        """
        estimator = KMeans(
            n_clusters=n_clusters,
            init_strategy='kmeans++',
            max_iter=100,
            fixed_seed=42,
        )
        model = estimator.fit(data).fetch_model()
        dtrajs = model.transform(data)

        print(f"  - 聚类数目: {n_clusters}")
        print(f"  - 有效微观状态: {len(np.unique(dtrajs))}")

        return model, [dtrajs]  # 包装成列表以保持兼容

    def select_lag_time(self, dtrajs: list, candidate_lags: list = None) -> tuple:
        """
        通过隐含时间尺度 (ITS) 分析选择 MSM lag time

        返回:
            its_result: ITSResult 对象
            selected_lag: 建议的 lag time
        """
        if candidate_lags is None:
            max_lag = min(500, len(dtrajs[0]) // 5)
            candidate_lags = [1, 2, 5, 10, 20, 50, 100, 200, 500]
            candidate_lags = [l for l in candidate_lags if l < max_lag]

        if len(candidate_lags) < 2:
            candidate_lags = [1, 2, 5, 10]

        # 手动构建 ITS
        timescales_list = []
        valid_lags = []

        for lag in candidate_lags:
            try:
                msm = MaximumLikelihoodMSM(lagtime=lag, reversible=True)
                model = msm.fit_fetch(dtrajs[0])
                ts = model.timescales()[:5]  # 取前 5 个时间尺度
                # 填充到固定长度
                ts_padded = np.full(5, np.nan)
                ts_padded[:len(ts)] = ts
                timescales_list.append(ts_padded)
                valid_lags.append(lag)
            except Exception as e:
                print(f"  - [WARN] lag={lag} 失败: {e}")
                continue

        if len(timescales_list) == 0:
            raise RuntimeError("无法估计任何 lag time 的 MSM")

        timescales_matrix = np.array(timescales_list)
        its_result = ITSResult(
            lags=np.array(valid_lags),
            timescales=timescales_matrix,
        )

        # 启发式选择：找到 ITS 变化 < 20% 的最小 lag
        selected_lag = valid_lags[-1]

        for i in range(1, len(valid_lags)):
            if not np.isnan(timescales_matrix[i, 0]) and not np.isnan(timescales_matrix[i-1, 0]):
                if timescales_matrix[i, 0] > 0:
                    rel_change = np.abs(
                        timescales_matrix[i, 0] - timescales_matrix[i-1, 0]
                    ) / timescales_matrix[i, 0]
                    if rel_change < 0.2:
                        selected_lag = valid_lags[i]
                        break

        print(f"  - ITS 分析完成，建议 lag: {selected_lag} 帧 = {selected_lag * self.timestep_ps:.1f} ps")

        return its_result, selected_lag

    def build_msm(self, dtrajs: list, lag: int, n_metastable: int = 4, validate: bool = True):
        """
        构建马尔科夫状态模型

        参数:
            dtrajs: 离散轨迹列表
            lag: MSM lag time
            n_metastable: 亚稳态数目（用于 CK 测试）
            validate: 是否进行 Chapman-Kolmogorov 测试

        返回:
            msm_model: Deeptime MSM 模型
            cktest: CK 测试结果
        """
        estimator = MaximumLikelihoodMSM(lagtime=lag, reversible=True)
        msm_model = estimator.fit_fetch(dtrajs[0])

        print(f"  - MSM lag: {lag} 帧 = {lag * self.timestep_ps:.1f} ps")
        print(f"  - 活跃状态数: {msm_model.n_states}")

        cktest = None
        if validate:
            print("  - 运行 Chapman-Kolmogorov 测试...")
            try:
                # Deeptime 的 CK 测试需要多个不同 lag 的模型
                test_lags = [lag * i for i in range(1, 6)]  # lag, 2*lag, ..., 5*lag
                models = []
                for test_lag in test_lags:
                    try:
                        test_msm = MaximumLikelihoodMSM(lagtime=test_lag, reversible=True)
                        test_model = test_msm.fit_fetch(dtrajs[0])
                        models.append(test_model)
                    except Exception:
                        break

                if len(models) >= 2:
                    cktest = msm_model.ck_test(models, n_metastable)
            except Exception as e:
                print(f"  - [WARN] CK 测试失败: {e}")

        return msm_model, cktest

    def run_pcca(self, msm_model, n_metastable: int):
        """
        PCCA+ 亚稳态分解

        参数:
            msm_model: MSM 模型
            n_metastable: 亚稳态数目

        返回:
            pcca_model: PCCAModel 对象
            pcca_assignments: 每个微观状态的亚稳态归属
            metastable_probs: 各亚稳态的平稳概率
        """
        # 检查亚稳态数目是否合理
        if n_metastable > msm_model.n_states:
            print(f"  - [WARN] 亚稳态数目 ({n_metastable}) 大于活跃状态数 ({msm_model.n_states})")
            n_metastable = min(n_metastable, msm_model.n_states)

        pcca_model = msm_model.pcca(n_metastable)
        pcca_assignments = pcca_model.assignments
        pi = msm_model.stationary_distribution

        metastable_probs = np.array([
            pi[pcca_assignments == i].sum()
            for i in range(n_metastable)
        ])

        print(f"  - 亚稳态数目: {n_metastable}")
        for i, prob in enumerate(metastable_probs):
            print(f"    State {i}: {prob * 100:.1f}%")

        return pcca_model, pcca_assignments, metastable_probs

    def compute_kinetics(self, msm_model, pcca_model, pcca_assignments) -> dict:
        """
        计算动力学量

        返回包含以下内容的字典:
        - lag_time_ps: MSM lag time (ps)
        - timescales_ps: 隐含时间尺度 (ps)
        - mfpt_matrix_ps: 平均首次通过时间矩阵 (ps)
        - stationary_distribution: 亚稳态平稳分布
        - free_energies_kcal: 相对自由能 (kcal/mol)
        """
        # 隐含时间尺度
        timescales = msm_model.timescales()
        timescales_ps = timescales * self.timestep_ps

        # 自由能
        n_meta = pcca_model.n_metastable
        pi = msm_model.stationary_distribution
        meta_pi = np.array([
            pi[pcca_assignments == i].sum()
            for i in range(n_meta)
        ])

        kT = 0.593  # kcal/mol @ 300K
        # 避免 log(0)
        meta_pi_safe = np.where(meta_pi > 0, meta_pi, 1e-10)
        free_energies = -kT * np.log(meta_pi_safe / meta_pi_safe.max())

        # MFPT 矩阵
        mfpt_matrix = np.zeros((n_meta, n_meta))
        for i in range(n_meta):
            for j in range(n_meta):
                if i != j:
                    source = np.where(pcca_assignments == i)[0]
                    target = np.where(pcca_assignments == j)[0]
                    try:
                        mfpt = msm_model.mfpt(source, target)
                        mfpt_matrix[i, j] = mfpt * self.timestep_ps
                    except Exception:
                        mfpt_matrix[i, j] = np.nan

        kinetics = {
            'lag_time_ps': msm_model.lagtime * self.timestep_ps,
            'timescales_ps': timescales_ps,
            'mfpt_matrix_ps': mfpt_matrix,
            'stationary_distribution': meta_pi,
            'free_energies_kcal': free_energies,
        }

        print(f"  - 前 3 个隐含时间尺度: {timescales_ps[:3]} ps")
        print(f"  - 相对自由能: {free_energies} kcal/mol")

        return kinetics

    def run_full_analysis(
        self,
        tica_lag: int = 50,
        n_clusters: int = 100,
        msm_lag: int = None,
        n_metastable: int = 4,
    ) -> MSMAnalysisResult:
        """
        运行完整的 MSM 分析流程

        参数:
            tica_lag: TICA lag time（帧数）
            n_clusters: k-means 聚类数目
            msm_lag: MSM lag time（帧数），None 表示自动选择
            n_metastable: PCCA+ 亚稳态数目

        返回:
            MSMAnalysisResult 数据对象
        """

        print("\n  [MSM 1/6] 计算二面角特征...")
        data = self.compute_features()

        print("\n  [MSM 2/6] TICA 降维...")
        tica_model, tica_output = self.run_tica(data, lag=tica_lag)

        print("\n  [MSM 3/6] k-means 聚类...")
        clustering, dtrajs = self.run_clustering(tica_output, n_clusters)

        print("\n  [MSM 4/6] 选择 MSM lag time...")
        its_result, auto_lag = self.select_lag_time(dtrajs)
        if msm_lag is None:
            msm_lag = auto_lag

        print("\n  [MSM 5/6] 构建 MSM...")
        msm_model, cktest = self.build_msm(dtrajs, lag=msm_lag, n_metastable=n_metastable)

        print("\n  [MSM 6/6] PCCA+ 亚稳态识别...")
        pcca_model, pcca_assignments, metastable_probs = self.run_pcca(msm_model, n_metastable)

        print("\n  [MSM] 计算动力学量...")
        kinetics = self.compute_kinetics(msm_model, pcca_model, pcca_assignments)

        # 组装结果
        result = MSMAnalysisResult(
            phi_deg=self.phi_deg,
            psi_deg=self.psi_deg,
            tica_model=tica_model,
            tica_output=tica_output,
            tica_lag=tica_lag,
            clustering=clustering,
            dtrajs=dtrajs,
            n_clusters=n_clusters,
            msm_model=msm_model,
            msm_lag=msm_lag,
            its_result=its_result,
            cktest=cktest,
            n_metastable=n_metastable,
            pcca_model=pcca_model,
            pcca_assignments=pcca_assignments,
            metastable_probs=metastable_probs,
            kinetics=kinetics,
            timestep_ps=self.timestep_ps,
        )

        return result
