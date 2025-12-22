#!/usr/bin/env python
"""
MSM 可视化模块

生成 MSM 分析结果的各种图表：
- TICA 投影图
- 隐含时间尺度图
- Chapman-Kolmogorov 测试图
- 自由能面
- 亚稳态分布图
- 转换网络图

依赖: matplotlib, numpy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MSMVisualizer:
    """MSM 结果可视化器"""

    def __init__(self, result, output_dir, solvent_name: str = "", temperature: int = 300):
        """
        初始化可视化器

        参数:
            result: MSMAnalysisResult 对象
            output_dir: 输出目录 (pathlib.Path)
            solvent_name: 溶剂名称（用于图标题）
            temperature: 温度（用于图标题）
        """
        self.result = result
        self.output_dir = output_dir
        self.solvent_name = solvent_name.upper() if solvent_name else ""
        self.temperature = temperature

    def plot_all(self):
        """生成所有图表"""
        self.plot_tica_projection()
        self.plot_implied_timescales()
        self.plot_cktest()
        self.plot_free_energy_surface()
        self.plot_metastable_states()
        self.plot_transition_network()
        self.plot_metastable_timeline()

    def _get_title_suffix(self) -> str:
        """获取图标题后缀"""
        parts = []
        if self.solvent_name:
            parts.append(self.solvent_name)
        if self.temperature:
            parts.append(f"{self.temperature}K")
        return f" ({', '.join(parts)})" if parts else ""

    def plot_tica_projection(self):
        """TICA 投影散点图（按亚稳态着色）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 获取每帧的亚稳态归属
        meta_labels = self.result.get_metastable_labels()

        # 左图：按亚稳态着色
        scatter = axes[0].scatter(
            self.result.tica_output[:, 0],
            self.result.tica_output[:, 1],
            c=meta_labels, cmap='Set1', s=5, alpha=0.3
        )
        axes[0].set_xlabel('TICA 1', fontsize=12)
        axes[0].set_ylabel('TICA 2', fontsize=12)
        axes[0].set_title(f'TICA Projection (lag={self.result.tica_lag} frames){self._get_title_suffix()}')
        plt.colorbar(scatter, ax=axes[0], label='Metastable State')

        # 右图：按时间着色
        time_ns = np.arange(len(self.result.tica_output)) * self.result.timestep_ps / 1000
        scatter2 = axes[1].scatter(
            self.result.tica_output[:, 0],
            self.result.tica_output[:, 1],
            c=time_ns, cmap='viridis', s=5, alpha=0.3
        )
        axes[1].set_xlabel('TICA 1', fontsize=12)
        axes[1].set_ylabel('TICA 2', fontsize=12)
        axes[1].set_title('TICA Projection (colored by time)')
        plt.colorbar(scatter2, ax=axes[1], label='Time (ns)')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'tica_projection.png', dpi=300)
        plt.close()
        print(f"    [OK] tica_projection.png")

    def plot_implied_timescales(self):
        """隐含时间尺度 vs lag time 图"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 使用 ITSResult 数据结构
        its = self.result.its_result
        lags = its.lags
        timescales = its.timescales

        # 绘制每个时间尺度
        colors = plt.cm.tab10(np.linspace(0, 1, timescales.shape[1]))
        for i in range(timescales.shape[1]):
            valid_mask = ~np.isnan(timescales[:, i])
            if valid_mask.any():
                ax.plot(
                    lags[valid_mask],
                    timescales[valid_mask, i] * self.result.timestep_ps,
                    'o-',
                    color=colors[i],
                    label=f'ITS {i+1}',
                    markersize=6
                )

        ax.set_xlabel('Lag time (frames)', fontsize=12)
        ax.set_ylabel('Implied Timescale (ps)', fontsize=12)

        # 标记选定的 lag time
        ax.axvline(self.result.msm_lag, color='red', linestyle='--', linewidth=2,
                   label=f'Selected lag = {self.result.msm_lag}')

        ax.legend(loc='best')
        ax.set_title(f'Implied Timescales{self._get_title_suffix()}')
        ax.grid(alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'implied_timescales.png', dpi=300)
        plt.close()
        print(f"    [OK] implied_timescales.png")

    def plot_cktest(self):
        """Chapman-Kolmogorov 测试图"""
        if self.result.cktest is None:
            print("    [SKIP] cktest.png (CK 测试未运行)")
            return

        try:
            # Deeptime 的 ChapmanKolmogorovTest 绘制
            cktest = self.result.cktest

            # 获取数据维度
            # predictions 和 estimates 形状: (n_lags, n_sets, n_sets)
            predictions = cktest.predictions
            estimates = cktest.estimates
            lagtimes = cktest.lagtimes

            n_sets = predictions.shape[1]  # 从数据形状推断

            fig, axes = plt.subplots(n_sets, n_sets, figsize=(3*n_sets, 3*n_sets))
            if n_sets == 1:
                axes = np.array([[axes]])
            elif n_sets > 1 and axes.ndim == 1:
                axes = axes.reshape(n_sets, n_sets)

            for i in range(n_sets):
                for j in range(n_sets):
                    ax = axes[i, j] if n_sets > 1 else axes[0, 0]

                    # 预测值（从 MSM 外推）
                    ax.plot(lagtimes, predictions[:, i, j], 'b-', linewidth=2, label='MSM prediction')

                    # 实际估计值
                    ax.plot(lagtimes, estimates[:, i, j], 'ro', markersize=6, label='Direct estimate')

                    ax.set_xlabel('Lag time (frames)')
                    ax.set_ylabel(f'P({i}->{j})')
                    ax.grid(alpha=0.3)

                    if i == 0 and j == n_sets - 1:
                        ax.legend(loc='best', fontsize=8)

            fig.suptitle(f'Chapman-Kolmogorov Test{self._get_title_suffix()}', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'cktest.png', dpi=300)
            plt.close()
            print(f"    [OK] cktest.png")

        except Exception as e:
            print(f"    [WARN] cktest.png 绘制失败: {e}")

    def plot_free_energy_surface(self):
        """自由能面（phi/psi 和 TICA 空间）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        kT = 0.593  # kcal/mol @ 300K

        # 左图：phi/psi 空间
        hist, xedges, yedges = np.histogram2d(
            self.result.phi_deg, self.result.psi_deg, bins=60, density=True
        )
        hist = np.where(hist > 0, -kT * np.log(hist), np.nan)
        hist -= np.nanmin(hist)

        im = axes[0].imshow(
            hist.T, origin='lower', extent=[-180, 180, -180, 180],
            cmap='viridis_r', aspect='auto', vmin=0, vmax=3
        )
        axes[0].set_xlabel('phi (degrees)', fontsize=12)
        axes[0].set_ylabel('psi (degrees)', fontsize=12)
        axes[0].set_title(f'Free Energy Surface (phi/psi){self._get_title_suffix()}')
        axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')
        axes[0].axvline(0, color='gray', linewidth=0.5, linestyle='--')
        plt.colorbar(im, ax=axes[0], label='Free Energy (kcal/mol)')

        # 右图：TICA 空间
        hist2, xe2, ye2 = np.histogram2d(
            self.result.tica_output[:, 0],
            self.result.tica_output[:, 1],
            bins=60, density=True
        )
        hist2 = np.where(hist2 > 0, -kT * np.log(hist2), np.nan)
        hist2 -= np.nanmin(hist2)

        im2 = axes[1].imshow(
            hist2.T, origin='lower',
            extent=[xe2[0], xe2[-1], ye2[0], ye2[-1]],
            cmap='viridis_r', aspect='auto', vmin=0, vmax=3
        )
        axes[1].set_xlabel('TICA 1', fontsize=12)
        axes[1].set_ylabel('TICA 2', fontsize=12)
        axes[1].set_title('Free Energy Surface (TICA)')
        plt.colorbar(im2, ax=axes[1], label='Free Energy (kcal/mol)')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'free_energy_surface.png', dpi=300)
        plt.close()
        print(f"    [OK] free_energy_surface.png")

    def plot_metastable_states(self):
        """各亚稳态在 Ramachandran 图中的分布"""
        n_meta = self.result.n_metastable
        ncols = min(n_meta, 4)
        nrows = (n_meta + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))

        if n_meta == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)

        meta_labels = self.result.get_metastable_labels()

        for i in range(n_meta):
            row, col = i // ncols, i % ncols
            ax = axes[row, col]

            mask = meta_labels == i

            if mask.sum() > 0:
                ax.hist2d(
                    self.result.phi_deg[mask],
                    self.result.psi_deg[mask],
                    bins=40, cmap='Blues', density=True
                )

            ax.set_xlim([-180, 180])
            ax.set_ylim([-180, 180])
            ax.set_xlabel('phi (degrees)')
            ax.set_ylabel('psi (degrees)')
            ax.set_title(f'Metastable {i} ({self.result.metastable_probs[i]*100:.1f}%)')
            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

        # 隐藏多余的子图
        for i in range(n_meta, nrows * ncols):
            row, col = i // ncols, i % ncols
            axes[row, col].axis('off')

        plt.suptitle(f'Metastable State Distributions{self._get_title_suffix()}', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metastable_states.png', dpi=300)
        plt.close()
        print(f"    [OK] metastable_states.png")

    def plot_transition_network(self):
        """亚稳态间的转换网络图"""
        fig, ax = plt.subplots(figsize=(10, 10))

        n_meta = self.result.n_metastable
        mfpt = self.result.kinetics.get('mfpt_matrix_ps', np.zeros((n_meta, n_meta)))
        probs = self.result.metastable_probs

        # 节点位置（圆形排列）
        angles = np.linspace(0, 2*np.pi, n_meta, endpoint=False) - np.pi/2
        pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

        # 绘制节点
        colors = plt.cm.Set1(np.linspace(0, 1, n_meta))
        for i in range(n_meta):
            size = 2000 + 3000 * probs[i] / probs.max()
            ax.scatter(*pos[i], s=size, c=[colors[i]], zorder=10, edgecolors='black', linewidth=2)
            ax.annotate(
                f'State {i}\n{probs[i]*100:.1f}%',
                pos[i], fontsize=11, ha='center', va='center', zorder=11,
                fontweight='bold'
            )

        # 绘制边（箭头）
        for i in range(n_meta):
            for j in range(n_meta):
                if i != j and not np.isnan(mfpt[i, j]) and mfpt[i, j] > 0:
                    # 箭头粗细与 MFPT 成反比（快转换 = 粗箭头）
                    mfpt_ns = mfpt[i, j] / 1000
                    width = max(0.5, min(3.0, 10.0 / (mfpt_ns + 1)))

                    ax.annotate(
                        '', xy=pos[j], xytext=pos[i],
                        arrowprops=dict(
                            arrowstyle='->', color='gray', lw=width,
                            shrinkA=50, shrinkB=50
                        )
                    )

                    # 标注 MFPT
                    mid = ((pos[i][0] + pos[j][0])/2 * 0.7,
                           (pos[i][1] + pos[j][1])/2 * 0.7)
                    if mfpt_ns >= 1:
                        label = f'{mfpt_ns:.1f} ns'
                    else:
                        label = f'{mfpt[i, j]:.0f} ps'
                    ax.text(mid[0], mid[1], label, fontsize=9, ha='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_xlim([-1.8, 1.8])
        ax.set_ylim([-1.8, 1.8])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Metastable State Transition Network{self._get_title_suffix()}', fontsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'transition_network.png', dpi=300)
        plt.close()
        print(f"    [OK] transition_network.png")

    def plot_metastable_timeline(self):
        """亚稳态随时间演化图"""
        fig, ax = plt.subplots(figsize=(14, 4))

        meta_labels = self.result.get_metastable_labels()
        time_ns = np.arange(len(meta_labels)) * self.result.timestep_ps / 1000.0

        scatter = ax.scatter(time_ns, meta_labels, c=meta_labels, cmap='Set1', s=5, alpha=0.5)

        ax.set_yticks(range(self.result.n_metastable))
        ax.set_yticklabels([f'State {i}' for i in range(self.result.n_metastable)])
        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel('Metastable State', fontsize=12)
        ax.set_title(f'Metastable State Evolution{self._get_title_suffix()}', fontsize=14)
        ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metastable_timeline.png', dpi=300)
        plt.close()
        print(f"    [OK] metastable_timeline.png")
