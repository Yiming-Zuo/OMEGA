# 通用小分子溶剂化采样评估方法

**采样目的**：为概率流自由能计算采集高质量的玻尔兹曼分布端点

**评估重点**：不是"探索了多少构象"，而是**采样是否真正代表了目标温度下的平衡分布**

---

## 评估维度

### 1. 收敛性（最关键）
采样是否已达到平衡态？

| 指标 | 判据 | 说明 |
|-----|------|-----|
| JS散度 | < 0.1 | 前后半分布一致 → 已收敛 |
| 分布相似度 | > 0.9 | 块间分布稳定 |
| 累积均值 | 平台期 | 不再漂移 |

### 2. 构象覆盖度
重要构象是否都采到了？

| 指标 | 判据 | 说明 |
|-----|------|-----|
| 聚类数稳定 | 不再增加 | 没有遗漏的构象态 |
| 二面角分布 | 多峰均有采样 | 各势阱都被访问 |
| 状态转变 | 多次往返 | 构象间可逆转变 |

### 3. 副本交换效率
增强采样是否在工作？

| 指标 | 判据 | 说明 |
|-----|------|-----|
| 交换接受率 | 20-40% | 过低=温度间隔太大，过高=资源浪费 |
| Round-trip | ≥ 3 | replica完成温度梯度往返 |

### 4. 统计质量
重加权后的样本够不够用？

| 指标 | 判据 | 说明 |
|-----|------|-----|
| 有效样本数 | > 100 | 概率流需要足够独立样本 |
| ESS/总样本 | > 10% | 采样效率合理 |

---

## 使用

```bash
python evaluation/evaluate_sampling.py \
    --traj-dir outputs/xxx/hremd/trajectories \
    --topology outputs/xxx/system.pdb \
    --samples outputs/xxx/hremd/samples.arrow \
    --output results/evaluation_xxx
```

## 输出

- `evaluation_report.txt` - 量化指标报告
- `convergence_analysis.png` - 收敛性诊断图
- `conformational_space.png` - 构象空间覆盖图
- `pca_projection.png` - PCA投影图
