# MolPilot / 外部 `.pt` → GenBench3D 与 `evaluate_diffusion` 风格汇总（命令备忘）

本文记录在本仓库中已使用的一套流程：**从 MolPilot 或兼容格式的 `torch.save` 结果（扁平 `list[dict]` 或带 `all_results` 的 `dict`）导出分组 SDF**，再跑 **GenBench3D**，以及用脚本生成与 **`evaluate_diffusion.run_evaluation`** 对齐的 **TSV 指标**。适用于 `benchmarks/from_molpilot/*.pt`、`tmp_eval/.../eval_results/metrics_*.pt` 等。

更通用的 GenBench3D 背景与依赖说明见同目录：**[2026-04-01_GenBench3D_Evaluation_Guide.md](./2026-04-01_GenBench3D_Evaluation_Guide.md)**。`benchmarks/from_molpilot/` 下另有简版 **MOLPILOT_PT_GENBENCH_WORKFLOW.md**（与本文互补）。

---

## 1. 仓库与共享路径

| 项 | 路径 |
|----|------|
| 仓库根目录 | `/shared/healthinfolab/phz24002/TAGMol` |
| MolPilot 类结果（示例） | `benchmarks/from_molpilot/*.pt` |
| GenBench 测试集（native + `_rec.pdb`） | `data/test_set/` |
| GenBench3D 仓库（默认） | `/shared/healthinfolab/phz24002/genbench3d` |

---

## 2. Python 环境

| 步骤 | 推荐解释器 | 说明 |
|------|------------|------|
| 从 `.pt` 导出 SDF（需 unpickle RDKit `mol`） | `~/anaconda3/envs/targetdiff/bin/python` | MolPilot 常用较新 RDKit pickle；**共享 `tagmol` 可能对部分文件 segfault**。可用环境变量 **`MOLPILOT_EXPORT_PYTHON`** 覆盖。 |
| GenBench3D（`sb_benchmark_mols.py`） | `~/anaconda3/envs/genbench3d/bin/python` | 需 MDAnalysis 等；用 **`GENBENCH_PYTHON`** 传给 `run_genbench_eval.py`。 |
| `get_genbench_report.py` | 与 GenBench 相同或任意有 `json` 的环境即可 | |

导出与 GenBench 前建议：

```bash
cd /shared/healthinfolab/phz24002/TAGMol
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

---

## 3. 从 `.pt` 导出 `*_generated.sdf`

脚本：`scripts/export_molpilot_pt_to_sdfs_grouped.py`  
逻辑与 `export_to_sdf_grouped.py` 一致：按 `ligand_filename` 分组，写入 `{key}_generated.sdf`，供 `run_genbench_eval.py` 使用。

```bash
/home/phz24002/anaconda3/envs/targetdiff/bin/python scripts/export_molpilot_pt_to_sdfs_grouped.py \
  --pt /path/to/file.pt \
  --out-dir /shared/healthinfolab/phz24002/TAGMol/benchmarks/from_molpilot/<SHORT>_sdfs_grouped
```

---

## 4. 一键：导出 + GenBench + 报告

脚本：`benchmarks/from_molpilot/run_molpilot_pt_genbench_eval.sh`

**约定：** SDF 与 GenBench 结果写在 `benchmarks/from_molpilot/{SHORT}_sdfs_grouped/`；日志为 `{SHORT}_export_sdfs.log`、`_genbench_eval.log`、`_genbench_report.log`。  
`SHORT` = 输入文件 basename 去掉 `.pt`，若后缀为 `_vina_docked_pose_checked` 则再剥掉该后缀（例如 `tagmol_vina_docked_pose_checked.pt` → `tagmol`）。

### 4.1 仅 stem（文件必须在 `benchmarks/from_molpilot/<stem>.pt`）

```bash
cd /shared/healthinfolab/phz24002/TAGMol
bash benchmarks/from_molpilot/run_molpilot_pt_genbench_eval.sh tagmol_vina_docked_pose_checked
bash benchmarks/from_molpilot/run_molpilot_pt_genbench_eval.sh pocket2mol_vina_docked_pose_checked
```

### 4.2 任意路径的 `.pt`（含 `tmp_eval/.../metrics_-1.pt`）

```bash
bash benchmarks/from_molpilot/run_molpilot_pt_genbench_eval.sh --pt tmp_eval/quick_eval_2026_04_22__13_48_25_xz76hb4g/eval_results/metrics_-1.pt
```

也可传入绝对路径。相对路径会相对 **仓库根** 解析（若存在 `ROOT/相对路径`）。

### 4.3 环境变量（可选）

```bash
export MOLPILOT_EXPORT_PYTHON=/home/phz24002/anaconda3/envs/targetdiff/bin/python
export GENBENCH_PYTHON=/home/phz24002/anaconda3/envs/genbench3d/bin/python
export GB3D_DIR=/shared/healthinfolab/phz24002/genbench3d
```

---

## 5. 计算节点上 `nohup` 连续跑多条（示例：cn466）

仓库内提供串联脚本（先 TAGMol MolPilot 文件，再 quick_eval 的 `metrics_-1.pt`）：

```bash
ssh cn466
cd /shared/healthinfolab/phz24002/TAGMol
nohup bash benchmarks/from_molpilot/nohup_genbench_tagmol_and_quick_eval.sh &
```

- **主日志（两段任务 + 子脚本输出）：**  
  `benchmarks/from_molpilot/nohup_genbench_tagmol_metrics.log`
- 各任务另有：`tagmol_*`、`metrics_-1_*` 三个分日志。

查看进度：

```bash
tail -f /shared/healthinfolab/phz24002/TAGMol/benchmarks/from_molpilot/nohup_genbench_tagmol_metrics.log
```

**单独**对两条各跑一次（也可分别 `nohup`）：

```bash
nohup bash benchmarks/from_molpilot/run_molpilot_pt_genbench_eval.sh tagmol_vina_docked_pose_checked &
# 上一段结束后再：
nohup bash benchmarks/from_molpilot/run_molpilot_pt_genbench_eval.sh \
  --pt tmp_eval/quick_eval_2026_04_22__13_48_25_xz76hb4g/eval_results/metrics_-1.pt &
```

---

## 6. `evaluate_diffusion` 风格指标（TSV，不进 GenBench log）

脚本：`scripts/summarize_molpilot_pt_like_evaluate_diffusion.py`  
聚合 **`mol_stable`、`atm_stable`、各类 `JSD_*`、`QED`/`SA`、RDK diversity、Vina score/min/dock** 等，定义尽量对齐 `scripts/evaluate_diffusion.py` 的 `run_evaluation`。  
**无 `pred_pos`/`pred_v` 时**用 **mol 构象** 做稳定性与原子对距，与「仅在 pred 上统计」的原始训练评测可能有细微差别。

```bash
cd /shared/healthinfolab/phz24002/TAGMol
export PYTHONPATH="${PWD}:${PYTHONPATH}"
/home/phz24002/anaconda3/envs/targetdiff/bin/python scripts/summarize_molpilot_pt_like_evaluate_diffusion.py \
  --pt benchmarks/from_molpilot/molcraft_vina_docked_pose_checked.pt
# 默认写出：同目录 <stem>_evaluate_like.tsv
# 指定输出：  --out benchmarks/from_molpilot/custom.tsv
```

---

## 7. GenBench 汇总单行表（粘 Excel）

对 **`get_genbench_report.py` / `*_genbench_report.log`** 中的聚合指标，可整理为制表符一行格式，例如：

`Pockets`、`Validity3D_pct`（0–1）、`Uniqueness3D_mean`、`Diversity3D_mean`、`Strain_energy_mean/median`、`Clash_free_pct`（0–100）、`Centroid_dist_mean_A`、`Diversity2D_pct`、`Uniqueness2D_pct`、`Validity2D_pct`（后三项 0–1）。

部分 run 无有限 TFD 时 **Uniqueness3D / Diversity3D** 为空列。历史上曾将多 run 写入：

- `benchmarks/from_molpilot/GENBENCH_SUMMARY_3RUNS.tsv`
- `benchmarks/from_molpilot/GENBENCH_SUMMARY_DECOMPDIFF_MOLJO.tsv`

重新生成报告：

```bash
/home/phz24002/anaconda3/envs/genbench3d/bin/python get_genbench_report.py \
  benchmarks/from_molpilot/<SHORT>_sdfs_grouped/genbench_results
```

---

## 8. 与原生训练评测链的对应关系

| 步骤 | 原生（自训模型目录） | MolPilot / 外部 `.pt` |
|------|----------------------|------------------------|
| 得到带 `mol` 的 metrics | `evaluate_diffusion.py` → `eval_results/metrics_*.pt` | 已有 `.pt` |
| 分组 SDF | `export_to_sdf_grouped.py <run_dir>` | `export_molpilot_pt_to_sdfs_grouped.py --pt ...` |
| GenBench | `run_genbench_eval.py --gen_dir .../sdfs_grouped` | 同上；`--test_set_dir` 指向 `data/test_set` |
| 控制台表 | `get_genbench_report.py .../genbench_results` | 同上 |

GenBench 常用参数（与 `quick_evaluate --run_genbench` 一致）：**`--do_conf_analysis`、`--no_vina`**（避免与 MolPilot 内已算 Vina 重复；需要 GenBench 内 Vina 时去掉 `--no_vina` 并确认 GenBench 配置）。

---

## 9. Google Drive 下载（MolPilot 包）

示例（需 `gdown`，可用共享 Anaconda 的 `gdown`）：

```bash
/shared/healthinfolab/phz24002/anaconda3/bin/gdown \
  "https://drive.google.com/uc?id=<FILE_ID>" \
  -O benchmarks/from_molpilot/<name>.pt
```

---

## 10. 常见问题

1. **`Bad pickle format: ENDMOL` / RDKit 版本警告**  
   升级当前环境的 **RDKit** 至不低于保存 `.pt` 时使用的版本，或换用 MolPilot 官方环境再 `torch.load`。

2. **Shell 脚本报 `pipefail` / `$'\r': command not found`**  
   在 Linux 上执行：`sed -i 's/\r$//' <脚本.sh>`。

3. **GenBench `Success` 少于 100**  
   部分方法 `.pt` 仅含 97–99 个口袋的有效 SDF，与数据一致，非脚本错误。

4. **`run_molpilot_pt_genbench_eval.sh` 与 `run_molcraft_genbench_eval.sh`**  
   后者为包装：`molcraft_vina_docked_pose_checked` → 调用通用 `run_molpilot_pt_genbench_eval.sh`。

---

*文档整理自仓库内已落地的 MolPilot/TAGMol GenBench 与 evaluate_like 流程；路径与脚本名以仓库当前版本为准。*
