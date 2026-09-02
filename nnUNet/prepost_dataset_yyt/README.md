# YYT pre/post nnUNet dataset scripts

这组脚本将 `/share/data/CryoET_Data/yanyt/4ls/` 中的 ROI mask 整理为 nnUNet raw dataset，并可继续生成 bin4 数据集。

## 环境与依赖

使用 `synapseseg` Conda 环境。需要 `numpy`、`mrcfile`、`py7zr`、`scipy` 和已安装的 `synapseseg`。

```bash
conda run -n synapseseg python create_dataset.py --help
conda run -n synapseseg python create_bin4_dataset.py --help
```

服务器上的环境也可以直接调用：

```bash
/share/home/liushuo/miniconda3/envs/synapseseg/bin/python create_dataset.py --help
```

## 当前 YYT 数据规则

- 自动发现 `*/synapse_seg/roi/*_roi_mask.mrc.7z`。
- 同一 tomo 的基础目录与 `-1` 目录标签合并；基础标签优先，`-1` 只填充背景，类别冲突保留基础标签。
- 普通图像使用 `TOMO/synapse_seg/TOMO_wbp_corrected.mrc`。
- `pp387` 自动使用 `pp387/synapse_seg/isonet/tomo_deconv/pp387_wbp_resample.mrc`。
- 默认执行 `pp366: 3 -> 2 (post)` 和 `pp4001: 3 -> 1 (pre)`。
- 输出标签只允许 `0=background`、`1=pre`、`2=post`。
- 所有输出先写入同级 staging 目录，成功后原子重命名；不会覆盖已存在的数据集。

## 创建原始数据集

默认路径对应 Dataset004：

```bash
conda run -n synapseseg python create_dataset.py --expected-count 34
```

显式指定路径：

```bash
conda run -n synapseseg python create_dataset.py \
  --source-root /share/data/CryoET_Data/yanyt/4ls \
  --output /share/data/CryoET_Data/liushuo/dataset/nnUNet/nnUNet_raw/Dataset004_synapseseg_roiprepost \
  --expected-count 34
```

额外的标签映射可重复指定：

```bash
conda run -n synapseseg python create_dataset.py \
  --label-remap pp1234:3:1 \
  --label-remap pp1234:4:2
```

特殊图像路径可用 `--special-image CASE=PATH` 覆盖。

## 创建 bin4 数据集

默认从 Dataset004 创建 Dataset005：

```bash
conda run -n synapseseg python create_bin4_dataset.py --expected-count 34
```

bin4 直接使用 SynapseSeg 中的实现：

- 图像：`resample_image_by_bin4`，连续两次 2×2×2 块平均，输出 `float32`。
- 标签：`resample_label_with_output_shape`，最近邻重采样到图像输出 shape，输出 `int8`。
- voxel size 乘以 4；不能被 4 整除的末端图像体素按 SynapseSeg 的现有逻辑裁剪。

## 一次创建两个数据集

目标目录必须都不存在：

```bash
conda run -n synapseseg python run_pipeline.py --expected-count 34
```

如数据数量变化，请同步调整 `--expected-count` 和目标数据集名称。
