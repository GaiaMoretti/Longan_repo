
# Longan cluster weight estimation from partial RGB-D observations

Code and data accompanying the article:

Moretti, G., Cerasola, V. A., Landolfo, M., Chintakovid, W., Pichakum, A., Phatrapornnant, T., Pennisi, G., Gianquinto, G., and Orsini, F. (2026). Comparison of RGB-D geometric descriptors for longan cluster weight estimation under partial cluster visibility. *Frontiers in Horticulture*. [DOI]

The repository contains the full pipeline: YOLO instance segmentation of longan clusters, extraction of geometric descriptors from RGB-D views with view-level quality control, calibration against measured cluster weight, aggregation analysis, and video-based cluster counting.

## Repository structure
```
scripts/
  01_make_yolo_dataset.py       Builds the YOLO dataset from the CVAT annotation tasks
  02_train_yolo.py              Trains the segmentation and detection models
  03_benchmark_models.py        Compares the trained models: accuracy, speed, robustness (Figure 3; Tables S4, S5)
  03b_cluster_class_metrics.py  Cluster-class metrics of each model (Table 2)
  04_build_samples.py           Pairs RGB and depth frames of each harvested cluster with its weight (samples.json)
  05_build_features_QC.py       Extracts view-level descriptors and quality-control metrics
  06_calibration_analysis.py    Calibration, statistics and all weight-estimation tables and figures
  07_count_clusters.py          Cluster counting with tracking, segmentation vs. detection (Tables 3, 4)
  08_count_sensitivity.py       Sensitivity of the counts to the counting parameters (Table S11)
  checks/
    find_cluster_split.py       Which harvested clusters were used for YOLO training or validation (Table S8)
    mask_selection_check.py     Agreement of the selected mask with the annotations (Section 3.3.1)
data/
  image_level_QC.xlsx           One row per view: descriptors, quality-control metrics, cluster weight
  cluster_split.csv             YOLO split of each harvested cluster
results/
  calibration_results.xlsx      Output of 06_calibration_analysis.py (one sheet per table)
  figures/                      Figures 5, 7, 8 and S1
```

## Reproducing the results

The weight-estimation results can be reproduced from the files in `data/` alone:

```bash
pip install -r requirements.txt
python scripts/06_calibration_analysis.py
```

The other scripts require the raw RGB-D recordings, the CVAT annotations and the trained models, which are available from the corresponding author on reasonable request. Each script has a configuration block at the top, where the paths to these files must be set. The scripts are numbered in the order in which they are run.

## Data description

`data/image_level_QC.xlsx` contains one row per RGB-D view of the 51 harvested clusters:

| Column | Description |
| --- | --- |
| Cluster_ID | Harvested cluster (1–51) |
| Weight | Measured cluster weight (kg) |
| RGB, Depth | File names of the paired RGB and depth frames (the timestamp in the name is in epoch milliseconds) |
| status | `ok` if the view passed the geometric filters, otherwise the reason for rejection |
| valid_cov | Fraction of mask pixels with a valid depth value |
| n_masks, sel_conf | Number of predicted cluster masks and confidence of the selected (largest) mask |
| Area, Width, Height, Aspect_Ratio, Density | Mask area (px), bounding-box size (px), aspect ratio and fill ratio |
| Depth_Mean, Depth_Std | Mean and standard deviation of the depth inside the mask (m) |
| Voxel_Volume, Convex_Hull_Volume | Occupied voxel volume and convex hull volume of the visible point cloud (m³) |
| Proxy_Volume | Mask area × squared mean depth (px·m²); the depth-scaled projected area of the article is Proxy_Volume / (fx · fy), in m² |

The time offset between RGB and depth frames and the quality-control flags are recomputed by `06_calibration_analysis.py`.

## Software

Python 3.9, Ultralytics 8.4.152 (YOLO26; Jocher et al., 2026), PyTorch 2.5. See `requirements.txt`.

## License

Code: MIT License (see `LICENSE`). Data: CC BY 4.0.

## Citation

If you use this code or data, please cite the article above.
