# Polygon Frame Classifier Training Guide

This folder contains the end-to-end pipeline for training the frame-level polygon interaction classifier used in Annolid.

## 1) Set up the environment
- Install dependencies (GPU optional): `conda env create -f environment.yml` then `conda activate annolid`.
- Ensure PyTorch matches your CUDA version (see the `environment_*` files if you need alternatives).

## 2) Prepare the dataset CSVs
1. Organize annotations as JSON files per frame under train/test video folders (intruder/resident polygons with behavior flags).
2. (Optional) Add tracked motion CSVs named `<video>_tracked.csv` alongside the video folder to fill motion_index features.
3. Build CSVs with fixed-length polygon features and geometric extras:
   ```bash
   python -m annolid.datasets.polygon_features \
     --train_folder /path/to/train_jsons \
     --test_folder  /path/to/test_jsons \
     --output_folder ./datasets/calms21_polygon_dataset_motion_index \
     --num_points 40 \
     --normalize   # centers polygons; drop this flag to keep raw coords
   ```
   This writes `train_dataset.csv` and `test_dataset.csv` with all required columns.

## 3) Configure a run
- Use `annolid/behavior/training/polygon_frame_config.yaml` as a template. Key sections:
  - `run`: paths, run/output/log dirs, seed, log level.
  - `feature`: polygon padding length, frame size, normalization/rescaling, `compute_dynamic_features` (set false to skip filling inter_animal_distance/relative_velocity/facing_angle when missing), `compute_motion_index` (set false to drop motion index features).
  - `model`: temporal window, hidden dim, kernel size, residual blocks, dropout, attention.
  - `training`: batch size/epochs, LR/weight decay, scheduler + early stopping, sampler, loss, noise/smoothing (use `--no_label_smoothing` to turn it off), rolling median.

## 4) Launch training
Run from the repo root:
```bash
python -m annolid.behavior.training.polygon_frame_training \
  --config annolid/behavior/training/polygon_frame_config.yaml
```
- Override any setting from CLI (e.g., `--batch_size 32 --learning_rate 1e-3 --no_attention`).
- If you prefer passing CSVs directly, supply `--train_csv` and `--test_csv` and other flags inline.

## 5) Outputs
- Runs are saved under `output_dir/run_name`, auto-incremented (`exp`, `exp2`, …).
- Artifacts: best/latest checkpoints (`polygon_frame_classifier_best_*.pt`, `*_latest_*.pt`), training curves/history, metrics JSON/YAML dumps, and a log file per run.
- Final evaluation is run on the provided test CSV using the learned label mapping and polygon lengths.

## Tips
- Small datasets: increase `training.val_split_ratio` or ensure multiple videos to avoid empty validation splits. You’ll see a warning if val is empty.
- GPU vs CPU: the script auto-selects CUDA → MPS → CPU; set `CUDA_VISIBLE_DEVICES` to control GPU use.
- Stability: reduce `batch_size` or `num_workers` if you hit OOM; turn off attention with `--no_attention` for a lighter model.
- Feature scaling: keep `normalize_features` and `rescale_coordinates` aligned with how your CSVs were generated to avoid train/test drift.
