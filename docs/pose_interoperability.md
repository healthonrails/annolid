# DeepLabCut, SLEAP & External Pose Interoperability

Annolid provides bidirectional import and export for pose and keypoint tracking ecosystems used in behavioral neuroscience:

| Ecosystem / Format | Import to Annolid | Export from Annolid | Notes |
| --- | --- | --- | --- |
| **DeepLabCut** analysis (`.csv`, `.h5`) | Single- & multi-animal (3- and 4-row headers), likelihood filtering, `config.yaml` skeleton | 3-row (single-animal) & 4-row (multi-animal) CSV (`x, y, likelihood`) | Auto-discovers `*DLC_*.csv` / `*DLC_*.h5`; reads fixed-format `.h5` with `h5py` (no PyTables required). |
| **DeepLabCut** training (`labeled-data/CollectedData_*.{csv,h5}`) | Single- & multi-animal CSV/H5, cross-platform paths, `config.yaml` edges, LabelMe-safe flags | Full project (`labeled-data/<video>/CollectedData_<scorer>.csv`, images, `config.yaml`) | Round-trips cleanly with `pose_schema.json`. |
| **Lightning Pose** (`.csv`, training projects) | Uses DLC 3- and 4-row CSV format | Same as DeepLabCut CSV / training project | Use `--format lightning-pose` or `--format auto`. |
| **SLEAP** projects (`.slp`, `.pkg.slp`) | Multi-video, `/tracks_json` track names, point/instance scores, skeleton edges & symmetry, occluded keypoints | `.slp` or self-contained `.pkg.slp` (`--embed-images`) | Prefers user labels over overlapping predictions on the same frame by default. |
| **SLEAP** Analysis HDF5 (`.analysis.h5`) | `/tracks`, `/node_names`, `/track_names`, `/edge_names`, `/point_scores`, `/instance_scores` | Standard SLEAP analysis HDF5 (`tracks`, `point_scores`, `instance_scores`, `track_occupancy`, `edge_names`, `edge_inds`) | Requires `h5py` (`pip install 'annolid[sleap]'`). |
| **SimBA / B-SOiD / VAME / Keypoint-MoSeq / `movement`** | Standard DLC CSV or SLEAP `.analysis.h5` | `--format deeplabcut-csv`, `--format simba`, or `--format sleap-h5` | Directly readable from Annolid LabelMe folders, `AnnotationStore` NDJSON, or `*_tracking.csv`. |
| **COCO Keypoints** (`.json`) | Keypoints + skeleton edges (with bbox-center fallback) | Existing COCO export + `coco2deeplabcut` real DLC CSV export | Preserves visibility (`1=occluded`, `2=visible`). |

---

## GUI Workflows

### DeepLabCut / Lightning Pose / SimBA

Open **Tools → DeepLabCut / Lightning Pose Import & Export…**:

1. **Import DLC / Lightning Pose analysis (CSV / H5 → LabelMe)**:
   - Pick a `.csv` or `.h5` file directly, or pick a video to auto-discover matching `*DLC_*.csv` / `*DLC_*.h5`.
   - Single-animal (3 header rows) and multi-animal (4 header rows: `scorer`, `individuals`, `bodyparts`, `coords`) layouts are auto-detected.
   - Optionally filter low-confidence predictions via **Min likelihood** and extract matching PNG frames.
2. **Import DLC / Lightning Pose training data**:
   - Select a `labeled-data` folder containing `CollectedData_*.csv` or `CollectedData_*.h5`.
   - If a sibling `config.yaml` exists, Annolid reads its `skeleton` edges and body-part ordering into `pose_schema.json`.
3. **Export Annolid LabelMe / Tracking → DeepLabCut / Lightning Pose / SimBA**:
   - Select a LabelMe annotation directory (including `*_annotations.ndjson` `AnnotationStore` files) or an Annolid `*_tracking.csv`.
   - Choose **Analysis CSV** (for DeepLabCut, Lightning Pose, SimBA, B-SOiD, VAME, or `movement`) or **Training Project** (`labeled-data/` + `config.yaml`).

### SLEAP (`.slp`, `.pkg.slp`, `.analysis.h5`)

Open **Tools → SLEAP (.slp / .h5) Import & Export…**:

1. **Import SLEAP → LabelMe**:
   - Select a `.slp`, `.pkg.slp`, or `.analysis.h5` file.
   - Optionally supply an external video (for non-packaged `.slp` or `.analysis.h5`) or an existing `pose_schema.json` for keypoint name mapping.
   - Choose whether to prefer user-labeled instances over overlapping predictions and whether to preserve occluded keypoints (`visible=False` with finite coordinates → COCO/Annolid `kp_visibility=1`).
2. **Export Annolid → SLEAP**:
   - Select a LabelMe folder or `*_tracking.csv`.
   - Export to **SLEAP Analysis HDF5 (`.analysis.h5`)**, **SLEAP Project (`.slp`)**, or **Self-Contained Package (`.pkg.slp`)** with embedded frames.

### Direct Overlay Playback in the GUI

When opening a video alongside a DeepLabCut or Lightning Pose analysis `.csv` or `.h5` (single- or multi-animal), Annolid parses 3- and 4-level headers automatically and overlays labeled points with multi-animal `group_id`, `instance_label`, and `score` metadata.

---

## CLI Workflows (`annolid-run`)

### Import External Pose Data (`import-pose`)

```bash
# Auto-detect DeepLabCut / Lightning Pose CSV or H5 (single- or multi-animal)
annolid-run import-pose trialDLC_dlcrnet_filtered.csv \
  --video trial.mp4 \
  --min-likelihood 0.5 \
  -o ./trial_labelme

# Import a SLEAP project (.slp / .pkg.slp) or SLEAP Analysis HDF5 (.analysis.h5)
annolid-run import-pose predictions.analysis.h5 \
  --video trial.mp4 \
  -o ./sleap_labelme

# Import a SLEAP .pkg.slp and generate an Annolid dataset index JSONL
annolid-run import-pose labels.pkg.slp \
  -o ./sleap_labelme \
  --write-index

# Import a DeepLabCut / Lightning Pose training project (legacy command also supported)
annolid-run import-deeplabcut-training-data \
  --source-dir /path/to/dlc_project \
  --labeled-data labeled-data \
  --write-pose-schema \
  --pose-schema-preset auto
```

### Export Annolid Annotations or Tracking (`export-pose`)

```bash
# Export LabelMe folder (or AnnotationStore) to DeepLabCut / Lightning Pose / SimBA CSV
annolid-run export-pose ./trial_labelme \
  --format deeplabcut-csv \
  --scorer my_lab \
  -o ./trial_dlc.csv

# Export LabelMe folder to a complete DeepLabCut / Lightning Pose training project
annolid-run export-pose ./trial_labelme \
  --format deeplabcut-project \
  --scorer my_lab \
  --video-name trial \
  -o ./dlc_project

# Export Annolid tracking CSV or LabelMe folder to SLEAP Analysis HDF5
annolid-run export-pose ./trial_tracking.csv \
  --format sleap-h5 \
  --video ./trial.mp4 \
  -o ./trial.analysis.h5

# Export LabelMe folder to a self-contained SLEAP .pkg.slp
annolid-run export-pose ./trial_labelme \
  --format sleap-slp \
  --embed-images \
  -o ./trial.pkg.slp
```

### COCO Keypoints to DeepLabCut CSV

```bash
python -m annolid.annotation.coco2deeplabcut annotations.json output_dlc.csv --scorer my_lab
```

---

## Dependencies & Format Notes

- **CSV formats** (DeepLabCut / Lightning Pose / SimBA / Annolid tracking CSV) work with Annolid's core dependencies (`pandas`, `PyYAML`, `Pillow`).
- **HDF5 & ZIP formats** (SLEAP HDF5 `.slp` / `.pkg.slp` / `.analysis.h5`, ZIP-based `.slp` archives, and DeepLabCut `fixed` & `table` format `.h5`) require `h5py` for HDF5 containers:
  ```bash
  pip install "annolid[sleap]"
  ```
- **External HDF5 compression filters**: Standard uncompressed, `gzip`, and `lzf` DeepLabCut `.h5` files (both `fixed` and PyTables `table` formats) are read directly via `h5py` without requiring `tables`; files compressed with optional third-party filters such as `blosc` require `hdf5plugin` or `tables` (or use the sibling `.csv`).
- **LabelMe flag safety**: Imported point shapes store boolean-only values in `shape["flags"]` (`kp_visible`, `predicted`) and keep string/numeric metadata (`instance_label`, `display_label`, `kp_visibility`, `score`) as top-level shape fields so LabelMe and Annolid GUI flag widgets never fail.
