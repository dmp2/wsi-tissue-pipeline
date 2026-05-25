# Source-Level-0 Production Runbook

This runbook prepares and manually launches the E241 source-level-0 production export. It does not run source level 0 automatically.

Production semantics:

- `primary_rgb_mode=masked_rgb`
- `masked_rgb_fill_value=0`
- `extra_margin_px=0`
- `pyramid_generation_policy=native_source_pyramid_crop`
- `source_tile_aligned_canvas=true`
- `native_mip_stop_level=segmentation_level`
- `labels/tissue_mask` retained
- empty mask chunks skip RGB ETS decode and RGB writes

## Preflight

```bash
REPO="/cis/home/dpadova/Documents/git/wsi-tissue-pipeline"
PY="/cis/home/dpadova/miniconda3/envs/wsi-pipeline/bin/python"
VSI="/cis/home/dpadova/Documents/temporal_bone_project/RE_ working with VSI files/OneDrive_1_7-12-2024/Image_01.vsi"
BASE="/cis/home/dpadova/Documents/temporal_bone_project/e241_vsi_production_source_level_0"

cd "$REPO"
export PYTHONPATH="$REPO/src"
mkdir -p "$BASE"
df -h "$BASE" "$(dirname "$VSI")"
```

## Estimate

```bash
OUT="$BASE/estimate"
mkdir -p "$OUT"

$PY -m wsi_pipeline.cli estimate-vsi-plating \
  --vsi "$VSI" \
  --source-level 0 \
  --segmentation-level 7 \
  --output-profile production \
  --tile-frame-level segmentation \
  --metadata-backend bioformats \
  --config configs/e241_production_default.yaml \
  --output-json "$OUT/production_estimate_s0.json" \
  2>&1 | tee "$OUT/production_estimate_s0.log"
```

Review the estimate before running production. Confirm:

- `extra_margin_px` and `context_margin_px` are `0`
- `source_tile_aligned_canvas` is `true`
- `output_scale_to_source_level` maps `s0..s7` to `0..7`
- `mip_stop_reason` is `segmentation_level`
- projected disk and runtime are acceptable
- there are no unexpected `baseline_config_mismatch:*` warnings

## Manual Production Export

Run this only after the estimate is acceptable.

```bash
OUT="$BASE"
mkdir -p "$OUT"

$PY - <<'PY' 2>&1 | tee "$OUT/source_level_0_production_write.log"
from pathlib import Path
import logging

from wsi_pipeline.config import SegmentationConfig, TileConfig
from wsi_pipeline.pipeline import process_vsi_directory_with_plating

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

INPUT_DIR = Path("/cis/home/dpadova/Documents/temporal_bone_project/RE_ working with VSI files/OneDrive_1_7-12-2024")
OUT = Path("/cis/home/dpadova/Documents/temporal_bone_project/e241_vsi_production_source_level_0")

process_vsi_directory_with_plating(
    input_dir=INPUT_DIR,
    output_dir=OUT,
    pattern="Image_01.vsi",
    source_level=0,
    segmentation_level=7,
    output_profile="production",
    tile_frame_level="segmentation",
    segmentation_config=SegmentationConfig(
        backend="local-entropy",
        target_long_side=1800,
        min_area_px=3000,
        struct_elem_px=5,
        split_touching=True,
        r_split=5,
        stain_gate=True,
        stain_gate_mode="adaptive-he",
        stain_min_od=0.10,
        stain_min_he_signal=0.0,
        stain_od_bg_percentile=0.80,
        stain_od_mad_multiplier=4.0,
        stain_min_saturation=0.08,
        stain_pre_open_px=0,
        keep_top_k=None,
        appendage_refinement_enabled=True,
        appendage_refinement_mode="trim",
        appendage_refinement_profile="he_sections",
        diagnostics=True,
    ),
    tile_config=TileConfig(
        chunk_size=512,
        pad_multiple=512,
        extra_margin_px=0,
        crop_shape_policy="compact_rectangle",
    ),
    metadata_backend="bioformats",
    metadata_schema="v0.4",
    overwrite_source=True,
    parallel=False,
    compression="lossless",
    primary_rgb_mode="masked_rgb",
    masked_rgb_fill_value=0,
    store_tissue_mask=True,
    store_unmasked_rgb=False,
    sparse_zero_chunks=True,
    pyramid_generation_policy="native_source_pyramid_crop",
    source_tile_aligned_canvas=True,
    native_mip_stop_level="segmentation_level",
    resume=True,
    progress_mode="both",
    progress_interval_s=60.0,
)
PY
```

Do not rely on the Python API return value for output paths. Discover outputs with:

```bash
find "$BASE/per_tissue_ngff" -maxdepth 1 -name "*.ome.zarr" -print | sort | tee "$BASE/tissue_paths.txt"
```

Resume behavior:

- `resume=True` skips completed `.ome.zarr` tissue outputs.
- If a stale `.incomplete` directory exists from an interrupted run, inspect the log before deleting or rerunning.

## QC

```bash
$PY - <<'PY'
from pathlib import Path
from wsi_pipeline.qc_grid import run_qc_workflow

OUT = Path("/cis/home/dpadova/Documents/temporal_bone_project/e241_vsi_production_source_level_0")
result = run_qc_workflow(
    input_dir=OUT / "per_tissue_ngff",
    output_dir=OUT / "qc",
    label_mode="both",
    qc_display_mode="auto",
)
print(f"QC records: {len(result.records)}")
print(f"Master grid: {result.artifacts.master_contact_sheet}")
print(f"Stats CSV:   {result.artifacts.stats_csv}")
PY
```

## Validation

```bash
$PY - <<'PY'
from pathlib import Path
import json
import numpy as np
import zarr

OUT = Path("/cis/home/dpadova/Documents/temporal_bone_project/e241_vsi_production_source_level_0")
tissues = sorted((OUT / "per_tissue_ngff").glob("*.ome.zarr"))
assert len(tissues) == 3, f"Expected 3 tissues, found {len(tissues)}"

for tissue_dir in tissues:
    root = zarr.open_group(str(tissue_dir), mode="r")
    manifest = json.loads((tissue_dir / "tissue_manifest.json").read_text())
    run_manifest = json.loads((tissue_dir / "run_manifest.json").read_text())
    mask_group = zarr.open_group(str(tissue_dir / "labels" / "tissue_mask"), mode="r")

    assert manifest["primary_rgb_mode"] == "masked_rgb"
    assert manifest["masked_rgb_fill_value"] == 0
    assert manifest["mask_applied_to_primary_rgb"] is True
    assert manifest["pyramid_generation_policy"] == "native_source_pyramid_crop"
    assert manifest["source_tile_aligned_canvas"] is True
    assert run_manifest["mip_stop_reason"] == "segmentation_level"
    assert run_manifest["output_scale_to_source_level"] == {f"s{i}": i for i in range(8)}

    for level in range(8):
        rgb = root[f"s{level}"]
        mask = mask_group[f"s{level}"]
        assert rgb.shape[1:] == mask.shape, (tissue_dir.name, level, rgb.shape, mask.shape)
        sample_mask = mask[: min(512, mask.shape[0]), : min(512, mask.shape[1])]
        assert set(np.unique(sample_mask).tolist()).issubset({0, 1})
        sample_rgb = rgb[:, : sample_mask.shape[0], : sample_mask.shape[1]]
        outside = sample_rgb[:, sample_mask == 0]
        if outside.size:
            assert int(outside.max()) == 0

    assert float(run_manifest["rgb_write_amplification"]) == 1.0
    assert float(run_manifest["mask_write_amplification"]) == 1.0
    assert int(run_manifest["rgb_chunks_skipped_before_decode"]) > 0
    print(tissue_dir.name, "OK")

print("source_level=0 production validation OK")
PY
```

## Disk Summary

```bash
du -sh "$BASE/per_tissue_ngff" | tee "$BASE/du_actual.txt"
du -sh --apparent-size "$BASE/per_tissue_ngff" | tee "$BASE/du_apparent.txt"
find "$BASE/per_tissue_ngff" -type f | wc -l | tee "$BASE/file_count.txt"
```
