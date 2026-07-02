# Native VSI/ETS to OME-TIFF Benchmark Runbook

This diagnostic path evaluates per-tissue pyramidal OME-TIFF/BigTIFF as an
interchange/export format alongside the validated native OME-Zarr path. It does
not replace or modify the OME-Zarr writer.

## E241 Fixture

```bash
VSI="/cis/home/dpadova/Documents/temporal_bone_project/RE_ working with VSI files/OneDrive_1_7-12-2024/Image_01.vsi"
```

The run should resolve the main ETS to `_Image_01_/stack10002/frame_t.ets` and
record the resolved path in `benchmark.json`.

## Smoke: Sampled Tiles

```bash
wsi-pipeline benchmark-vsi-ometiff \
  --vsi "$VSI" \
  --output-dir output/ometiff_source_level3_smoke \
  --config configs/e241_benchmark_actual.yaml \
  --source-level 3 \
  --segmentation-level 7 \
  --max-tissues 1 \
  --max-tiles 64 \
  --tile-sampling stratified \
  --compression deflate
```

`--max-tiles` produces benchmark-only partial payloads. Do not use those files
as final interchange artifacts.

## Smoke: Valid Readback

```bash
wsi-pipeline benchmark-vsi-ometiff \
  --vsi "$VSI" \
  --output-dir output/ometiff_source_level3_valid \
  --config configs/e241_benchmark_actual.yaml \
  --source-level 3 \
  --segmentation-level 7 \
  --max-tissues 1 \
  --compression deflate
```

## Full Source-Level-2 Run

Run only after both smoke stages pass.

```bash
wsi-pipeline benchmark-vsi-ometiff \
  --vsi "$VSI" \
  --output-dir output/ometiff_source_level2_full \
  --config configs/e241_benchmark_actual.yaml \
  --source-level 2 \
  --segmentation-level 7 \
  --max-tissues 3 \
  --compression deflate
```

Expected native mapping: `s0..s5 -> ETS levels 2..7`.

## Source-Level-0 Estimate Only

Do not launch source-level 0 production export automatically.

```bash
wsi-pipeline benchmark-vsi-ometiff \
  --vsi "$VSI" \
  --output-dir output/ometiff_source_level0_estimate \
  --config configs/e241_benchmark_actual.yaml \
  --source-level 0 \
  --segmentation-level 7 \
  --estimate-only
```

Expected native mapping: `s0..s7 -> ETS levels 0..7`.
