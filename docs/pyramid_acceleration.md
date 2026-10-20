# Tinybrain pyramid construction

`build_mips_from_yxc(base_yxc, num_mips)` keeps the audited NumPy numerical contract:
ceil-sized YXC levels, unchanged dtype, partial-block border means, integer nearest
rounding with ties to even **after every level**, and the original base object as
level zero. No public signature or caller changes are required.

## Investigation

The installed distribution is tinybrain **1.7.0** (its internal `__version__` string
incorrectly says 1.4.1). The installed `accelerated.pyx` and `accelerated.hpp` were
inspected and checked using small asymmetric arrays (`--probe` below).

* The raw `accelerated.average_pooling_2x2` API takes an array, reduction count and
  optional sparse flag. It returns only generated levels, retaining trailing axes.
  It expands inputs to five dimensions, then passes a raw pointer to C++ without
  passing strides. The kernel assumes Fortran-contiguous storage. A channel view
  of C-contiguous YXC data is neither packed nor Fortran contiguous: it scrambles
  both spatial positions and channels. The public `downsample_with_averaging`
  wrapper calls `asfortranarray`, but that entails a whole-image conversion.
* In the single-level kernel, the odd first-axis branch writes to `ox - 1`,
  overwriting the previous output and leaving the edge empty. The odd second-axis
  branch reuses a pixel pair outside its loop and uses a nonzero-count denominator
  even in nonsparse mode. Some reads can extend outside the logical image.
  Avoiding odd inputs is necessary, including on subsequent levels.
* Tinybrain rejects reductions once the smallest spatial dimension is less than
  `2**num_mips`; the NumPy contract continues down to 1x1.
* Integer native means truncate toward zero. The multilevel implementation keeps
  accumulated sums rather than feeding each emitted, rounded level into the next
  reduction (and periodically truncates accumulators). Repeating one-level calls
  and issuing one multilevel call are numerically different operations.
* Example with integral means: a Fortran `(6,10,3)` uint16 ramp multiplied by four
  has no mismatches; C storage has 44 mismatched channel values in its first level.
  A Fortran `(5,7,3)` input has 27 mismatches in the single-level call. In a 4x4
  example whose block sums are `[2,2,2,10]`, native multilevel returns 1 at the last
  level, whereas repeated native calls and the NumPy reference return 0.

## Implementation and numerical guarantees

The accelerated path accepts native uint8/uint16 YXC arrays with at least 4096 XY
pixels and both spatial dimensions at least two. C, F, cropped, spatial/channel
strides, transposes and negative strides are supported. It copies one channel tile
at a time to packed float32 Fortran storage, choosing XY or YX orientation to favor
the input's faster spatial axis. The orientation is reversed on output.

Tiles have even dimensions and sides at most 1024. Input conversion uses at most
4 MiB, with at most 1 MiB of native float32 output per tile, independent of full
image size. No complete image or channel is converted to floating point. Four
uint16 values sum to at most 262140, exactly representable in float32. Tinybrain's
single-level float kernel actually promotes its arithmetic to double; its output
still represents every quarter-integer mean exactly. In-place `np.rint` followed
by assignment to the integer output exactly matches the reference, including
maxima and all ties. Every level is computed from the preceding integer level.

Odd trailing rows/columns use the unchanged NumPy helper; the even interior still
uses tinybrain. Entire levels use NumPy for smaller/degenerate images, float32 and
other dtypes, nonnative byte order, or unavailable tinybrain. Float32 retains the
reference's float64 accumulation, avoiding changes for cancellation and large
magnitudes. No relaxed numerical tolerance is needed against the existing NumPy
implementation for any supported dtype.

## Validation

The original 39 pyramid cases remain. Added tests compare entire arrays exactly
across layouts, asymmetric even/odd sizes, channel counts, dtypes and repeated
reductions to 1x1. Tests check every possible four-pixel uint8/uint16 sum, including
all rounding ties, and exercise multiple tiles in both spatial directions. A spy
on the real native kernel proves representative large inputs use it, verifies
Fortran float32 input and even dimensions, and bounds input tile bytes. Writer/read
round trips check every level for uint8, uint16 and float32, in C and F order.

Only pyramid implementation, pyramid tests, benchmark script and this report are
changed. Native source pyramid ingestion, QNP assets, registration, plating,
densification and Allen `WSI_PIN` enforcement are untouched. All 200 focused/downstream
tests pass (91 pyramid tests); no failure required a baseline reproduction. Existing
scikit-image deprecation warnings occur in the processing tests.

## Reproduction

From the repository root, using the existing environment:

```bash
PY=/cis/home/dpadova/miniconda3/envs/wsi-pipeline/bin/python
export PYTHONPATH=src
"$PY" scripts/benchmark_pyramid.py --probe
"$PY" scripts/benchmark_pyramid.py
"$PY" scripts/benchmark_pyramid.py --methods prepare --mips 2
"$PY" scripts/benchmark_pyramid.py --shape 2048 1536 3 --layouts C F --repeats 1
"$PY" scripts/benchmark_pyramid.py --shape 2049 1537 3 --dtype uint8 --methods numpy hybrid --layouts C F --repeats 1
"$PY" scripts/benchmark_pyramid.py --shape 2048 1536 3 --dtype float32 --methods numpy hybrid --layouts C --repeats 3
"$PY" -m pytest --no-cov tests/test_omezarr_pyramid.py tests/test_omezarr_writers.py tests/test_zarr_reader_compat.py tests/test_plating_metadata.py tests/test_vsi_ets_pipeline.py tests/test_processing.py tests/test_vsi_benchmark.py tests/test_vsi_ometiff_benchmark.py tests/test_vsi_converter_ometiff.py
"$PY" -m ruff check src/wsi_pipeline/omezarr/pyramid.py tests/test_omezarr_pyramid.py scripts/benchmark_pyramid.py
"$PY" -m black --check src/wsi_pipeline/omezarr/pyramid.py tests/test_omezarr_pyramid.py scripts/benchmark_pyramid.py
git diff --check
```

## Benchmarks

Linux x86_64, Xeon E5-1650 v4 at 3.60 GHz, 125 GiB RAM, Python 3.12,
NumPy 2.4.3, tinybrain distribution 1.7.0. The large base is
8192x6144x3 uint16 (288 MiB of image data). Each measurement is a fresh subprocess;
time is the median of three runs and peak RSS is the maximum of the three.
Input construction and module startup are outside timing; **all pyramid work,
layout/dtype conversion, rounding, output allocation and stacking are inside**.
No WSI was reconstructed. Two levels means one reduction; five means four.

RSS includes imports, input backing storage, all returned levels and temporary
buffers. The package imports have a substantial baseline (~732 MiB). A strided
input views every other column of a 576 MiB backing array, so its total RSS is
higher. Extra RSS is the increase in process high-water RSS over the high-water
mark immediately after input construction; this is a comparative estimate, not
a separate allocator measurement.

The original implementation is timed only with F storage and spatial dimensions
divisible by the complete reduction factor, where its memory indexing is valid.
Its integer truncation and multilevel averaging conventions still differ from the
required reference; these timings are historical performance comparisons, not an
endorsement of numerical equivalence.

| Layout | Reductions | Implementation | Wall time (s) | Peak RSS (MiB) | Extra RSS (MiB) |
|---|---:|---|---:|---:|---:|
| C | 1 | numpy | 3.3399 | 2028.5 | 1008.2 |
| C | 1 | hybrid | 0.2202 | 1095.3 | 75.0 |
| C | 4 | numpy | 4.4498 | 2028.3 | 1008.4 |
| C | 4 | hybrid | 0.2864 | 1118.0 | 97.6 |
| F | 1 | original | 0.2107 | 1452.7 | 432.3 |
| F | 1 | numpy | 1.4668 | 2028.8 | 1008.4 |
| F | 1 | hybrid | 0.2353 | 1095.2 | 75.4 |
| F | 4 | original | 0.3387 | 1507.1 | 487.0 |
| F | 4 | numpy | 1.9505 | 2028.4 | 1008.3 |
| F | 4 | hybrid | 0.2971 | 1118.3 | 97.7 |
| strided | 1 | numpy | 3.5483 | 2316.4 | 1008.3 |
| strided | 1 | hybrid | 0.2832 | 1383.8 | 75.1 |
| strided | 4 | numpy | 4.6844 | 2316.7 | 1008.2 |
| strided | 4 | hybrid | 0.3674 | 1406.3 | 97.6 |

The hybrid is 6.2–15.5 times faster than NumPy on these layouts. F-input hybrid
speed is within ~12% of the original for one reduction, and faster for four,
while preserving the reference pixels and avoiding a duplicate full-resolution
base plus per-channel output stacks. The extra RSS is ~75/98 MiB for one/four
reductions rather than NumPy's ~1008 MiB. Returned downsampled levels themselves
occupy 72 MiB for one reduction and ~95.6 MiB for four.

### Conversion cost

Preparing just the base image tiles (no pooling or returned pyramid), median of
three subprocesses:

| Layout | Conversion time (s) | Extra peak RSS (MiB) |
|---|---:|---:|
| C | 0.1084 | 4.1 |
| F | 0.0589 | 4.0 |
| strided | 0.1665 | 4.0 |

Conversion is a material part of the total time, but remains bounded in memory
and substantially cheaper than the NumPy reference. These supplementary times
are **not subtracted** from the end-to-end results above.

### Smaller odd images

2049x1537x3 uint8; single subprocess per measurement. Peak RSS includes input
and imports as above. The odd edges use NumPy while the interior uses tinybrain.

| Layout | Reductions | Implementation | Wall time (s) | Peak RSS (MiB) |
|---|---:|---|---:|---:|
| C | 1 | numpy | 0.2054 | 802.8 |
| C | 1 | hybrid | 0.0154 | 747.0 |
| C | 4 | numpy | 0.2757 | 802.8 |
| C | 4 | hybrid | 0.0230 | 747.0 |
| F | 1 | numpy | 0.0888 | 802.8 |
| F | 1 | hybrid | 0.0155 | 746.9 |
| F | 4 | numpy | 0.1222 | 802.6 |
| F | 4 | hybrid | 0.0200 | 747.1 |

### Float32 fallback

2048x1536x3 float32, C layout; median of three subprocesses:

| Reductions | Implementation | Wall time (s) | Peak RSS (MiB) | Extra RSS (MiB) |
|---:|---|---:|---:|---:|
| 1 | numpy | 0.2188 | 875.5 | 108.1 |
| 1 | hybrid | 0.2199 | 875.4 | 108.1 |
| 4 | numpy | 0.2904 | 875.6 | 108.1 |
| 4 | hybrid | 0.2880 | 875.8 | 108.1 |

The float32 fallback uses the same NumPy helper and has essentially identical
runtime and memory. Its float64 accumulation is intentionally retained rather
than changing cancellation behavior to obtain acceleration.
