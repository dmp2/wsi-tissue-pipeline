"""End-to-end pyramid benchmark; each repetition runs in a fresh process.

Run from the checkout: PYTHONPATH=src python scripts/benchmark_pyramid.py
The original raw tinybrain code is measured ONLY with valid F-contiguous input.
It truncates integer means and retains sums across levels, unlike the reference.
RSS includes the base image, imports, returned levels, and conversion buffers.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from importlib.metadata import version

import numpy as np

from wsi_pipeline.omezarr.pyramid import _average_pool_2x2_yxc, build_mips_from_yxc


def original(base, levels):
    import tinybrain

    channels = []
    for c in range(base.shape[-1]):
        channels.append(
            [base[..., c]]
            + list(tinybrain.accelerated.average_pooling_2x2(base[..., c], levels - 1))
        )
    return [np.stack([channel[k] for channel in channels], axis=-1) for k in range(levels)]


def probe():
    """Small asymmetric, integral-mean arrays isolate layout/edge errors."""
    import tinybrain

    print("tinybrain", version("tinybrain"), tinybrain.__file__)
    for shape in [(6, 10, 3), (5, 7, 3), (6, 7, 1)]:
        for order in ["C", "F"]:
            base = np.array(
                4 * np.arange(np.prod(shape), dtype=np.uint16).reshape(shape), order=order
            )
            for count in [1, 2]:
                expected = base
                errors = []
                for actual in tinybrain.accelerated.average_pooling_2x2(base, count):
                    expected = _average_pool_2x2_yxc(expected)
                    errors.append(int(np.count_nonzero(actual != expected)))
                print(shape, order, "reductions", count, "mismatched pixels", errors)
    base = np.zeros((4, 4, 1), dtype=np.uint8, order="F")
    base[::2, ::2, 0] = [[2, 2], [2, 10]]
    native = tinybrain.accelerated.average_pooling_2x2
    direct = native(base, 2)
    repeated = native(native(base, 1)[0], 1)[0]
    reference = _average_pool_2x2_yxc(_average_pool_2x2_yxc(base))
    print(
        "integer multilevel",
        direct[-1].ravel(),
        "repeated native",
        repeated.ravel(),
        "reference",
        reference.ravel(),
    )


def worker(args):
    __import__("tinybrain.accelerated")  # exclude module startup consistently for all methods

    h, w, c = args.shape
    # Populate in strips so input creation does not inflate the peak with a full copy.
    if args.layout == "strided":
        storage = np.empty((h, w * 2, c), dtype=args.dtype)
        base = storage[:, ::2]
    else:
        base = np.empty((h, w, c), dtype=args.dtype, order=args.layout)
    rng = np.random.default_rng(42)
    for y in range(0, h, 64):
        shape = base[y : y + 64].shape
        if base.dtype.kind == "f":
            base[y : y + 64] = rng.random(shape, dtype=np.float32)
        else:
            base[y : y + 64] = rng.integers(
                0, np.iinfo(base.dtype).max + 1, shape, dtype=base.dtype
            )
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    start = time.perf_counter()
    if args.method == "prepare":
        from wsi_pipeline.omezarr.pyramid import _TINYBRAIN_TILE_SIDE

        side = _TINYBRAIN_TILE_SIDE
        transpose = abs(base.strides[1]) < abs(base.strides[0])
        for channel in range(c):
            for y in range(0, h - h % 2, side):
                for x in range(0, w - w % 2, side):
                    plane = base[
                        y : min(y + side, h - h % 2), x : min(x + side, w - w % 2), channel
                    ]
                    work = np.array(plane.T if transpose else plane, dtype=np.float32, order="F")
                    del work
        mips = [base]
    elif args.method == "original":
        mips = original(base, args.levels)
    elif args.method == "numpy":
        mips = [base]
        for _ in range(args.levels - 1):
            mips.append(_average_pool_2x2_yxc(mips[-1]))
    else:
        mips = build_mips_from_yxc(base, args.levels)
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        json.dumps(
            {
                "seconds": elapsed,
                "peak_mib": peak,
                "extra_mib": peak - before,
                "checksum": float(mips[-1].sum()),
                "tinybrain": version("tinybrain"),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=[8192, 6144, 3])
    parser.add_argument("--dtype", default="uint16", choices=["uint8", "uint16", "float32"])
    parser.add_argument("--layouts", nargs="+", default=["C", "F", "strided"])
    parser.add_argument("--mips", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["original", "numpy", "hybrid"],
        choices=["original", "numpy", "hybrid", "prepare"],
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method")
    parser.add_argument("--layout")
    parser.add_argument("--levels", type=int)
    args = parser.parse_args()
    if args.probe:
        probe()
        return
    if args.worker:
        worker(args)
        return
    for layout in args.layouts:
        for levels in args.mips:
            for method in args.methods:
                if method == "original" and (
                    layout != "F" or any(size % (2 ** (levels - 1)) for size in args.shape[:2])
                ):
                    continue
                runs = []
                for _ in range(args.repeats):
                    command = [
                        sys.executable,
                        __file__,
                        "--worker",
                        "--shape",
                        *map(str, args.shape),
                        "--dtype",
                        args.dtype,
                        "--method",
                        method,
                        "--layout",
                        layout,
                        "--levels",
                        str(levels),
                    ]
                    runs.append(json.loads(subprocess.check_output(command, text=True)))
                print(
                    json.dumps(
                        {
                            "shape": args.shape,
                            "dtype": args.dtype,
                            "layout": layout,
                            "levels": levels,
                            "method": method,
                            "seconds": round(statistics.median(r["seconds"] for r in runs), 4),
                            "peak_mib": round(max(r["peak_mib"] for r in runs), 1),
                            "extra_mib": round(max(r["extra_mib"] for r in runs), 1),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
