#!/usr/bin/env python3
"""
Download Sample Data for WSI Tissue Pipeline

Downloads example whole-slide images for testing the pipeline.
Includes synthetic test data and links to public datasets.

Usage:
    python download_sample_data.py --output ./sample_data
    python download_sample_data.py --dataset tcga --output ./data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Optional

try:
    from wsi_pipeline.demo_data import create_synthetic_wsi
except ImportError:
    src_root = Path(__file__).resolve().parents[1] / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from wsi_pipeline.demo_data import create_synthetic_wsi

# Sample datasets available for download
DATASETS = {
    "synthetic": {
        "description": "Synthetic test images for pipeline validation",
        "files": [
            {
                "name": "synthetic_wsi_001.png",
                "url": None,  # Generated locally
                "size": "~2MB",
                "type": "synthetic",
            },
            {
                "name": "synthetic_wsi_002.png",
                "url": None,
                "size": "~2MB",
                "type": "synthetic",
            },
        ],
    },
    "openslide-testdata": {
        "description": "OpenSlide test data (small SVS files)",
        "files": [
            {
                "name": "CMU-1-Small-Region.svs",
                "url": "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs",
                "size": "1.8MB",
                "md5": "7ce01e9e6f7b5dbced1c4f3b0cd2c3be",
            },
        ],
    },
    "info": {
        "description": "Information about publicly available WSI datasets",
        "datasets": [
            {
                "name": "TCGA (The Cancer Genome Atlas)",
                "url": "https://portal.gdc.cancer.gov/",
                "description": "Large collection of cancer pathology images",
                "access": "Requires GDC Data Portal account",
            },
            {
                "name": "CAMELYON16/17",
                "url": "https://camelyon17.grand-challenge.org/",
                "description": "Breast cancer metastasis detection challenge",
                "access": "Registration required",
            },
            {
                "name": "GTEx",
                "url": "https://gtexportal.org/",
                "description": "Gene-tissue expression project histology images",
                "access": "Open access with registration",
            },
            {
                "name": "ANHIR",
                "url": "https://anhir.grand-challenge.org/",
                "description": "Automatic Non-rigid Histological Image Registration",
                "access": "Open access",
            },
        ],
    },
}

def download_file(url: str, output_path: Path, expected_md5: Optional[str] = None) -> bool:
    """Download a file with progress indication."""
    print(f"  Downloading: {output_path.name}")
    print(f"  URL: {url}")
    
    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print()  # New line after progress
        
        # Verify MD5 if provided
        if expected_md5:
            actual_md5 = hashlib.md5(output_path.read_bytes()).hexdigest()
            if actual_md5 != expected_md5:
                print(f"  Warning: MD5 mismatch! Expected {expected_md5}, got {actual_md5}")
                return False
            print(f"  MD5 verified: {expected_md5}")
        
        print(f"  Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def list_datasets():
    """Print information about available datasets."""
    print("\nAvailable Datasets")
    print("=" * 60)
    
    for name, info in DATASETS.items():
        if name == "info":
            continue
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        if "files" in info:
            print(f"  Files: {len(info['files'])}")
            for f in info["files"]:
                print(f"    - {f['name']} ({f.get('size', 'unknown')})")
    
    print("\n\nPublic WSI Datasets (external):")
    print("-" * 60)
    for dataset in DATASETS["info"]["datasets"]:
        print(f"\n{dataset['name']}:")
        print(f"  URL: {dataset['url']}")
        print(f"  Description: {dataset['description']}")
        print(f"  Access: {dataset['access']}")


def main():
    parser = argparse.ArgumentParser(
        description="Download sample data for WSI Tissue Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./sample_data"),
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=list(DATASETS.keys()) + ["all"],
        default="synthetic",
        help="Dataset to download (default: synthetic)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return 0
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("WSI Tissue Pipeline - Sample Data Download")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"Dataset: {args.dataset}")
    print()
    
    datasets_to_download = (
        [k for k in DATASETS.keys() if k != "info"]
        if args.dataset == "all"
        else [args.dataset]
    )
    
    for dataset_name in datasets_to_download:
        if dataset_name == "info":
            list_datasets()
            continue
        
        dataset = DATASETS[dataset_name]
        print(f"\nDownloading: {dataset_name}")
        print(f"Description: {dataset['description']}")
        print("-" * 40)
        
        dataset_dir = args.output / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        for file_info in dataset.get("files", []):
            output_path = dataset_dir / file_info["name"]
            
            if output_path.exists():
                print(f"  Skipping (exists): {file_info['name']}")
                continue
            
            if file_info.get("type") == "synthetic":
                # Generate synthetic data
                seed = hash(file_info["name"]) % (2**32)
                try:
                    created = create_synthetic_wsi(output_path, seed=seed)
                    print(
                        f"  Created: {created.name} "
                        f"({created.stat().st_size / 1024 / 1024:.1f} MB)"
                    )
                except ImportError:
                    print("  Note: numpy, Pillow, and scipy are required for synthetic data generation")
            elif file_info.get("url"):
                # Download from URL
                success = download_file(
                    file_info["url"],
                    output_path,
                    file_info.get("md5"),
                )
                if not success:
                    print(f"  Failed to download {file_info['name']}")
    
    # Create a README in the output directory
    readme_path = args.output / "README.md"
    readme_content = f"""# Sample Data for WSI Tissue Pipeline

This directory contains sample whole-slide images for testing the pipeline.

## Contents

"""
    for dataset_name in datasets_to_download:
        if dataset_name == "info":
            continue
        dataset = DATASETS[dataset_name]
        readme_content += f"### {dataset_name}\n"
        readme_content += f"{dataset['description']}\n\n"
        for f in dataset.get("files", []):
            readme_content += f"- `{f['name']}`: {f.get('size', 'unknown size')}\n"
        readme_content += "\n"
    
    readme_content += """
## Usage

```bash
# Process all sample images
wsi-pipeline batch -i sample_data/synthetic -o output/ --pattern "*.png"

# Process single image
wsi-pipeline process -i sample_data/synthetic/synthetic_wsi_001.png -o output/
```

## Additional Datasets

For larger datasets, see the public repositories:
- TCGA: https://portal.gdc.cancer.gov/
- CAMELYON: https://camelyon17.grand-challenge.org/
- GTEx: https://gtexportal.org/
"""
    
    readme_path.write_text(readme_content)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Data saved to: {args.output}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
