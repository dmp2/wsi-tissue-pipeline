#!/usr/bin/env python3
"""
Test script for SciServer integration modules.

Run with: python test_sciserver_modules.py

This tests the modules without requiring the full wsi-tissue-pipeline
package to be installed. Place this in the same directory as the modules.
"""

import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / "sciserver_modules"))


def test_environment_detection():
    """Test environment detection module."""
    print("\n" + "=" * 50)
    print("Test 1: Environment Detection")
    print("=" * 50)

    # Import directly since we're not in a package
    from sciserver_modules.environment import (
        get_sciserver_token,
        get_sciserver_user,
        is_sciserver_environment,
    )

    is_ss = is_sciserver_environment()
    print(f"  Is SciServer: {is_ss}")

    if is_ss:
        user = get_sciserver_user()
        token = get_sciserver_token()
        print(f"  Username: {user}")
        print(f"  Token available: {token is not None}")
    else:
        print("  (Running locally - SciServer features not available)")

    print("  [OK] Environment detection works")
    return True


def test_storage_config():
    """Test storage configuration module."""
    print("\n" + "=" * 50)
    print("Test 2: Storage Configuration")
    print("=" * 50)

    from sciserver_modules.storage import StorageConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test local config
        config = StorageConfig.for_local(tmpdir)
        config.ensure_directories()

        print(f"  MLFlow dir: {config.mlflow_dir}")
        print(f"  Lineage dir: {config.lineage_dir}")
        print(f"  Data dir: {config.data_dir}")

        # Verify directories were created
        assert config.mlflow_dir.exists(), "MLFlow dir not created"
        assert config.lineage_dir.exists(), "Lineage dir not created"
        assert config.data_dir.exists(), "Data dir not created"

        # Test path helpers
        data_path = config.get_data_path("specimen_001.zarr")
        print(f"  Sample data path: {data_path}")

    print("  [OK] Storage configuration works")
    return True


def test_lineage_tracking():
    """Test lineage tracking module."""
    print("\n" + "=" * 50)
    print("Test 3: Lineage Tracking")
    print("=" * 50)

    from sciserver_lineage import LineageTracker, tracked_run

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = LineageTracker(tmpdir)

        # Test 1: Basic run lifecycle
        print("  Testing run lifecycle...")
        run_id = tracker.start_run(
            job_name="test_job",
            inputs=[{"name": "input.zarr", "namespace": "test://data"}]
        )
        print(f"    Started run: {run_id[:8]}...")

        tracker.complete_run(
            run_id=run_id,
            job_name="test_job",
            outputs=[{"name": "output.zarr"}],
            metrics={"n_tiles": 42, "processing_time": 123.4}
        )
        print("    Completed run")

        # Verify in index
        assert tracker.index["runs"][run_id]["status"] == "COMPLETE"
        print("    [OK] Run lifecycle works")

        # Test 2: Query lineage
        print("  Testing lineage queries...")
        lineage = tracker.get_dataset_lineage("output.zarr")
        assert len(lineage["runs"]) > 0
        print(f"    Found {len(lineage['runs'])} runs for dataset")
        print("    [OK] Lineage queries work")

        # Test 3: Context manager
        print("  Testing tracked_run context manager...")
        with tracked_run(tracker, "context_test", inputs=["test.zarr"]) as ctx:
            ctx.add_output("result.zarr")
            ctx.log_metric("accuracy", 0.95)

        details = tracker.get_run_details(ctx.run_id)
        assert details["status"] == "COMPLETE"
        print("    [OK] Context manager works")

        # Test 4: Export for Marquez
        print("  Testing OpenLineage export...")
        events = tracker.export_for_marquez()
        assert len(events) >= 4  # At least 2 runs * 2 events each
        assert "schemaURL" in events[0]
        print(f"    Exported {len(events)} events")
        print("    [OK] OpenLineage export works")

    print("  [OK] Lineage tracking works")
    return True


def test_mlflow_config():
    """Test MLFlow configuration module."""
    print("\n" + "=" * 50)
    print("Test 4: MLFlow Configuration")
    print("=" * 50)

    try:
        import mlflow
    except ImportError:
        print("  [WARN] MLFlow not installed - skipping")
        return True

    from sciserver_mlflow import SciServerMLFlowConfig, mlflow_run

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test config creation
        config = SciServerMLFlowConfig(
            experiment_name="test-experiment",
            local_fallback_dir=tmpdir
        )

        result = config.get_config()
        print(f"  Tracking URI: {result['tracking_uri']}")
        print(f"  Artifact location: {result['artifact_location']}")
        print(f"  Is SciServer: {result['is_sciserver']}")

        # Test setup
        config.setup()
        print("  [OK] MLFlow setup completed")

        # Test run context
        with mlflow_run("test-run"):
            mlflow.log_param("test_param", "value")
            mlflow.log_metric("test_metric", 42.0)
        print("  [OK] MLFlow run completed")

    print("  [OK] MLFlow configuration works")
    return True


def test_integration():
    """Test full integration module."""
    print("\n" + "=" * 50)
    print("Test 5: Full Integration")
    print("=" * 50)

    from sciserver_integration import SciServerPipeline, StorageConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create pipeline with manual storage config
        pipeline = SciServerPipeline(
            experiment_name="integration-test",
            auto_setup=False
        )

        # Override storage for testing
        pipeline.storage = StorageConfig.for_local(tmpdir)
        pipeline.setup()

        print(f"  Is SciServer: {pipeline.is_sciserver}")
        print(f"  MLFlow URI: {pipeline.mlflow_tracking_uri}")
        print(f"  Storage path: {pipeline.storage.user_volume}")

        # Test tracked experiment
        print("  Testing tracked experiment...")
        with pipeline.tracked_experiment(
            "test-specimen",
            inputs=["input.vsi"],
            tags={"test": "true"}
        ) as exp:
            exp.log_param("chunk_size", 512)
            exp.log_metric("n_tiles", 100)
            exp.log_output("output.ome.zarr")

        print("    [OK] Tracked experiment completed")

        # Verify lineage was recorded
        if pipeline.lineage_tracker:
            history = pipeline.lineage_tracker.get_job_history("test-specimen")
            assert len(history) > 0
            print(f"    Lineage recorded: {len(history)} runs")

    print("  [OK] Full integration works")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("SciServer Integration Module Tests")
    print("=" * 50)

    tests = [
        ("Environment Detection", test_environment_detection),
        ("Storage Configuration", test_storage_config),
        ("Lineage Tracking", test_lineage_tracking),
        ("MLFlow Configuration", test_mlflow_config),
        ("Full Integration", test_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")

    print(f"\nResults: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
