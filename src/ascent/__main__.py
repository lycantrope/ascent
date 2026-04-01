import os
import sys

import torch.multiprocessing as mp

import ascent.cli


def cleanup():
    """Manual cleanup of shared memory objects owned by the current user."""
    # This is a fallback to try and unlink semaphores on exit
    try:
        from multiprocessing import resource_tracker

        # This helps the resource tracker finalize its accounting
        resource_tracker._resource_tracker._stop()  # type: ignore
    except Exception:
        pass


if __name__ == "__main__":
    # 1. CRITICAL: Set sharing strategy BEFORE any other imports or logic
    # 'file_system' is much more stable on HPC clusters than 'file_descriptor'
    try:
        mp.set_sharing_strategy("file_system")
    except Exception as e:
        print(f"Warning: Could not set sharing strategy: {e}")

    # 2. Run the actual application
    try:
        ascent.cli.main()
    except Exception as e:
        print(f"\n[ERROR] Ascent crashed: {e}")
        # Exit with error code so the HPC job controller knows it failed
        sys.exit(1)
    finally:
        # 3. Always attempt to clean up workers
        cleanup()
        print("Process finished. Resources released.")
