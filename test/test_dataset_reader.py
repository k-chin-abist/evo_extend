#!/usr/bin/env python3
"""
Simple test script for KITTI dataset reading
"""

import sys
from pathlib import Path

# Add project root to path
# Assumes structure: evo_extend/test/test_kitti.py and evo_extend/evo/tools/dataset_reader.py
project_root = Path(__file__).parent.parent  # Go up to evo_extend/
sys.path.insert(0, str(project_root))

from evo.tools.dataset_read import KITTIDataset

print("=" * 60)
print("KITTI Dataset Test")
print("=" * 60)
print()

# Test reading sequence 00
print("Testing KITTI sequence 00")
print("-" * 60)

try:
    # KITTIDataset will automatically load config
    dataset = KITTIDataset(sequence_id='00')
    print(f"✓ Created dataset: {dataset}")
    print()
    
    # Read poses
    print("Reading poses...")
    timestamps, poses = dataset.read_poses()
    print(f"✓ Poses shape: {poses.shape}")
    print(f"✓ Timestamps shape: {timestamps.shape}")
    print(f"✓ Number of frames: {len(poses)}")
    print()
    
    # Read calibration
    print("Reading calibration...")
    calib = dataset.read_calib()
    print(f"✓ Calibration keys: {list(calib.keys())}")
    print()
    
    # Check images
    print("Checking images...")
    left_images = dataset.get_left_images()
    right_images = dataset.get_right_images()
    print(f"✓ Left images: {len(left_images)}")
    print(f"✓ Right images: {len(right_images)}")
    print()
    
    print("=" * 60)
    print("✓ All tests PASSED!")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    print()
    print("Make sure KITTI is configured:")
    print("  python dataset_settings.py set kitti /path/to/KITTI")
    import traceback
    traceback.print_exc()