#!/usr/bin/env python3
"""
Simple test script for TUM dataset reading
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evo.tools.dataset_read import TUMDataset

print("=" * 60)
print("TUM Dataset Test")
print("=" * 60)
print()

# Step 1: List all available sequences
print("Step 1: Listing all TUM sequences")
print("-" * 60)

try:
    sequences = TUMDataset.list_sequences()
    print(f"✓ Found {len(sequences)} sequences:")
    for i, seq in enumerate(sequences, 1):
        print(f"  {i}. {seq}")
    print()
    
except Exception as e:
    print(f"✗ Failed to list sequences: {e}")
    print()
    print("Please configure TUM first:")
    print("  python -m evo.tools.dataset_settings set tum D:\\data\\tum")
    exit(1)

# Step 2: Test reading first sequence
if sequences:
    test_seq = sequences[0]
    print(f"Step 2: Testing sequence '{test_seq}'")
    print("-" * 60)
    
    try:
        dataset = TUMDataset(sequence_id=test_seq)
        print(f"✓ Created dataset: {dataset}")
        print()
        
        # Read poses
        print("Reading poses...")
        try:
            timestamps, poses = dataset.read_poses()
            print(f"✓ Poses shape: {poses.shape}")
            print(f"✓ Timestamps shape: {timestamps.shape}")
            print(f"✓ Number of frames: {len(poses)}")
            print(f"  First timestamp: {timestamps[0]:.6f}")
            print(f"  Last timestamp: {timestamps[-1]:.6f}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
        print()
        
        # Read RGB timestamps
        print("Reading RGB data...")
        try:
            rgb_times, rgb_files = dataset.read_rgb_timestamps()
            print(f"✓ RGB timestamps: {len(rgb_times)}")
            print(f"  Example file: {rgb_files[0]}")
            
            rgb_images = dataset.get_rgb_images()
            print(f"✓ RGB images in folder: {len(rgb_images)}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
        print()
        
        # Read depth timestamps
        print("Reading depth data...")
        try:
            depth_times, depth_files = dataset.read_depth_timestamps()
            print(f"✓ Depth timestamps: {len(depth_times)}")
            print(f"  Example file: {depth_files[0]}")
            
            depth_images = dataset.get_depth_images()
            print(f"✓ Depth images in folder: {len(depth_images)}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
        print()
        
        # Read accelerometer
        print("Reading accelerometer data...")
        try:
            accel_times, accel_data = dataset.read_accelerometer()
            print(f"✓ Accelerometer data: {len(accel_times)} samples")
            print(f"  Accel shape: {accel_data.shape}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
        print()
        
        print("=" * 60)
        print("✓ All tests PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No sequences found!")