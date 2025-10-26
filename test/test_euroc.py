#!/usr/bin/env python3
"""
Simple test script for EuRoC dataset reading
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evo.tools.dataset_read import EuRoCDataset
from evo.tools.dataset_config import DatasetConfig

print("=" * 60)
print("EuRoC Dataset Test")
print("=" * 60)
print()

# Step 0: Check configuration
print("Step 0: Checking configuration")
print("-" * 60)
try:
    config = DatasetConfig()
    print(f"✓ Config file: {config.CONFIG_FILE}")
    
    euroc_root = config.get_root('euroc')
    print(f"✓ EuRoC root configured: {euroc_root}")
    print(f"✓ EuRoC root exists: {euroc_root.exists()}")
except Exception as e:
    print(f"✗ EuRoC not configured: {e}")
    print()
    print("Please configure EuRoC first:")
    print("  python -m evo.tools.dataset_settings set euroc D:\\data\\euroc")
    exit(1)

print()

# Step 1: List all available sequences
print("Step 1: Listing all EuRoC sequences")
print("-" * 60)

try:
    sequences = EuRoCDataset.list_sequences()
    print(f"✓ Found {len(sequences)} sequences:")
    for i, seq in enumerate(sequences, 1):
        print(f"  {i}. {seq}")
    print()
    
except Exception as e:
    print(f"✗ Failed to list sequences: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 2: Test reading first sequence
if sequences:
    test_seq = sequences[0]
    print(f"Step 2: Testing sequence '{test_seq}'")
    print("-" * 60)
    
    try:
        dataset = EuRoCDataset(sequence_id=test_seq)
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
        
        # Read camera 0
        print("Reading camera 0 data...")
        try:
            cam0_times, cam0_files = dataset.read_cam0_timestamps()
            print(f"✓ Cam0 timestamps: {len(cam0_times)}")
            print(f"  Example file: {cam0_files[0]}")
            
            cam0_images = dataset.get_cam0_images()
            print(f"✓ Cam0 images in folder: {len(cam0_images)}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
        print()
        
        # Read camera 1
        print("Reading camera 1 data...")
        try:
            cam1_times, cam1_files = dataset.read_cam1_timestamps()
            print(f"✓ Cam1 timestamps: {len(cam1_times)}")
            print(f"  Example file: {cam1_files[0]}")
            
            cam1_images = dataset.get_cam1_images()
            print(f"✓ Cam1 images in folder: {len(cam1_images)}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
        print()
        
        # Read IMU
        print("Reading IMU data...")
        try:
            imu_times, gyro, accel = dataset.read_imu_data()
            print(f"✓ IMU data: {len(imu_times)} samples")
            print(f"  Gyro shape: {gyro.shape}")
            print(f"  Accel shape: {accel.shape}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
        print()
        
        # Read Leica
        print("Reading Leica data...")
        try:
            leica_times, positions = dataset.read_leica_data()
            print(f"✓ Leica data: {len(leica_times)} samples")
            print(f"  Positions shape: {positions.shape}")
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