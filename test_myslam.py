#!/usr/bin/env python3
"""
Test script for MySLAM implementation

This script tests the MySLAM system to ensure it works correctly
before using it with the eval.py framework.
"""

import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from myslam import MySLAM, MySLAMAdvanced


def test_myslam():
    """Test basic MySLAM functionality"""
    print("="*60)
    print("Testing MySLAM")
    print("="*60)
    
    # Create SLAM instance
    slam = MySLAM()
    
    # Initialize
    slam.initialize()
    
    # Process some frames
    print("\nProcessing test frames...")
    for i in range(10):
        timestamp = i * 0.1  # 10 Hz
        success = slam.process_frame(timestamp)
        if not success:
            print(f"ERROR: Frame {i} processing failed")
            return False
    
    # Get trajectory
    timestamps, poses = slam.get_trajectory()
    print(f"\nGenerated trajectory:")
    print(f"  Frames: {len(poses)}")
    print(f"  Duration: {timestamps[-1] - timestamps[0]:.2f}s")
    print(f"  Final position: [{poses[-1][0]:.3f}, {poses[-1][1]:.3f}, {poses[-1][2]:.3f}]")
    
    # Shutdown
    slam.shutdown()
    
    print("SUCCESS: MySLAM test completed successfully!")
    return True


def test_myslam_advanced():
    """Test advanced MySLAM functionality"""
    print("\n" + "="*60)
    print("Testing MySLAMAdvanced")
    print("="*60)
    
    # Create advanced SLAM instance
    slam = MySLAMAdvanced()
    
    # Initialize
    slam.initialize()
    
    # Process some frames
    print("\nProcessing test frames...")
    success_count = 0
    for i in range(20):
        timestamp = i * 0.1  # 10 Hz
        success = slam.process_frame(timestamp)
        if success:
            success_count += 1
    
    print(f"Successfully processed {success_count}/20 frames")
    
    # Get trajectory
    timestamps, poses = slam.get_trajectory()
    print(f"\nGenerated trajectory:")
    print(f"  Frames: {len(poses)}")
    if len(poses) > 0:
        print(f"  Duration: {timestamps[-1] - timestamps[0]:.2f}s")
        print(f"  Final position: [{poses[-1][0]:.3f}, {poses[-1][1]:.3f}, {poses[-1][2]:.3f}]")
    
    # Shutdown
    slam.shutdown()
    
    print("SUCCESS: MySLAMAdvanced test completed successfully!")
    return True


def test_with_config():
    """Test SLAM with configuration file"""
    print("\n" + "="*60)
    print("Testing MySLAM with configuration")
    print("="*60)
    
    # Create SLAM with config
    slam = MySLAM("myslam_config.json")
    
    # Initialize
    slam.initialize()
    
    # Process frames
    print("\nProcessing frames with custom config...")
    for i in range(5):
        timestamp = i * 0.1
        success = slam.process_frame(timestamp)
        if not success:
            print(f"ERROR: Frame {i} processing failed")
            return False
    
    # Get trajectory
    timestamps, poses = slam.get_trajectory()
    print(f"\nTrajectory with custom config:")
    print(f"  Frames: {len(poses)}")
    print(f"  Final position: [{poses[-1][0]:.3f}, {poses[-1][1]:.3f}, {poses[-1][2]:.3f}]")
    
    slam.shutdown()
    
    print("SUCCESS: Configuration test completed successfully!")
    return True


def main():
    """Run all tests"""
    print("MySLAM Test Suite")
    print("="*60)
    
    tests = [
        ("Basic MySLAM", test_myslam),
        ("Advanced MySLAM", test_myslam_advanced),
        ("Configuration Test", test_with_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"ERROR: {test_name} failed")
        except Exception as e:
            print(f"ERROR: {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("SUCCESS: All tests passed! MySLAM is ready to use.")
        print("\nYou can now run:")
        print("  python eval.py --slam myslam.MySLAM --dataset kitti --seq 00 --plot")
        print("  python eval.py --slam myslam.MySLAMAdvanced --dataset kitti --seq 00 --plot")
    else:
        print("ERROR: Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
