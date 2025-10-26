"""
SLAM System Interface Template

Your SLAM needs to implement this interface
"""

from pathlib import Path
import numpy as np
from typing import List, Tuple


class SLAMSystem:
    """
    Base SLAM system class
    
    Your SLAM needs to inherit this class and implement all methods
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize SLAM system
        
        Args:
            config_file: Configuration file path (optional)
        """
        pass
    
    def initialize(self):
        """
        Initialize SLAM (called before processing first frame)
        """
        pass
    
    def process_frame(self, 
                     timestamp: float,
                     rgb_image: np.ndarray = None,
                     depth_image: np.ndarray = None,
                     left_image: np.ndarray = None,
                     right_image: np.ndarray = None) -> bool:
        """
        Process one frame
        
        Args:
            timestamp: Timestamp (seconds)
            rgb_image: RGB image (H, W, 3)
            depth_image: Depth image (H, W)
            left_image: Left camera image (H, W) or (H, W, 3)
            right_image: Right camera image (H, W) or (H, W, 3)
            
        Returns:
            Whether processing was successful
        """
        raise NotImplementedError("Please implement process_frame method")
    
    def get_trajectory(self) -> Tuple[List[float], List[np.ndarray]]:
        """
        Get current trajectory estimate
        
        Returns:
            timestamps: List of timestamps
            poses: List of poses, each pose as [tx, ty, tz, qx, qy, qz, qw]
        """
        raise NotImplementedError("Please implement get_trajectory method")
    
    def save_trajectory(self, output_file: str):
        """
        Save trajectory to file (TUM format)
        
        Args:
            output_file: Output file path
        """
        timestamps, poses = self.get_trajectory()
        
        with open(output_file, 'w') as f:
            for t, pose in zip(timestamps, poses):
                # TUM format: timestamp tx ty tz qx qy qz qw
                f.write(f"{t:.6f} {pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} "
                       f"{pose[3]:.6f} {pose[4]:.6f} {pose[5]:.6f} {pose[6]:.6f}\n")
    
    def shutdown(self):
        """
        Shutdown SLAM system (cleanup resources)
        """
        pass


# ============================================================================
# Example implementation: Dummy SLAM (for testing)
# ============================================================================

class DummySLAM(SLAMSystem):
    """
    Dummy SLAM - for testing evaluation framework
    Directly outputs ground truth + random noise
    """
    
    def __init__(self, config_file: str = None):
        self.timestamps = []
        self.poses = []
        self.noise_level = 0.1  # 10cm noise
    
    def initialize(self):
        print("DummySLAM initialized")
    
    def process_frame(self, timestamp, rgb_image=None, depth_image=None, 
                     left_image=None, right_image=None):
        # Simulate processing (actually does nothing)
        # In real SLAM, this would perform feature extraction, tracking, mapping, etc.
        
        # Generate fake pose (should actually come from SLAM estimation)
        # For testing purposes, generate a simple trajectory
        t = timestamp
        pose = np.array([
            t * 0.5 + np.random.normal(0, self.noise_level),  # tx
            t * 0.1 + np.random.normal(0, self.noise_level),  # ty
            0.0 + np.random.normal(0, self.noise_level),      # tz
            0.0, 0.0, 0.0, 1.0  # qx, qy, qz, qw
        ])
        
        self.timestamps.append(timestamp)
        self.poses.append(pose)
        
        return True
    
    def get_trajectory(self):
        return self.timestamps, self.poses
    
    def shutdown(self):
        print(f"DummySLAM finished. Total frames: {len(self.timestamps)}")


# ============================================================================
# Your SLAM implementation example
# ============================================================================

class MySLAM(SLAMSystem):
    """
    This is your SLAM implementation example
    """
    
    def __init__(self, config_file: str = None):
        # Initialize your SLAM
        # e.g., load vocabulary, set parameters, etc.
        self.timestamps = []
        self.poses = []
        
        # Your SLAM state
        self.is_initialized = False
        self.current_pose = np.array([0, 0, 0, 0, 0, 0, 1])  # [tx,ty,tz,qx,qy,qz,qw]
    
    def initialize(self):
        """Initialize SLAM"""
        print("MySLAM: Initializing...")
        # Initialize your tracker, mapper, etc.
        self.is_initialized = True
    
    def process_frame(self, timestamp, rgb_image=None, depth_image=None,
                     left_image=None, right_image=None):
        """
        Process one frame
        
        Implement your SLAM logic here:
        1. Feature extraction
        2. Feature matching/tracking
        3. Pose estimation
        4. Local mapping
        5. Loop closure detection (optional)
        """
        
        # Example: Process RGB-D data
        if rgb_image is not None and depth_image is not None:
            # 1. Feature extraction
            # features = self.extract_features(rgb_image)
            
            # 2. Feature matching
            # matches = self.match_features(features)
            
            # 3. Pose estimation
            # self.current_pose = self.estimate_pose(matches, depth_image)
            
            # This is just demonstration, actual SLAM algorithm needed
            pass
        
        # Example: Process stereo data
        elif left_image is not None and right_image is not None:
            # Process stereo images
            pass
        
        # Save trajectory
        self.timestamps.append(timestamp)
        self.poses.append(self.current_pose.copy())
        
        return True
    
    def get_trajectory(self):
        """Return estimated trajectory"""
        return self.timestamps, self.poses
    
    def shutdown(self):
        """Shutdown SLAM"""
        print(f"MySLAM: Finished. Processed {len(self.timestamps)} frames")


# ============================================================================
# C++ SLAM Python Binding Example
# ============================================================================

class MyCppSLAM(SLAMSystem):
    """
    If your SLAM is written in C++, use pybind11 for binding
    """
    
    def __init__(self, config_file: str = None):
        # Import C++ module
        # import myslam_cpp
        # self.cpp_slam = myslam_cpp.SLAM(config_file)
        
        self.timestamps = []
        self.poses = []
    
    def initialize(self):
        # self.cpp_slam.initialize()
        pass
    
    def process_frame(self, timestamp, rgb_image=None, depth_image=None,
                     left_image=None, right_image=None):
        # Call C++ functions
        # success = self.cpp_slam.process_frame(timestamp, rgb_image, depth_image)
        
        # Get current pose
        # pose = self.cpp_slam.get_current_pose()
        
        # self.timestamps.append(timestamp)
        # self.poses.append(pose)
        
        return True
    
    def get_trajectory(self):
        return self.timestamps, self.poses
    
    def shutdown(self):
        # self.cpp_slam.shutdown()
        pass