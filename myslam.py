#!/usr/bin/env python3
"""
Test SLAM Implementation for EVO Evaluation Framework

This module provides a simple test SLAM that can be used with the eval.py script.
It generates a trajectory with controlled drift and noise to test the evaluation framework.

Usage:
    python eval.py --slam myslam.MySLAM --dataset kitti --seq 00 --plot
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import time

# Import the SLAM interface
import sys
sys.path.append(str(Path(__file__).parent))
from evo.core.slam_interface import SLAMSystem


class MySLAM(SLAMSystem):
    """
    Test SLAM Implementation
    
    This SLAM generates a trajectory that follows a simple pattern with:
    - Linear motion in X direction
    - Small sinusoidal motion in Y direction  
    - Controlled drift and noise
    - Configurable parameters for testing different scenarios
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize MySLAM
        
        Args:
            config_file: Configuration file path (optional)
        """
        super().__init__(config_file)
        
        # SLAM state
        self.timestamps = []
        self.poses = []
        self.is_initialized = False
        
        # Current pose [tx, ty, tz, qx, qy, qz, qw]
        self.current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        # Configuration parameters
        self.config = {
            'linear_velocity': 0.5,      # m/s in X direction
            'amplitude_y': 0.2,          # Amplitude of Y oscillation
            'frequency_y': 0.1,          # Frequency of Y oscillation
            'drift_rate': 0.01,          # Drift rate per frame
            'noise_level': 0.05,        # Noise level in meters
            'frame_rate': 10.0,          # Assumed frame rate (Hz)
            'enable_drift': True,        # Enable drift simulation
            'enable_noise': True,        # Enable noise simulation
        }
        
        # Load configuration if provided
        if config_file and Path(config_file).exists():
            self._load_config(config_file)
        
        print(f"MySLAM initialized with config: {self.config}")
    
    def _load_config(self, config_file: str):
        """Load configuration from file"""
        try:
            import json
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
                print(f"Loaded config from {config_file}")
        except Exception as e:
            print(f"WARNING: Failed to load config from {config_file}: {e}")
    
    def initialize(self):
        """Initialize SLAM system"""
        print("MySLAM: Initializing...")
        self.is_initialized = True
        self.timestamps = []
        self.poses = []
        self.current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        print("MySLAM: Initialization complete")
    
    def process_frame(self, 
                     timestamp: float,
                     rgb_image: Optional[np.ndarray] = None,
                     depth_image: Optional[np.ndarray] = None,
                     left_image: Optional[np.ndarray] = None,
                     right_image: Optional[np.ndarray] = None) -> bool:
        """
        Process one frame
        
        Args:
            timestamp: Frame timestamp
            rgb_image: RGB image (for RGB-D SLAM)
            depth_image: Depth image (for RGB-D SLAM)
            left_image: Left stereo image
            right_image: Right stereo image
            
        Returns:
            True if processing successful
        """
        if not self.is_initialized:
            print("ERROR: SLAM not initialized!")
            return False
        
        try:
            # Calculate time delta (use frame rate if timestamps are not reliable)
            if len(self.timestamps) > 0:
                dt = timestamp - self.timestamps[-1]
            else:
                dt = 1.0 / self.config['frame_rate']
            
            # Ensure reasonable time delta
            dt = max(0.001, min(dt, 1.0))  # Between 1ms and 1s
            
            # Generate trajectory based on time
            t_total = timestamp
            
            # Linear motion in X direction
            x = self.config['linear_velocity'] * t_total
            
            # Sinusoidal motion in Y direction
            y = self.config['amplitude_y'] * np.sin(2 * np.pi * self.config['frequency_y'] * t_total)
            
            # Small motion in Z direction
            z = 0.1 * np.sin(0.5 * t_total)
            
            # Add drift if enabled
            if self.config['enable_drift']:
                drift_x = self.config['drift_rate'] * t_total
                drift_y = self.config['drift_rate'] * 0.5 * t_total
                x += drift_x
                y += drift_y
            
            # Add noise if enabled
            if self.config['enable_noise']:
                noise_x = np.random.normal(0, self.config['noise_level'])
                noise_y = np.random.normal(0, self.config['noise_level'])
                noise_z = np.random.normal(0, self.config['noise_level'] * 0.5)
                x += noise_x
                y += noise_y
                z += noise_z
            
            # Simple rotation (small roll and pitch)
            roll = 0.05 * np.sin(0.3 * t_total)
            pitch = 0.03 * np.cos(0.2 * t_total)
            yaw = 0.02 * np.sin(0.1 * t_total)
            
            # Convert to quaternion
            qx, qy, qz, qw = self._euler_to_quaternion(roll, pitch, yaw)
            
            # Update current pose
            self.current_pose = np.array([x, y, z, qx, qy, qz, qw])
            
            # Store trajectory
            self.timestamps.append(timestamp)
            self.poses.append(self.current_pose.copy())
            
            # Optional: Print progress every 100 frames
            if len(self.timestamps) % 100 == 0:
                print(f"MySLAM: Processed {len(self.timestamps)} frames, "
                      f"current pose: [{x:.3f}, {y:.3f}, {z:.3f}]")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Frame processing failed: {e}")
            return False
    
    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        """Convert Euler angles to quaternion"""
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw
    
    def get_trajectory(self) -> Tuple[List[float], List[np.ndarray]]:
        """
        Get current trajectory estimate
        
        Returns:
            timestamps: List of timestamps
            poses: List of poses [tx, ty, tz, qx, qy, qz, qw]
        """
        return self.timestamps.copy(), [pose.copy() for pose in self.poses]
    
    def shutdown(self):
        """Shutdown SLAM system"""
        print(f"MySLAM: Shutdown complete. Processed {len(self.timestamps)} frames")
        if len(self.timestamps) > 0:
            total_time = self.timestamps[-1] - self.timestamps[0]
            avg_speed = np.sqrt(self.current_pose[0]**2 + self.current_pose[1]**2) / total_time if total_time > 0 else 0
            print(f"MySLAM: Total distance: {np.sqrt(self.current_pose[0]**2 + self.current_pose[1]**2):.2f}m")
            print(f"MySLAM: Average speed: {avg_speed:.2f}m/s")


class MySLAMAdvanced(SLAMSystem):
    """
    Advanced Test SLAM with more realistic behavior
    
    This SLAM simulates more realistic SLAM behavior with:
    - Feature-based tracking simulation
    - Loop closure detection simulation
    - Scale drift simulation
    - Configurable accuracy levels
    """
    
    def __init__(self, config_file: Optional[str] = None):
        super().__init__(config_file)
        
        self.timestamps = []
        self.poses = []
        self.is_initialized = False
        self.current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        # Advanced configuration
        self.config = {
            'accuracy_level': 'medium',  # 'low', 'medium', 'high'
            'enable_loop_closure': True,
            'scale_drift_rate': 0.001,
            'rotation_drift_rate': 0.0005,
            'feature_tracking_quality': 0.8,
        }
        
        # Accuracy presets
        self.accuracy_presets = {
            'low': {'noise_level': 0.1, 'drift_rate': 0.02, 'tracking_quality': 0.6},
            'medium': {'noise_level': 0.05, 'drift_rate': 0.01, 'tracking_quality': 0.8},
            'high': {'noise_level': 0.02, 'drift_rate': 0.005, 'tracking_quality': 0.95},
        }
        
        if config_file and Path(config_file).exists():
            self._load_config(config_file)
        
        # Apply accuracy preset
        preset = self.accuracy_presets[self.config['accuracy_level']]
        self.config.update(preset)
        
        print(f"MySLAMAdvanced initialized with {self.config['accuracy_level']} accuracy")
    
    def _load_config(self, config_file: str):
        """Load configuration from file"""
        try:
            import json
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
        except Exception as e:
            print(f"WARNING: Failed to load config: {e}")
    
    def initialize(self):
        """Initialize advanced SLAM"""
        print("MySLAMAdvanced: Initializing with feature tracking...")
        self.is_initialized = True
        self.timestamps = []
        self.poses = []
        self.current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        print("MySLAMAdvanced: Feature tracker initialized")
    
    def process_frame(self, timestamp: float, **kwargs) -> bool:
        """Process frame with advanced SLAM simulation"""
        if not self.is_initialized:
            return False
        
        try:
            # Simulate feature tracking quality
            tracking_quality = self.config['tracking_quality']
            if np.random.random() > tracking_quality:
                # Simulate tracking failure
                print(f"WARNING: Feature tracking failed at frame {len(self.timestamps)}")
                return False
            
            # Generate trajectory with more realistic motion
            t_total = timestamp
            
            # Simulate vehicle motion (circular path with drift)
            radius = 10.0
            angular_velocity = 0.1
            
            x = radius * np.cos(angular_velocity * t_total)
            y = radius * np.sin(angular_velocity * t_total)
            z = 0.1 * np.sin(0.5 * t_total)
            
            # Add scale drift
            scale_drift = 1.0 + self.config['scale_drift_rate'] * t_total
            x *= scale_drift
            y *= scale_drift
            
            # Add rotation drift
            rotation_drift = self.config['rotation_drift_rate'] * t_total
            roll = rotation_drift
            pitch = rotation_drift * 0.5
            yaw = angular_velocity * t_total + rotation_drift
            
            # Convert to quaternion
            qx, qy, qz, qw = self._euler_to_quaternion(roll, pitch, yaw)
            
            # Add noise
            noise_x = np.random.normal(0, self.config['noise_level'])
            noise_y = np.random.normal(0, self.config['noise_level'])
            noise_z = np.random.normal(0, self.config['noise_level'] * 0.5)
            
            self.current_pose = np.array([
                x + noise_x, y + noise_y, z + noise_z,
                qx, qy, qz, qw
            ])
            
            # Store trajectory
            self.timestamps.append(timestamp)
            self.poses.append(self.current_pose.copy())
            
            return True
            
        except Exception as e:
            print(f"ERROR: Advanced SLAM processing failed: {e}")
            return False
    
    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        """Convert Euler angles to quaternion"""
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw
    
    def get_trajectory(self) -> Tuple[List[float], List[np.ndarray]]:
        """Get trajectory"""
        return self.timestamps.copy(), [pose.copy() for pose in self.poses]
    
    def shutdown(self):
        """Shutdown advanced SLAM"""
        print(f"MySLAMAdvanced: Shutdown complete. Processed {len(self.timestamps)} frames")


# Example configuration file content
EXAMPLE_CONFIG = {
    "linear_velocity": 0.5,
    "amplitude_y": 0.2,
    "frequency_y": 0.1,
    "drift_rate": 0.01,
    "noise_level": 0.05,
    "frame_rate": 10.0,
    "enable_drift": True,
    "enable_noise": True,
    "accuracy_level": "medium"
}


def create_example_config(filename: str = "myslam_config.json"):
    """Create an example configuration file"""
    import json
    
    with open(filename, 'w') as f:
        json.dump(EXAMPLE_CONFIG, f, indent=2)
    
    print(f"Created example config file: {filename}")
    print("You can modify this file to adjust SLAM behavior")


if __name__ == "__main__":
    # Create example config file
    create_example_config()
    
    print("\n" + "="*60)
    print("MySLAM Test Implementation")
    print("="*60)
    print()
    print("Usage:")
    print("  python eval.py --slam myslam.MySLAM --dataset kitti --seq 00 --plot")
    print("  python eval.py --slam myslam.MySLAMAdvanced --dataset kitti --seq 00 --plot")
    print()
    print("Available SLAM classes:")
    print("  - MySLAM: Simple test SLAM with configurable parameters")
    print("  - MySLAMAdvanced: Advanced SLAM with realistic behavior simulation")
    print()
    print("Configuration:")
    print("  - Create myslam_config.json to customize SLAM behavior")
    print("  - See EXAMPLE_CONFIG in this file for available parameters")
