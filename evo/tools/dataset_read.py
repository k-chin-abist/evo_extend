#!/usr/bin/env python3
"""
Dataset Reader for EVO
Provides unified interface to read KITTI, TUM, and EuRoC datasets
"""

from pathlib import Path
import numpy as np
from typing import Tuple, List, Optional


class KITTIDataset:
    """KITTI Odometry Dataset Reader"""
    
    def __init__(self, sequence_id: str, root_path: Optional[str] = None, use_color: bool = False):
        """
        Initialize KITTI dataset reader
        
        Actual KITTI Structure:
        root_path/
            ├── data_odometry_gray/dataset/sequences/
            │   ├── 00/
            │   │   ├── image_0/
            │   │   ├── image_1/
            │   │   └── times.txt
            │   ├── 01/
            │   └── ...
            ├── data_odometry_poses/dataset/poses/
            │   ├── 00.txt
            │   ├── 01.txt
            │   └── ...
            └── data_odometry_calib/dataset/sequences/
                ├── 00/
                │   └── calib.txt
                └── ...
        
        Args:
            sequence_id: Sequence ID (e.g., "00", "01", ..., "10")
            root_path: Path to KITTI root directory. If None, uses dataset_settings config
            use_color: Use color images if True, grayscale if False (not used yet)
        """
        self.sequence_id = str(sequence_id).zfill(2)  # Ensure 2-digit format
        self.use_color = use_color
        
        # Get root path from config if not provided
        if root_path is None:
            try:
                from evo.tools.dataset_config import DatasetConfig
                config = DatasetConfig()
                self.root_path = config.get_root('kitti')
            except Exception as e:
                raise ValueError(
                    f"No root_path provided and failed to load from config: {e}\n"
                    "Either provide root_path or configure it using dataset_settings"
                )
        else:
            self.root_path = Path(root_path)
        
        # Validate root path
        if not self.root_path.exists():
            raise FileNotFoundError(f"Root path not found: {self.root_path}")
        
        # Setup paths based on actual KITTI structure
        image_folder = "data_odometry_color" if use_color else "data_odometry_gray"
        
        self.sequence_dir = self.root_path / image_folder / "dataset" / "sequences" / self.sequence_id
        self.poses_file = self.root_path / "data_odometry_poses" / "dataset" / "poses" / f"{self.sequence_id}.txt"
        self.calib_file = self.root_path / "data_odometry_calib" / "dataset" / "sequences" / self.sequence_id / "calib.txt"
        
        # Image directories
        self.image_0_dir = self.sequence_dir / "image_0"  # Left camera
        self.image_1_dir = self.sequence_dir / "image_1"  # Right camera
        
        # Times file (if exists)
        self.times_file = self.sequence_dir / "times.txt"
        
        # Velodyne (optional)
        self.velodyne_dir = self.root_path / "data_odometry_velodyne" / "dataset" / "sequences" / self.sequence_id / "velodyne"
    
    def read_poses(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read ground truth poses
        
        Returns:
            timestamps: (N,) array of timestamps
            poses: (N, 3, 4) array of SE(3) poses [R|t]
        """
        if not self.poses_file.exists():
            raise FileNotFoundError(f"Poses file not found: {self.poses_file}")
        
        # Read poses (each line: 12 values representing 3x4 matrix)
        data = np.loadtxt(self.poses_file)
        
        # Reshape to (N, 3, 4)
        poses = data.reshape(-1, 3, 4)
        
        # Read timestamps if available
        if self.times_file.exists():
            timestamps = np.loadtxt(self.times_file)
        else:
            # Generate synthetic timestamps (assuming 10 Hz)
            timestamps = np.arange(len(poses)) * 0.1
        
        return timestamps, poses
    
    def read_calib(self) -> dict:
        """
        Read calibration data
        
        Returns:
            dict: Calibration parameters (P0, P1, P2, P3, Tr)
        """
        if not self.calib_file.exists():
            raise FileNotFoundError(f"Calib file not found: {self.calib_file}")
        
        calib = {}
        with open(self.calib_file, 'r') as f:
            for line in f:
                if line.strip():
                    key, values = line.split(':', 1)
                    calib[key.strip()] = np.fromstring(values, sep=' ')
        
        return calib
    
    def get_left_images(self) -> List[Path]:
        """
        Get list of left camera image files (image_0)
        
        Returns:
            List of image file paths
        """
        if not self.image_0_dir.exists():
            return []
        
        return sorted(self.image_0_dir.glob("*.png"))
    
    def get_right_images(self) -> List[Path]:
        """
        Get list of right camera image files (image_1)
        
        Returns:
            List of image file paths
        """
        if not self.image_1_dir.exists():
            return []
        
        return sorted(self.image_1_dir.glob("*.png"))
    
    def get_velodyne_files(self) -> List[Path]:
        """
        Get list of velodyne point cloud files
        
        Returns:
            List of velodyne file paths (.bin files)
        """
        if not self.velodyne_dir.exists():
            return []
        
        return sorted(self.velodyne_dir.glob("*.bin"))
    
    def read_velodyne(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Read velodyne point cloud for specific frame
        
        Args:
            frame_id: Frame index
            
        Returns:
            (N, 4) array of points [x, y, z, intensity] or None if not available
        """
        velodyne_file = self.velodyne_dir / f"{frame_id:06d}.bin"
        
        if not velodyne_file.exists():
            return None
        
        # Read binary point cloud
        points = np.fromfile(velodyne_file, dtype=np.float32)
        return points.reshape(-1, 4)  # x, y, z, intensity
    
    def __len__(self) -> int:
        """Return number of poses"""
        if self.poses_file.exists():
            return sum(1 for _ in open(self.poses_file))
        return 0
    
    def __repr__(self) -> str:
        color_str = "color" if self.use_color else "gray"
        return f"KITTIDataset(sequence={self.sequence_id}, frames={len(self)}, mode={color_str})"


class TUMDataset:
    """TUM RGB-D Dataset Reader"""
    
    def __init__(self, sequence_id: str, root_path: Optional[str] = None):
        """
        Initialize TUM dataset reader
        
        Actual TUM Structure:
        root_path/
            ├── rgbd_dataset_freiburg2_xyz/              ← sequence directory
            │   ├── rgb/                                 ← RGB images folder
            │   │   ├── 1311868164.363181.png
            │   │   └── ...
            │   ├── depth/                               ← Depth images folder
            │   │   ├── 1311868164.363181.png
            │   │   └── ...
            │   ├── rgb.txt                              ← RGB image list
            │   ├── depth.txt                            ← Depth image list
            │   ├── groundtruth.txt                      ← Ground truth poses
            │   └── accelerometer.txt                    ← Accelerometer data
            └── ...
        
        Args:
            sequence_id: Sequence name (e.g., "rgbd_dataset_freiburg2_xyz")
            root_path: Path to TUM root directory. If None, uses dataset_settings config
        """
        self.sequence_id = sequence_id
        
        # Get root path from config if not provided
        if root_path is None:
            try:
                from evo.tools.dataset_config import DatasetConfig
                config = DatasetConfig()
                root = config.get_root('tum')
                self.sequence_path = root / sequence_id
            except Exception as e:
                raise ValueError(
                    f"No root_path provided and failed to load from config: {e}\n"
                    "Either provide root_path or configure it using dataset_settings"
                )
        else:
            self.sequence_path = Path(root_path) / sequence_id
        
        # Validate paths
        if not self.sequence_path.exists():
            raise FileNotFoundError(f"Sequence path not found: {self.sequence_path}")
        
        # Setup file paths
        self.groundtruth_file = self.sequence_path / "groundtruth.txt"
        self.rgb_file = self.sequence_path / "rgb.txt"
        self.depth_file = self.sequence_path / "depth.txt"
        self.accelerometer_file = self.sequence_path / "accelerometer.txt"
        
        # Setup directory paths
        self.rgb_dir = self.sequence_path / "rgb"
        self.depth_dir = self.sequence_path / "depth"
    
    @staticmethod
    def list_sequences(root_path: Optional[str] = None) -> List[str]:
        """
        List all available TUM sequences in the root directory
        
        Args:
            root_path: Path to TUM root directory. If None, uses dataset_settings config
            
        Returns:
            List of sequence names
        """
        # Get root path from config if not provided
        if root_path is None:
            try:
                from evo.tools.dataset_config import DatasetConfig
                config = DatasetConfig()
                root = config.get_root('tum')
            except Exception as e:
                raise ValueError(
                    f"No root_path provided and failed to load from config: {e}\n"
                    "Either provide root_path or configure it using dataset_settings"
                )
        else:
            root = Path(root_path)
        
        if not root.exists():
            raise FileNotFoundError(f"Root path not found: {root}")
        
        # Find all directories that contain groundtruth.txt (typical TUM sequence)
        sequences = []
        for item in root.iterdir():
            if item.is_dir():
                # Check if it's a valid TUM sequence (has groundtruth.txt or rgb.txt)
                if (item / "groundtruth.txt").exists() or (item / "rgb.txt").exists():
                    sequences.append(item.name)
        
        return sorted(sequences)
    
    def read_poses(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read ground truth poses in TUM format
        
        TUM format: timestamp tx ty tz qx qy qz qw
        
        Returns:
            timestamps: (N,) array of timestamps
            poses: (N, 7) array [tx, ty, tz, qx, qy, qz, qw]
        """
        if not self.groundtruth_file.exists():
            raise FileNotFoundError(f"Groundtruth file not found: {self.groundtruth_file}")
        
        data = []
        with open(self.groundtruth_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    values = [float(x) for x in line.split()]
                    data.append(values)
        
        data = np.array(data)
        timestamps = data[:, 0]
        poses = data[:, 1:8]  # tx, ty, tz, qx, qy, qz, qw
        
        return timestamps, poses
    
    def read_rgb_timestamps(self) -> Tuple[np.ndarray, List[str]]:
        """
        Read RGB image timestamps and filenames
        
        Format in rgb.txt:
        # timestamp filename
        1311868164.363181 rgb/1311868164.363181.png
        
        Returns:
            timestamps: (N,) array of timestamps
            filenames: List of RGB image filenames (relative to sequence path)
        """
        if not self.rgb_file.exists():
            raise FileNotFoundError(f"RGB file not found: {self.rgb_file}")
        
        timestamps = []
        filenames = []
        
        with open(self.rgb_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    timestamps.append(float(parts[0]))
                    filenames.append(parts[1])  # e.g., "rgb/1311868164.363181.png"
        
        return np.array(timestamps), filenames
    
    def get_rgb_images(self) -> List[Path]:
        """
        Get list of RGB image file paths
        
        Returns:
            List of RGB image file paths
        """
        if not self.rgb_dir.exists():
            return []
        
        return sorted(self.rgb_dir.glob("*.png"))
    
    def read_depth_timestamps(self) -> Tuple[np.ndarray, List[str]]:
        """
        Read depth image timestamps and filenames
        
        Format in depth.txt:
        # timestamp filename
        1311868164.363181 depth/1311868164.363181.png
        
        Returns:
            timestamps: (N,) array of timestamps
            filenames: List of depth image filenames (relative to sequence path)
        """
        if not self.depth_file.exists():
            raise FileNotFoundError(f"Depth file not found: {self.depth_file}")
        
        timestamps = []
        filenames = []
        
        with open(self.depth_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    timestamps.append(float(parts[0]))
                    filenames.append(parts[1])  # e.g., "depth/1311868164.363181.png"
        
        return np.array(timestamps), filenames
    
    def get_depth_images(self) -> List[Path]:
        """
        Get list of depth image file paths
        
        Returns:
            List of depth image file paths
        """
        if not self.depth_dir.exists():
            return []
        
        return sorted(self.depth_dir.glob("*.png"))
    
    def read_accelerometer(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read accelerometer data
        
        Returns:
            timestamps: (N,) array of timestamps
            accel: (N, 3) array of accelerometer data [ax, ay, az]
        """
        if not self.accelerometer_file.exists():
            raise FileNotFoundError(f"Accelerometer file not found: {self.accelerometer_file}")
        
        timestamps = []
        accel = []
        
        with open(self.accelerometer_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    values = [float(x) for x in line.split()]
                    timestamps.append(values[0])
                    accel.append(values[1:4])  # ax, ay, az
        
        return np.array(timestamps), np.array(accel)
    
    def __len__(self) -> int:
        """Return number of poses"""
        if self.groundtruth_file.exists():
            count = 0
            with open(self.groundtruth_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        count += 1
            return count
        return 0
    
    def __repr__(self) -> str:
        return f"TUMDataset(sequence={self.sequence_id}, frames={len(self)})"


class EuRoCDataset:
    """EuRoC MAV Dataset Reader"""
    
    def __init__(self, sequence_id: str, root_path: Optional[str] = None):
        """
        Initialize EuRoC dataset reader
        
        Actual EuRoC Structure:
        root_path/
            ├── MH_01_easy/                              ← sequence directory
            │   └── mav0/
            │       ├── cam0/
            │       │   ├── data/                        ← camera 0 images (.png)
            │       │   └── data.csv                     ← timestamp + filename
            │       ├── cam1/
            │       │   ├── data/
            │       │   └── data.csv
            │       ├── imu0/
            │       │   └── data.csv                     ← IMU data
            │       ├── leica0/
            │       │   └── data.csv                     ← Leica position data
            │       └── state_groundtruth_estimate0/
            │           └── data.csv                     ← Ground truth poses
            └── ...
        
        Args:
            sequence_id: Sequence name (e.g., "MH_01_easy", "V1_02_medium")
            root_path: Path to EuRoC root directory. If None, uses dataset_settings config
        """
        self.sequence_id = sequence_id
        
        # Get root path from config if not provided
        if root_path is None:
            try:
                from evo.tools.dataset_config import DatasetConfig
                config = DatasetConfig()
                root = config.get_root('euroc')
                self.sequence_path = root / sequence_id
            except Exception as e:
                raise ValueError(
                    f"No root_path provided and failed to load from config: {e}\n"
                    "Either provide root_path or configure it using dataset_settings"
                )
        else:
            self.sequence_path = Path(root_path) / sequence_id
        
        # Validate paths
        if not self.sequence_path.exists():
            raise FileNotFoundError(f"Sequence path not found: {self.sequence_path}")
        
        self.mav0_dir = self.sequence_path / "mav0"
        
        if not self.mav0_dir.exists():
            raise FileNotFoundError(f"mav0 directory not found: {self.mav0_dir}")
        
        # Setup paths
        self.cam0_dir = self.mav0_dir / "cam0"
        self.cam1_dir = self.mav0_dir / "cam1"
        self.imu_dir = self.mav0_dir / "imu0"
        self.leica_dir = self.mav0_dir / "leica0"
        self.groundtruth_dir = self.mav0_dir / "state_groundtruth_estimate0"
        
        # Data files
        self.groundtruth_file = self.groundtruth_dir / "data.csv"
        self.cam0_csv = self.cam0_dir / "data.csv"
        self.cam1_csv = self.cam1_dir / "data.csv"
        self.imu_file = self.imu_dir / "data.csv"
        self.leica_file = self.leica_dir / "data.csv"
        
        # Image directories
        self.cam0_data = self.cam0_dir / "data"
        self.cam1_data = self.cam1_dir / "data"
    
    @staticmethod
    def list_sequences(root_path: Optional[str] = None) -> List[str]:
        """
        List all available EuRoC sequences in the root directory
        
        Args:
            root_path: Path to EuRoC root directory. If None, uses dataset_settings config
            
        Returns:
            List of sequence names
        """
        # Get root path from config if not provided
        if root_path is None:
            try:
                from evo.tools.dataset_config import DatasetConfig
                config = DatasetConfig()
                root = config.get_root('euroc')
            except Exception as e:
                raise ValueError(
                    f"No root_path provided and failed to load from config: {e}\n"
                    "Either provide root_path or configure it using dataset_settings"
                )
        else:
            root = Path(root_path)
        
        if not root.exists():
            raise FileNotFoundError(f"Root path not found: {root}")
        
        # Find all directories that contain mav0 (typical EuRoC sequence)
        sequences = []
        for item in root.iterdir():
            if item.is_dir() and (item / "mav0").exists():
                sequences.append(item.name)
        
        return sorted(sequences)
    
    def read_poses(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read ground truth poses from EuRoC format
        
        EuRoC format (CSV): timestamp, p_x, p_y, p_z, q_w, q_x, q_y, q_z, ...
        
        Returns:
            timestamps: (N,) array of timestamps (in seconds)
            poses: (N, 7) array [tx, ty, tz, qx, qy, qz, qw]
        """
        if not self.groundtruth_file.exists():
            raise FileNotFoundError(f"Groundtruth file not found: {self.groundtruth_file}")
        
        data = []
        with open(self.groundtruth_file, 'r') as f:
            # Skip header
            next(f)
            
            for line in f:
                line = line.strip()
                if line:
                    values = [float(x) for x in line.split(',')]
                    # Extract: timestamp, px, py, pz, qw, qx, qy, qz
                    timestamp = values[0] / 1e9  # Convert nanoseconds to seconds
                    tx, ty, tz = values[1], values[2], values[3]
                    qw, qx, qy, qz = values[4], values[5], values[6], values[7]
                    
                    # Store as [timestamp, tx, ty, tz, qx, qy, qz, qw]
                    data.append([timestamp, tx, ty, tz, qx, qy, qz, qw])
        
        data = np.array(data)
        timestamps = data[:, 0]
        poses = data[:, 1:8]  # tx, ty, tz, qx, qy, qz, qw
        
        return timestamps, poses
    
    def read_cam0_timestamps(self) -> Tuple[np.ndarray, List[str]]:
        """
        Read camera 0 timestamps and image filenames
        
        Format in data.csv:
        #timestamp [ns],filename
        1403636579763555584,1403636579763555584.png
        
        Returns:
            timestamps: (N,) array of timestamps (in seconds)
            filenames: List of image filenames
        """
        if not self.cam0_csv.exists():
            raise FileNotFoundError(f"Cam0 data file not found: {self.cam0_csv}")
        
        timestamps = []
        filenames = []
        
        with open(self.cam0_csv, 'r') as f:
            # Skip header
            next(f)
            
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    timestamp = int(parts[0]) / 1e9  # Convert nanoseconds to seconds
                    filename = parts[1]
                    timestamps.append(timestamp)
                    filenames.append(filename)
        
        return np.array(timestamps), filenames
    
    def read_cam1_timestamps(self) -> Tuple[np.ndarray, List[str]]:
        """
        Read camera 1 timestamps and image filenames
        
        Returns:
            timestamps: (N,) array of timestamps (in seconds)
            filenames: List of image filenames
        """
        if not self.cam1_csv.exists():
            raise FileNotFoundError(f"Cam1 data file not found: {self.cam1_csv}")
        
        timestamps = []
        filenames = []
        
        with open(self.cam1_csv, 'r') as f:
            # Skip header
            next(f)
            
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    timestamp = int(parts[0]) / 1e9  # Convert to seconds
                    filename = parts[1]
                    timestamps.append(timestamp)
                    filenames.append(filename)
        
        return np.array(timestamps), filenames
    
    def get_cam0_images(self) -> List[Path]:
        """
        Get list of camera 0 image file paths
        
        Returns:
            List of camera 0 image file paths
        """
        if not self.cam0_data.exists():
            return []
        
        return sorted(self.cam0_data.glob("*.png"))
    
    def get_cam1_images(self) -> List[Path]:
        """
        Get list of camera 1 image file paths
        
        Returns:
            List of camera 1 image file paths
        """
        if not self.cam1_data.exists():
            return []
        
        return sorted(self.cam1_data.glob("*.png"))
    
    def read_imu_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read IMU data
        
        Format in data.csv:
        #timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
        
        Returns:
            timestamps: (N,) array of timestamps (in seconds)
            gyro: (N, 3) array of gyroscope data [wx, wy, wz]
            accel: (N, 3) array of accelerometer data [ax, ay, az]
        """
        if not self.imu_file.exists():
            raise FileNotFoundError(f"IMU file not found: {self.imu_file}")
        
        timestamps = []
        gyro = []
        accel = []
        
        with open(self.imu_file, 'r') as f:
            # Skip header
            next(f)
            
            for line in f:
                line = line.strip()
                if line:
                    values = [float(x) for x in line.split(',')]
                    timestamp = values[0] / 1e9  # Convert nanoseconds to seconds
                    wx, wy, wz = values[1], values[2], values[3]
                    ax, ay, az = values[4], values[5], values[6]
                    
                    timestamps.append(timestamp)
                    gyro.append([wx, wy, wz])
                    accel.append([ax, ay, az])
        
        return np.array(timestamps), np.array(gyro), np.array(accel)
    
    def read_leica_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read Leica position data
        
        Format in data.csv:
        #timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m]
        
        Returns:
            timestamps: (N,) array of timestamps (in seconds)
            positions: (N, 3) array of positions [x, y, z]
        """
        if not self.leica_file.exists():
            raise FileNotFoundError(f"Leica file not found: {self.leica_file}")
        
        timestamps = []
        positions = []
        
        with open(self.leica_file, 'r') as f:
            # Skip header
            next(f)
            
            for line in f:
                line = line.strip()
                if line:
                    values = [float(x) for x in line.split(',')]
                    timestamp = values[0] / 1e9  # Convert nanoseconds to seconds
                    x, y, z = values[1], values[2], values[3]
                    
                    timestamps.append(timestamp)
                    positions.append([x, y, z])
        
        return np.array(timestamps), np.array(positions)
    
    def __len__(self) -> int:
        """Return number of poses"""
        if self.groundtruth_file.exists():
            with open(self.groundtruth_file, 'r') as f:
                # Subtract 1 for header
                return sum(1 for _ in f) - 1
        return 0
    
    def __repr__(self) -> str:
        return f"EuRoCDataset(sequence={self.sequence_id}, frames={len(self)})"


def load_dataset(dataset_name: str, **kwargs):
    """
    Factory function to load appropriate dataset reader
    
    Args:
        dataset_name: Dataset type ('kitti', 'tum', or 'euroc')
        **kwargs: Arguments for specific dataset class
            For KITTI: root_path, sequence_id, use_color=False
            For TUM: sequence_path
            For EuRoC: sequence_path
        
    Returns:
        Dataset reader instance
    
    Examples:
        >>> # KITTI
        >>> dataset = load_dataset('kitti', root_path='/path/to/KITTI', sequence_id='00')
        >>> 
        >>> # TUM
        >>> dataset = load_dataset('tum', sequence_path='/path/to/TUM/freiburg1_xyz')
        >>> 
        >>> # EuRoC
        >>> dataset = load_dataset('euroc', sequence_path='/path/to/EuRoC/MH_01_easy')
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'kitti':
        return KITTIDataset(**kwargs)
    elif dataset_name == 'tum':
        return TUMDataset(**kwargs)
    elif dataset_name == 'euroc':
        return EuRoCDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: kitti, tum, euroc")


if __name__ == "__main__":
    # Example usage
    print("Dataset Reader Examples:")
    print("=" * 60)
    
    print("\nMethod 1: Use dataset_settings (Recommended)")
    print("-" * 60)
    print("# First, configure root path once:")
    print("from dataset_settings import DatasetConfig")
    print("config = DatasetConfig()")
    print("config.set_root('kitti', '/path/to/KITTI')")
    print()
    print("# Then simply use sequence_id:")
    print("from dataset_reader import KITTIDataset")
    print("dataset = KITTIDataset(sequence_id='00')  # Auto loads from config")
    print("timestamps, poses = dataset.read_poses()")
    
    print("\n" + "-" * 60)
    print("\nMethod 2: Provide root_path directly")
    print("-" * 60)
    
    # KITTI
    print("\n1. KITTI Dataset:")
    print("   dataset = KITTIDataset(")
    print("       sequence_id='00',")
    print("       root_path='/path/to/KITTI',  # Optional")
    print("       use_color=False")
    print("   )")
    print("   timestamps, poses = dataset.read_poses()")
    print("   calib = dataset.read_calib()")
    print("   left_images = dataset.get_left_images()")
    print("   velodyne_points = dataset.read_velodyne(frame_id=0)")
   
    
    print("\n" + "=" * 60)