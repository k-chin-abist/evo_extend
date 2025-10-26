# EVO Extend - SLAM Evaluation Framework

A comprehensive SLAM evaluation framework built on top of [EVO](https://github.com/MichaelGrupp/evo) with automated evaluation pipeline, dataset readers, and SLAM system interfaces.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Support](#dataset-support)
- [SLAM Implementation](#slam-implementation)
- [Evaluation & Visualization](#evaluation--visualization)
- [Project Structure](#project-structure)

## ✨ Features

- **Automated SLAM Evaluation**: Run your SLAM system on standard datasets and automatically evaluate against ground truth
- **Universal SLAM Interface**: Easy-to-implement interface for any SLAM system
- **Multiple Dataset Support**: KITTI, TUM RGB-D, EuRoC MAV
- **Comprehensive Metrics**: APE (Absolute Pose Error) and RPE (Relative Pose Error) evaluation
- **High-Performance Readers**: Python and C++ dataset readers

## 🚀 Installation

### Prerequisites

- Python >= 3.10
- CMake >= 3.10 (for C++ components, optional)
- C++ compiler (gcc, clang, or MSVC) - optional

### Install

```bash
# Clone the repository
git clone <repository-url>
cd evo_extend

# Install dependencies
pip install -e .

# Optional: Install with additional features
pip install -e .[gui,geo,rerun,ros]
```

## ⚡ Quick Start

### 1. Configure Datasets

```bash
# Set dataset root directories
python evo/tools/dataset_config.py --set-root kitti /path/to/kitti/dataset
python evo/tools/dataset_config.py --set-root tum /path/to/tum/dataset
python evo/tools/dataset_config.py --set-root euroc /path/to/euroc/dataset
```

### 2. Implement Your SLAM

Create a SLAM class inheriting from `SLAMSystem`:

```python
from evo.core.slam_interface import SLAMSystem
import numpy as np

class MySLAM(SLAMSystem):
    def __init__(self, config_file: str = None):
        super().__init__(config_file)
        self.timestamps = []
        self.poses = []
    
    def initialize(self):
        print("Initializing SLAM...")
        self.is_initialized = True
    
    def process_frame(self, timestamp, rgb_image=None, depth_image=None,
                      left_image=None, right_image=None) -> bool:
        # Your SLAM processing logic
        pose = self.estimate_pose(rgb_image, depth_image)
        self.timestamps.append(timestamp)
        self.poses.append(pose)
        return True
    
    def get_trajectory(self):
        return self.timestamps, self.poses
    
    def shutdown(self):
        print("SLAM complete")
```

### 3. Run Evaluation

```bash
# Evaluate with plotting
python evo/core/eval.py --slam myslam.MySLAM --dataset kitti --seq 00 --plot

# Evaluate on TUM dataset
python evo/core/eval.py --slam myslam.MySLAM --dataset tum --seq rgbd_dataset_freiburg2_xyz

# Evaluate on EuRoC dataset
python evo/core/eval.py --slam myslam.MySLAM --dataset euroc --seq MH_01_easy
```

## 📊 Dataset Support

### Expected Directory Structures

#### KITTI Dataset
```
kitti/dataset/sequences/00/
├── image_0/          # Left stereo images
├── image_1/          # Right stereo images
└── times.txt         # Timestamps
kitti/dataset/poses/00.txt  # Ground truth
```

#### TUM RGB-D Dataset
```
tum/rgbd_dataset_freiburg2_xyz/
├── rgb.txt           # RGB file list
├── depth.txt         # Depth file list
├── groundtruth.txt   # GT poses
├── rgb/             # RGB images
└── depth/           # Depth images
```

#### EuRoC MAV Dataset
```
euroc/MH_01_easy/mav0/
├── cam0/data/                # Camera images
├── cam0/data.csv             # Image timestamps
├── imu0/data.csv             # IMU data
└── state_groundtruth_estimate0/data.csv  # GT poses
```

### Dataset Readers

**Python API** (`evo/tools/dataset_read.py`):
```python
from evo.tools.dataset_read import KITTIDataset, TUMDataset, EuRoCDataset

# KITTI
kitti = KITTIDataset(sequence="00")
left_imgs = kitti.get_left_images()
right_imgs = kitti.get_right_images()
timestamps, poses = kitti.read_poses()

# TUM RGB-D
tum = TUMDataset(sequence="rgbd_dataset_freiburg2_xyz")
rgb_imgs = tum.get_rgb_images()
depth_imgs = tum.get_depth_images()
timestamps, poses = tum.read_poses()

# EuRoC
euroc = EuRoCDataset(sequence="MH_01_easy")
images = euroc.get_images()
imu = euroc.get_imu_data()
timestamps, poses = euroc.read_poses()
```

## 🔧 SLAM Implementation

### Required Methods

- `__init__(self, config_file)`: Initialize SLAM
- `initialize()`: Called before first frame
- `process_frame(timestamp, rgb_image, depth_image, left_image, right_image)`: Process one frame
- `get_trajectory()`: Return (timestamps, poses)
- `shutdown()`: Called after last frame

### Examples

See `myslam.py` for:
- **MySLAM**: Basic test SLAM with configurable trajectory
- **MySLAMAdvanced**: Advanced SLAM with feature tracking simulation

## 📈 Evaluation & Visualization

### Metrics

- **APE (Absolute Pose Error)**: Overall trajectory error
- **RPE (Relative Pose Error)**: Local relative error

### Command-Line Usage

```bash
# Basic evaluation with plots
python evo/core/eval.py --slam myslam.MySLAM --dataset kitti --seq 00 --plot

# Limit frames for testing
python evo/core/eval.py --slam myslam.MySLAM --dataset kitti --seq 00 --max-frames 100 --plot

# Custom configuration
python evo/core/eval.py --slam myslam.MySLAM --dataset kitti --seq 00 --config config.json --plot
```

### Output Structure

```
results/MySLAM_kitti_00_timestamp/
├── trajectory.txt         # Estimated trajectory
├── results.json          # Evaluation results
├── ape_result.zip        # APE details
└── rpe_result.zip        # RPE details
```

### Plot Types

#### 1. Trajectory Plot (3D/2D)
**Shows**: Ground truth vs. estimated trajectory  
**Read**: Blue = GT, Red = Estimated, Arrows = orientation  
**Generate**: `--plot` flag with trajectory comparison

#### 2. APE Plot
**Shows**: Absolute error over time  
**Read**: Y-axis = error magnitude (m/rad), Color = error intensity  
**Stats**: RMSE, mean, median, std shown in legend

#### 3. RPE Plot
**Shows**: Relative error between consecutive poses  
**Read**: Local accuracy per delta distance/time

#### 4. Error Distribution
**Shows**: Histogram of error magnitudes  
**Read**: Distribution shape indicates error consistency

### Trajectory Format Conversion

```bash
# Convert between formats
evo_traj tum trajectory.txt --save_as_kitti trajectory_kitti.txt
evo_traj kitti trajectory_kitti.txt --save_as_tum trajectory_tum.txt
evo_traj tum trajectory.txt --save_as_euroc trajectory_euroc.csv

# Convert with plotting
evo_traj tum est.txt --save_as_kitti est_kitti.txt -p
```

**Formats**:
- **TUM**: `timestamp tx ty tz qx qy qz qw`
- **KITTI**: `3x4 transformation matrix`
- **EuRoC**: `CSV with timestamp, position, quaternion`

## 🏗️ Project Structure

```
evo_extend/
├── evo/                      # Main package
│   ├── core/
│   │   ├── eval.py          # Main evaluation script
│   │   └── slam_interface.py # SLAM interface
│   └── tools/
│       ├── dataset_read.py  # Python dataset readers
│       └── dataset_config.py # Dataset config
├── cpp/                      # C++ components
│   ├── src/                 # C++ dataset readers
│   └── test/                # C++ tests
├── myslam.py                # Example SLAM implementations
└── test_myslam.py           # SLAM tests
```

## 📄 License

This project is licensed under the GNU General Public License v3 (GPLv3).

## 🙏 Acknowledgments

- Built on [EVO](https://github.com/MichaelGrupp/evo) by Michael Grupp
- Dataset formats from KITTI, TUM, and EuRoC