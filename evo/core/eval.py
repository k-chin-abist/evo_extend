#!/usr/bin/env python3
"""
Automatic SLAM Evaluation Framework

Reads dataset frame-by-frame → Feeds to SLAM → Gets trajectory → Evaluates with EVO

Usage:
    python eval.py --slam myslam.MySLAM --dataset kitti --seq 00 --plot
    python eval.py --slam myslam.DummySLAM --dataset tum --seq rgbd_dataset_freiburg2_xyz --plot
"""

import argparse
import sys
from pathlib import Path
import json
import time
from datetime import datetime
import importlib
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from evo.tools.dataset_read import KITTIDataset, TUMDataset, EuRoCDataset
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.core.metrics import PoseRelation, Unit
from evo.tools import file_interface
import numpy as np


def load_slam_class(slam_module_path):
    """
    Dynamically load SLAM class
    
    Args:
        slam_module_path: Module path in format "module.ClassName"
                         e.g., "myslam.MySLAM" or "slam_interface.DummySLAM"
    
    Returns:
        SLAM class
    """
    try:
        parts = slam_module_path.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid module path: {slam_module_path}")
        
        module_name, class_name = parts
        
        # Import module
        module = importlib.import_module(module_name)
        
        # Get class
        slam_class = getattr(module, class_name)
        
        return slam_class
        
    except Exception as e:
        print(f"ERROR: Failed to load SLAM class: {e}")
        print(f"   Module path: {slam_module_path}")
        print(f"   Make sure the module is in Python path")
        sys.exit(1)


def run_slam_on_dataset(slam_system, dataset, dataset_type, max_frames=None):
    """
    Run SLAM on dataset
    
    Args:
        slam_system: SLAM system instance
        dataset: Dataset object
        dataset_type: Dataset type ('kitti', 'tum', 'euroc')
        max_frames: Maximum frames to process (None = all)
    
    Returns:
        success: Success flag
        elapsed_time: Execution time
    """
    print(f"\n{'='*60}")
    print(f"INFO: Running SLAM on Dataset")
    print(f"{'='*60}")
    
    # Initialize SLAM
    slam_system.initialize()
    
    start_time = time.time()
    frame_count = 0
    
    try:
        if dataset_type == 'kitti':
            # KITTI: Stereo grayscale images
            left_images = dataset.get_left_images()
            right_images = dataset.get_right_images()
            timestamps, _ = dataset.read_poses()
            
            n_frames = len(left_images)
            if max_frames:
                n_frames = min(n_frames, max_frames)
            
            print(f"Processing {n_frames} frames...")
            
            for i in range(n_frames):
                if i % 100 == 0:
                    print(f"  Frame {i}/{n_frames} ({100*i/n_frames:.1f}%)")
                
                # Read images
                left_img = cv2.imread(str(left_images[i]), cv2.IMREAD_GRAYSCALE)
                right_img = cv2.imread(str(right_images[i]), cv2.IMREAD_GRAYSCALE)
                
                # Pass to SLAM
                success = slam_system.process_frame(
                    timestamp=timestamps[i],
                    left_image=left_img,
                    right_image=right_img
                )
                
                if not success:
                    print(f"WARNING: Frame {i} processing failed")
                
                frame_count += 1
        
        elif dataset_type == 'tum':
            # TUM: RGB-D
            rgb_images = dataset.get_rgb_images()
            depth_images = dataset.get_depth_images()
            rgb_times, _ = dataset.read_rgb_timestamps()
            depth_times, _ = dataset.read_depth_timestamps()
            
            # Align RGB and Depth (simple version: assume already aligned)
            n_frames = min(len(rgb_images), len(depth_images))
            if max_frames:
                n_frames = min(n_frames, max_frames)
            
            print(f"Processing {n_frames} frames...")
            
            for i in range(n_frames):
                if i % 100 == 0:
                    print(f"  Frame {i}/{n_frames} ({100*i/n_frames:.1f}%)")
                
                # Read images
                rgb_img = cv2.imread(str(rgb_images[i]))
                depth_img = cv2.imread(str(depth_images[i]), cv2.IMREAD_UNCHANGED)
                
                # Pass to SLAM
                success = slam_system.process_frame(
                    timestamp=rgb_times[i],
                    rgb_image=rgb_img,
                    depth_image=depth_img
                )
                
                if not success:
                    print(f"WARNING: Frame {i} processing failed")
                
                frame_count += 1
        
        elif dataset_type == 'euroc':
            # EuRoC: Stereo monochrome
            cam0_images = dataset.get_cam0_images()
            cam1_images = dataset.get_cam1_images()
            cam0_times, _ = dataset.read_cam0_timestamps()
            
            n_frames = min(len(cam0_images), len(cam1_images))
            if max_frames:
                n_frames = min(n_frames, max_frames)
            
            print(f"Processing {n_frames} frames...")
            
            for i in range(n_frames):
                if i % 100 == 0:
                    print(f"  Frame {i}/{n_frames} ({100*i/n_frames:.1f}%)")
                
                # Read images
                left_img = cv2.imread(str(cam0_images[i]), cv2.IMREAD_GRAYSCALE)
                right_img = cv2.imread(str(cam1_images[i]), cv2.IMREAD_GRAYSCALE)
                
                # Pass to SLAM
                success = slam_system.process_frame(
                    timestamp=cam0_times[i],
                    left_image=left_img,
                    right_image=right_img
                )
                
                if not success:
                    print(f"WARNING: Frame {i} processing failed")
                
                frame_count += 1
        
        elapsed_time = time.time() - start_time
        
        print(f"\nSUCCESS: SLAM completed!")
        print(f"   Processed: {frame_count} frames")
        print(f"   Time: {elapsed_time:.2f}s")
        print(f"   FPS: {frame_count/elapsed_time:.2f}")
        
        # Shutdown SLAM
        slam_system.shutdown()
        
        return True, elapsed_time
        
    except Exception as e:
        print(f"ERROR: Error during SLAM execution: {e}")
        import traceback
        traceback.print_exc()
        return False, time.time() - start_time


def load_ground_truth(dataset, dataset_type):
    """Load ground truth"""
    try:
        if dataset_type == 'kitti':
            timestamps, poses = dataset.read_poses()
            poses_se3 = []
            for pose in poses:
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = pose
                poses_se3.append(pose_4x4)
            
            from scipy.spatial.transform import Rotation
            positions = np.array([p[:3, 3] for p in poses_se3])
            quats = []
            for p in poses_se3:
                r = Rotation.from_matrix(p[:3, :3])
                q = r.as_quat()
                quats.append([q[3], q[0], q[1], q[2]])
            
            return PoseTrajectory3D(
                positions_xyz=positions,
                orientations_quat_wxyz=np.array(quats),
                timestamps=np.array(timestamps)
            )
        
        elif dataset_type in ['tum', 'euroc']:
            timestamps, poses = dataset.read_poses()
            positions = np.array([p[:3] for p in poses])
            quats = np.array([[p[6], p[3], p[4], p[5]] for p in poses])
            
            return PoseTrajectory3D(
                positions_xyz=positions,
                orientations_quat_wxyz=quats,
                timestamps=np.array(timestamps)
            )
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None


def slam_trajectory_to_evo(timestamps, poses):
    """Convert SLAM trajectory to EVO format"""
    positions = np.array([p[:3] for p in poses])
    # poses: [tx, ty, tz, qx, qy, qz, qw] -> [w, x, y, z]
    quats = np.array([[p[6], p[3], p[4], p[5]] for p in poses])
    
    return PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=quats,
        timestamps=np.array(timestamps)
    )


def evaluate_trajectory(gt_traj, est_traj):
    """Evaluate trajectory using EVO's built-in functions"""
    print(f"\n{'='*60}")
    print(f"INFO: Evaluating Trajectory")
    print(f"{'='*60}")
    
    # Debug information
    print(f"Ground truth trajectory: {len(gt_traj.positions_xyz)} poses")
    print(f"Estimated trajectory: {len(est_traj.positions_xyz)} poses")
    print(f"GT time range: {gt_traj.timestamps[0]:.3f} - {gt_traj.timestamps[-1]:.3f}")
    print(f"EST time range: {est_traj.timestamps[0]:.3f} - {est_traj.timestamps[-1]:.3f}")
    
    # Synchronize trajectories using EVO's sync module
    print("Synchronizing trajectories...")
    try:
        from evo.core import sync
        gt_traj_sync, est_traj_sync = sync.associate_trajectories(
            gt_traj, est_traj, max_diff=0.01
        )
        print(f"After sync - GT: {len(gt_traj_sync.positions_xyz)} poses, EST: {len(est_traj_sync.positions_xyz)} poses")
    except Exception as e:
        print(f"WARNING: Sync failed: {e}")
        print("Using original trajectories...")
        gt_traj_sync, est_traj_sync = gt_traj, est_traj
    
    # Use EVO's built-in APE evaluation
    from evo.main_ape import ape
    from evo.main_rpe import rpe
    from evo.core import metrics
    
    print("Computing APE using EVO...")
    try:
        ape_result = ape(
            traj_ref=gt_traj_sync,
            traj_est=est_traj_sync,
            pose_relation=metrics.PoseRelation.translation_part,
            align=True,  # Enable trajectory alignment
            ref_name="ground_truth",
            est_name="slam_estimate"
        )
    except Exception as e:
        print(f"ERROR: APE computation failed: {e}")
        print("Trying without alignment...")
        ape_result = ape(
            traj_ref=gt_traj_sync,
            traj_est=est_traj_sync,
            pose_relation=metrics.PoseRelation.translation_part,
            align=False,  # Disable trajectory alignment
            ref_name="ground_truth",
            est_name="slam_estimate"
        )
    
    print("Computing RPE using EVO...")
    try:
        rpe_result = rpe(
            traj_ref=gt_traj_sync,
            traj_est=est_traj_sync,
            pose_relation=metrics.PoseRelation.translation_part,
            delta=1.0,
            delta_unit=metrics.Unit.frames,
            align=True,
            ref_name="ground_truth",
            est_name="slam_estimate"
        )
    except Exception as e:
        print(f"ERROR: RPE computation failed: {e}")
        print("Trying without alignment...")
        rpe_result = rpe(
            traj_ref=gt_traj_sync,
            traj_est=est_traj_sync,
            pose_relation=metrics.PoseRelation.translation_part,
            delta=1.0,
            delta_unit=metrics.Unit.frames,
            align=False,
            ref_name="ground_truth",
            est_name="slam_estimate"
        )
    
    # Extract statistics
    ape_stats = ape_result.stats
    rpe_stats = rpe_result.stats
    
    results = {
        'num_poses': len(ape_result.trajectories["slam_estimate"].positions_xyz),
        'ape': {k: float(v) for k, v in ape_stats.items()},
        'rpe': {k: float(v) for k, v in rpe_stats.items()}
    }
    
    # Print results using EVO's formatting
    print(f"\n{'='*60}")
    print(f"INFO: Results")
    print(f"{'='*60}")
    print(f"Matched poses: {results['num_poses']}")
    print(f"\nAPE (Absolute Pose Error) [m]:")
    print(f"  RMSE:   {results['ape']['rmse']:.6f}")
    print(f"  Mean:   {results['ape']['mean']:.6f}")
    print(f"  Median: {results['ape']['median']:.6f}")
    print(f"  Std:    {results['ape']['std']:.6f}")
    print(f"\nRPE (Relative Pose Error) [m]:")
    print(f"  RMSE:   {results['rpe']['rmse']:.6f}")
    print(f"  Mean:   {results['rpe']['mean']:.6f}")
    print(f"  Median: {results['rpe']['median']:.6f}")
    print(f"  Std:    {results['rpe']['std']:.6f}")
    print(f"{'='*60}")
    
    return results, ape_result, rpe_result


def plot_results(ape_result, rpe_result, save_path=None):
    """Plot results using EVO's built-in plotting"""
    import evo.common_ape_rpe as common
    import argparse
    
    # Create mock args for EVO's plotting function
    args = argparse.Namespace()
    args.plot = True
    args.save_plot = save_path is not None
    args.plot_mode = 'xy'
    args.verbose = False
    args.no_warnings = False
    
    # Use EVO's built-in plotting
    print("INFO: Generating plots using EVO...")
    common.plot_result(
        args=args,
        result=ape_result,
        traj_ref=ape_result.trajectories["ground_truth"],
        traj_est=ape_result.trajectories["slam_estimate"]
    )
    
    if save_path:
        print(f"INFO: Plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Automatic SLAM Evaluation')
    parser.add_argument('--slam', required=True, 
                       help='SLAM class (format: module.ClassName, e.g., slam_interface.DummySLAM)')
    parser.add_argument('--dataset', required=True, choices=['kitti', 'tum', 'euroc'])
    parser.add_argument('--seq', required=True, help='Sequence ID')
    parser.add_argument('--config', help='SLAM config file (optional)')
    parser.add_argument('--max_frames', type=int, help='Max frames to process (for testing)')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--save_plots', action='store_true', help='Save plots')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('results') / f"{args.slam.split('.')[-1]}_{args.dataset}_{args.seq}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"INFO: Loading dataset: {args.dataset} - {args.seq}")
    if args.dataset == 'kitti':
        dataset = KITTIDataset(args.seq)
    elif args.dataset == 'tum':
        dataset = TUMDataset(args.seq)
    elif args.dataset == 'euroc':
        dataset = EuRoCDataset(args.seq)
    
    # Load SLAM class
    print(f"INFO: Loading SLAM: {args.slam}")
    SLAMClass = load_slam_class(args.slam)
    
    # Create SLAM instance
    slam_system = SLAMClass(config_file=args.config)
    print(f"SUCCESS: SLAM system created: {slam_system.__class__.__name__}")
    
    # Run SLAM
    # If max_frames is specified, use it; otherwise process all frames
    max_frames = args.max_frames if args.max_frames else None
    success, elapsed_time = run_slam_on_dataset(
        slam_system, dataset, args.dataset, max_frames
    )
    
    if not success:
        print("ERROR: SLAM execution failed")
        return 1
    
    # Get SLAM trajectory
    print("\nINFO: Getting SLAM trajectory...")
    slam_timestamps, slam_poses = slam_system.get_trajectory()
    print(f"   SLAM estimated {len(slam_poses)} poses")
    
    # Save trajectory
    traj_file = output_dir / 'trajectory.txt'
    slam_system.save_trajectory(str(traj_file))
    print(f"INFO: Trajectory saved to: {traj_file}")
    
    # Load ground truth
    print("\nINFO: Loading ground truth...")
    gt_traj = load_ground_truth(dataset, args.dataset)
    
    # Convert SLAM trajectory to EVO format
    est_traj = slam_trajectory_to_evo(slam_timestamps, slam_poses)
    
    # Evaluate using EVO's built-in functions
    results, ape_result, rpe_result = evaluate_trajectory(gt_traj, est_traj)
    results['execution_time'] = elapsed_time
    results['dataset'] = args.dataset
    results['sequence'] = args.seq
    results['slam'] = args.slam
    
    # Save results
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nINFO: Results saved to: {results_file}")
    
    # Save EVO result files
    ape_file = output_dir / 'ape_result.zip'
    rpe_file = output_dir / 'rpe_result.zip'
    from evo.tools import file_interface
    file_interface.save_res_file(str(ape_file), ape_result)
    file_interface.save_res_file(str(rpe_file), rpe_result)
    print(f"INFO: EVO results saved to: {ape_file}, {rpe_file}")
    
    # Plot using EVO's built-in plotting
    if args.plot or args.save_plots:
        plot_file = output_dir / 'evaluation.png' if args.save_plots else None
        plot_results(ape_result, rpe_result, plot_file)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())