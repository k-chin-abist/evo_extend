#include "dataset_reader.hpp"
#include <iostream>
#include <iomanip>

void testKITTI() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "KITTI Dataset Test" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        KITTIDataset dataset("00");
        std::cout << "[OK] Created KITTI dataset" << std::endl;
        std::cout << "[OK] Number of frames: " << dataset.size() << std::endl;

        std::vector<double> timestamps;
        std::vector<Eigen::Matrix<double, 3, 4>> poses;
        
        if (dataset.readPoses(timestamps, poses)) {
            std::cout << "[OK] Read " << poses.size() << " poses" << std::endl;
            std::cout << "     First timestamp: " << std::fixed << std::setprecision(3) 
                      << timestamps[0] << "s" << std::endl;
            std::cout << "     Last timestamp: " << timestamps.back() << "s" << std::endl;
        }

        auto left_images = dataset.getLeftImages();
        std::cout << "[OK] Left images: " << left_images.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
    }
}

void testTUM() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TUM Dataset Test" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // Try first available sequence - you may need to change this
        TUMDataset dataset("rgbd_dataset_freiburg2_xyz");
        std::cout << "[OK] Created TUM dataset" << std::endl;
        std::cout << "[OK] Number of frames: " << dataset.size() << std::endl;

        std::vector<double> timestamps;
        std::vector<Eigen::Matrix<double, 7, 1>> poses;
        
        if (dataset.readPoses(timestamps, poses)) {
            std::cout << "[OK] Read " << poses.size() << " poses" << std::endl;
            std::cout << "     First timestamp: " << std::fixed << std::setprecision(6) 
                      << timestamps[0] << "s" << std::endl;
            std::cout << "     Last timestamp: " << timestamps.back() << "s" << std::endl;
            
            // Print first pose (tx, ty, tz, qx, qy, qz, qw)
            std::cout << "     First pose: [" 
                      << poses[0](0) << ", " << poses[0](1) << ", " << poses[0](2) << ", "
                      << poses[0](3) << ", " << poses[0](4) << ", " << poses[0](5) << ", " 
                      << poses[0](6) << "]" << std::endl;
        }

        std::vector<double> rgb_times;
        std::vector<std::string> rgb_files;
        if (dataset.readRGBTimestamps(rgb_times, rgb_files)) {
            std::cout << "[OK] RGB timestamps: " << rgb_times.size() << std::endl;
        }

        auto rgb_images = dataset.getRGBImages();
        std::cout << "[OK] RGB images: " << rgb_images.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
    }
}

void testEuRoC() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "EuRoC Dataset Test" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // Try first sequence - you may need to change this
        EuRoCDataset dataset("MH_01_easy");
        std::cout << "[OK] Created EuRoC dataset" << std::endl;
        std::cout << "[OK] Number of frames: " << dataset.size() << std::endl;

        std::vector<double> timestamps;
        std::vector<Eigen::Matrix<double, 7, 1>> poses;
        
        if (dataset.readPoses(timestamps, poses)) {
            std::cout << "[OK] Read " << poses.size() << " poses" << std::endl;
            std::cout << "     First timestamp: " << std::fixed << std::setprecision(6) 
                      << timestamps[0] << "s" << std::endl;
            std::cout << "     Last timestamp: " << timestamps.back() << "s" << std::endl;
            
            // Print first pose
            std::cout << "     First pose: [" 
                      << poses[0](0) << ", " << poses[0](1) << ", " << poses[0](2) << ", "
                      << poses[0](3) << ", " << poses[0](4) << ", " << poses[0](5) << ", " 
                      << poses[0](6) << "]" << std::endl;
        }

        std::vector<double> cam0_times;
        std::vector<std::string> cam0_files;
        if (dataset.readCam0Timestamps(cam0_times, cam0_files)) {
            std::cout << "[OK] Cam0 timestamps: " << cam0_times.size() << std::endl;
        }

        std::vector<double> imu_times;
        std::vector<Eigen::Vector3d> gyro, accel;
        if (dataset.readIMUData(imu_times, gyro, accel)) {
            std::cout << "[OK] IMU samples: " << imu_times.size() << std::endl;
        }

        auto cam0_images = dataset.getCam0Images();
        std::cout << "[OK] Cam0 images: " << cam0_images.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset Reader C++ Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    testKITTI();
    testTUM();
    testEuRoC();

    std::cout << "\n========================================" << std::endl;
    std::cout << "All tests completed!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}