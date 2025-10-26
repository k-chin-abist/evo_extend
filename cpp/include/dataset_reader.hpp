#ifndef DATASET_READER_HPP
#define DATASET_READER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <Eigen/Dense>

namespace fs = std::filesystem;

// ============================================================================
// KITTI Dataset
// ============================================================================
class KITTIDataset {
public:
    /**
     * Constructor - reads root_path from config if not provided
     * @param sequence_id Sequence ID (e.g., "00", "01")
     * @param root_path KITTI root directory (optional, reads from config if empty)
     * @param use_color Use color images if true, grayscale if false
     */
    KITTIDataset(const std::string& sequence_id,
                 const std::string& root_path = "",
                 bool use_color = false);

    /**
     * Read ground truth poses
     * @param timestamps Output: timestamps (seconds)
     * @param poses Output: SE(3) poses as 3x4 matrices
     * @return true if successful
     */
    bool readPoses(std::vector<double>& timestamps,
                   std::vector<Eigen::Matrix<double, 3, 4>>& poses);

    /**
     * Get left camera image file paths
     * @return Vector of image file paths
     */
    std::vector<fs::path> getLeftImages();

    /**
     * Get right camera image file paths
     * @return Vector of image file paths
     */
    std::vector<fs::path> getRightImages();

    /**
     * Get number of frames
     * @return Number of poses/frames
     */
    size_t size() const;

private:
    std::string root_path_;
    std::string sequence_id_;
    bool use_color_;

    fs::path sequence_dir_;
    fs::path poses_file_;
    fs::path calib_file_;
    fs::path image_0_dir_;
    fs::path image_1_dir_;
    fs::path times_file_;

    void setupPaths();
    bool fileExists(const fs::path& path) const;
    std::string readRootFromConfig();
};

// ============================================================================
// TUM Dataset
// ============================================================================
class TUMDataset {
public:
    /**
     * Constructor - reads root_path from config if not provided
     * @param sequence_id Sequence name (e.g., "rgbd_dataset_freiburg2_xyz")
     * @param root_path TUM root directory (optional, reads from config if empty)
     */
    TUMDataset(const std::string& sequence_id,
               const std::string& root_path = "");

    /**
     * Read ground truth poses
     * Format: timestamp tx ty tz qx qy qz qw
     * @param timestamps Output: timestamps (seconds)
     * @param poses Output: poses as [tx, ty, tz, qx, qy, qz, qw]
     * @return true if successful
     */
    bool readPoses(std::vector<double>& timestamps,
                   std::vector<Eigen::Matrix<double, 7, 1>>& poses);

    /**
     * Read RGB timestamps and filenames
     * @param timestamps Output: timestamps (seconds)
     * @param filenames Output: image filenames
     * @return true if successful
     */
    bool readRGBTimestamps(std::vector<double>& timestamps,
                           std::vector<std::string>& filenames);

    /**
     * Read depth timestamps and filenames
     * @param timestamps Output: timestamps (seconds)
     * @param filenames Output: depth image filenames
     * @return true if successful
     */
    bool readDepthTimestamps(std::vector<double>& timestamps,
                             std::vector<std::string>& filenames);

    /**
     * Get RGB image file paths
     * @return Vector of RGB image file paths
     */
    std::vector<fs::path> getRGBImages();

    /**
     * Get depth image file paths
     * @return Vector of depth image file paths
     */
    std::vector<fs::path> getDepthImages();

    /**
     * Get number of poses
     * @return Number of poses
     */
    size_t size() const;

private:
    std::string root_path_;
    std::string sequence_id_;

    fs::path sequence_dir_;
    fs::path groundtruth_file_;
    fs::path rgb_txt_;
    fs::path depth_txt_;
    fs::path rgb_dir_;
    fs::path depth_dir_;

    void setupPaths();
    bool fileExists(const fs::path& path) const;
    std::string readRootFromConfig();
};

// ============================================================================
// EuRoC Dataset
// ============================================================================
class EuRoCDataset {
public:
    /**
     * Constructor - reads root_path from config if not provided
     * @param sequence_id Sequence name (e.g., "MH_01_easy")
     * @param root_path EuRoC root directory (optional, reads from config if empty)
     */
    EuRoCDataset(const std::string& sequence_id,
                 const std::string& root_path = "");

    /**
     * Read ground truth poses
     * Format: timestamp,px,py,pz,qw,qx,qy,qz,...
     * @param timestamps Output: timestamps (seconds)
     * @param poses Output: poses as [tx, ty, tz, qx, qy, qz, qw]
     * @return true if successful
     */
    bool readPoses(std::vector<double>& timestamps,
                   std::vector<Eigen::Matrix<double, 7, 1>>& poses);

    /**
     * Read camera 0 timestamps and filenames
     * @param timestamps Output: timestamps (seconds)
     * @param filenames Output: image filenames
     * @return true if successful
     */
    bool readCam0Timestamps(std::vector<double>& timestamps,
                            std::vector<std::string>& filenames);

    /**
     * Read camera 1 timestamps and filenames
     * @param timestamps Output: timestamps (seconds)
     * @param filenames Output: image filenames
     * @return true if successful
     */
    bool readCam1Timestamps(std::vector<double>& timestamps,
                            std::vector<std::string>& filenames);

    /**
     * Read IMU data
     * @param timestamps Output: timestamps (seconds)
     * @param gyro Output: gyroscope data [wx, wy, wz]
     * @param accel Output: accelerometer data [ax, ay, az]
     * @return true if successful
     */
    bool readIMUData(std::vector<double>& timestamps,
                     std::vector<Eigen::Vector3d>& gyro,
                     std::vector<Eigen::Vector3d>& accel);

    /**
     * Get camera 0 image file paths
     * @return Vector of camera 0 image file paths
     */
    std::vector<fs::path> getCam0Images();

    /**
     * Get camera 1 image file paths
     * @return Vector of camera 1 image file paths
     */
    std::vector<fs::path> getCam1Images();

    /**
     * Get number of poses
     * @return Number of poses
     */
    size_t size() const;

private:
    std::string root_path_;
    std::string sequence_id_;

    fs::path sequence_dir_;
    fs::path mav0_dir_;
    fs::path groundtruth_file_;
    fs::path cam0_csv_;
    fs::path cam1_csv_;
    fs::path imu_file_;
    fs::path cam0_data_;
    fs::path cam1_data_;

    void setupPaths();
    bool fileExists(const fs::path& path) const;
    std::string readRootFromConfig();
};

#endif // DATASET_READER_HPP