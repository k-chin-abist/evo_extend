#include "dataset_reader.hpp"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cstdlib>

KITTIDataset::KITTIDataset(const std::string& sequence_id,
                           const std::string& root_path,
                           bool use_color)
    : sequence_id_(sequence_id)
    , use_color_(use_color)
{
    // Get root path from config if not provided
    if (root_path.empty()) {
        root_path_ = readRootFromConfig();
        if (root_path_.empty()) {
            throw std::runtime_error(
                "KITTI root path not provided and not found in config.\n"
                "Please configure using: python -m evo.tools.dataset_settings set kitti /path/to/kitti"
            );
        }
    } else {
        root_path_ = root_path;
    }
    
    // Ensure sequence_id is 2 digits
    if (sequence_id_.length() == 1) {
        sequence_id_ = "0" + sequence_id_;
    }
    
    setupPaths();
}

std::string KITTIDataset::readRootFromConfig() {
#ifdef _WIN32
    const char* user_profile = std::getenv("USERPROFILE");
    if (!user_profile) {
        return "";
    }
    
    std::string config_path = std::string(user_profile) + "\\.evo\\dataset_config.json";
#else
    const char* home = std::getenv("HOME");
    if (!home) {
        return "";
    }
    std::string config_path = std::string(home) + "/.evo/dataset_config.json";
#endif

    std::ifstream file(config_path);
    if (!file.is_open()) {
        return "";
    }

    // Simple JSON parsing for "kitti": "path"
    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find("\"kitti\"");
        if (pos != std::string::npos) {
            // Find the path after "kitti":
            size_t colon = line.find(":", pos);
            if (colon != std::string::npos) {
                size_t first_quote = line.find("\"", colon + 1);
                size_t second_quote = line.find("\"", first_quote + 1);
                if (first_quote != std::string::npos && second_quote != std::string::npos) {
                    std::string path = line.substr(first_quote + 1, second_quote - first_quote - 1);
                    // Replace \\ with /
                    for (size_t i = 0; i < path.length(); ++i) {
                        if (path[i] == '\\' && i + 1 < path.length() && path[i + 1] == '\\') {
                            path.replace(i, 2, "/");
                        }
                    }
                    return path;
                }
            }
        }
    }
    
    return "";
}

void KITTIDataset::setupPaths() {
    fs::path root(root_path_);
    
    // Setup paths based on KITTI structure
    std::string image_folder = use_color_ ? "data_odometry_color" : "data_odometry_gray";
    
    sequence_dir_ = root / image_folder / "dataset" / "sequences" / sequence_id_;
    poses_file_ = root / "data_odometry_poses" / "dataset" / "poses" / (sequence_id_ + ".txt");
    calib_file_ = root / "data_odometry_calib" / "dataset" / "sequences" / sequence_id_ / "calib.txt";
    
    image_0_dir_ = sequence_dir_ / "image_0";
    image_1_dir_ = sequence_dir_ / "image_1";
    times_file_ = sequence_dir_ / "times.txt";
}

bool KITTIDataset::fileExists(const fs::path& path) const {
    return fs::exists(path);
}

bool KITTIDataset::readPoses(std::vector<double>& timestamps,
                             std::vector<Eigen::Matrix<double, 3, 4>>& poses) {
    if (!fileExists(poses_file_)) {
        std::cerr << "Poses file not found: " << poses_file_ << std::endl;
        return false;
    }

    std::ifstream file(poses_file_);
    if (!file.is_open()) {
        std::cerr << "Failed to open poses file: " << poses_file_ << std::endl;
        return false;
    }

    poses.clear();
    std::string line;
    int frame_id = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        Eigen::Matrix<double, 3, 4> pose;
        
        // Read 12 values (3x4 matrix)
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (!(iss >> pose(i, j))) {
                    std::cerr << "Error reading pose at frame " << frame_id << std::endl;
                    return false;
                }
            }
        }
        
        poses.push_back(pose);
        frame_id++;
    }

    file.close();

    // Read timestamps if available
    timestamps.clear();
    if (fileExists(times_file_)) {
        std::ifstream times_file(times_file_);
        double timestamp;
        while (times_file >> timestamp) {
            timestamps.push_back(timestamp);
        }
        times_file.close();
    } else {
        // Generate synthetic timestamps (assuming 10 Hz)
        for (size_t i = 0; i < poses.size(); ++i) {
            timestamps.push_back(i * 0.1);
        }
    }

    return true;
}

std::vector<fs::path> KITTIDataset::getLeftImages() {
    std::vector<fs::path> images;
    
    if (!fs::exists(image_0_dir_)) {
        return images;
    }

    for (const auto& entry : fs::directory_iterator(image_0_dir_)) {
        if (entry.path().extension() == ".png") {
            images.push_back(entry.path());
        }
    }

    std::sort(images.begin(), images.end());
    return images;
}

std::vector<fs::path> KITTIDataset::getRightImages() {
    std::vector<fs::path> images;
    
    if (!fs::exists(image_1_dir_)) {
        return images;
    }

    for (const auto& entry : fs::directory_iterator(image_1_dir_)) {
        if (entry.path().extension() == ".png") {
            images.push_back(entry.path());
        }
    }

    std::sort(images.begin(), images.end());
    return images;
}

size_t KITTIDataset::size() const {
    if (!fileExists(poses_file_)) {
        return 0;
    }

    std::ifstream file(poses_file_);
    size_t count = 0;
    std::string line;
    
    while (std::getline(file, line)) {
        if (!line.empty()) {
            count++;
        }
    }
    
    return count;
}

// ============================================================================
// TUM Dataset Implementation
// ============================================================================

TUMDataset::TUMDataset(const std::string& sequence_id,
                       const std::string& root_path)
    : sequence_id_(sequence_id)
{
    // Get root path from config if not provided
    if (root_path.empty()) {
        root_path_ = readRootFromConfig();
        if (root_path_.empty()) {
            throw std::runtime_error(
                "TUM root path not provided and not found in config.\n"
                "Please configure using: python -m evo.tools.dataset_settings set tum /path/to/tum"
            );
        }
    } else {
        root_path_ = root_path;
    }
    
    setupPaths();
}

std::string TUMDataset::readRootFromConfig() {
#ifdef _WIN32
    const char* user_profile = std::getenv("USERPROFILE");
    if (!user_profile) {
        return "";
    }
    std::string config_path = std::string(user_profile) + "\\.evo\\dataset_config.json";
#else
    const char* home = std::getenv("HOME");
    if (!home) {
        return "";
    }
    std::string config_path = std::string(home) + "/.evo/dataset_config.json";
#endif

    std::ifstream file(config_path);
    if (!file.is_open()) {
        return "";
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find("\"tum\"");
        if (pos != std::string::npos) {
            size_t colon = line.find(":", pos);
            if (colon != std::string::npos) {
                size_t first_quote = line.find("\"", colon + 1);
                size_t second_quote = line.find("\"", first_quote + 1);
                if (first_quote != std::string::npos && second_quote != std::string::npos) {
                    std::string path = line.substr(first_quote + 1, second_quote - first_quote - 1);
                    for (size_t i = 0; i < path.length(); ++i) {
                        if (path[i] == '\\' && i + 1 < path.length() && path[i + 1] == '\\') {
                            path.replace(i, 2, "/");
                        }
                    }
                    return path;
                }
            }
        }
    }
    
    return "";
}

void TUMDataset::setupPaths() {
    fs::path root(root_path_);
    sequence_dir_ = root / sequence_id_;
    groundtruth_file_ = sequence_dir_ / "groundtruth.txt";
    rgb_txt_ = sequence_dir_ / "rgb.txt";
    depth_txt_ = sequence_dir_ / "depth.txt";
    rgb_dir_ = sequence_dir_ / "rgb";
    depth_dir_ = sequence_dir_ / "depth";
}

bool TUMDataset::fileExists(const fs::path& path) const {
    return fs::exists(path);
}

bool TUMDataset::readPoses(std::vector<double>& timestamps,
                           std::vector<Eigen::Matrix<double, 7, 1>>& poses) {
    if (!fileExists(groundtruth_file_)) {
        std::cerr << "Groundtruth file not found: " << groundtruth_file_ << std::endl;
        return false;
    }

    std::ifstream file(groundtruth_file_);
    if (!file.is_open()) {
        return false;
    }

    timestamps.clear();
    poses.clear();
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        double timestamp, tx, ty, tz, qx, qy, qz, qw;
        
        if (iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
            timestamps.push_back(timestamp);
            
            Eigen::Matrix<double, 7, 1> pose;
            pose << tx, ty, tz, qx, qy, qz, qw;
            poses.push_back(pose);
        }
    }

    return true;
}

bool TUMDataset::readRGBTimestamps(std::vector<double>& timestamps,
                                   std::vector<std::string>& filenames) {
    if (!fileExists(rgb_txt_)) {
        std::cerr << "RGB timestamps file not found: " << rgb_txt_ << std::endl;
        return false;
    }

    std::ifstream file(rgb_txt_);
    timestamps.clear();
    filenames.clear();
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        double timestamp;
        std::string filename;
        
        if (iss >> timestamp >> filename) {
            timestamps.push_back(timestamp);
            filenames.push_back(filename);
        }
    }

    return true;
}

bool TUMDataset::readDepthTimestamps(std::vector<double>& timestamps,
                                     std::vector<std::string>& filenames) {
    if (!fileExists(depth_txt_)) {
        std::cerr << "Depth timestamps file not found: " << depth_txt_ << std::endl;
        return false;
    }

    std::ifstream file(depth_txt_);
    timestamps.clear();
    filenames.clear();
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        double timestamp;
        std::string filename;
        
        if (iss >> timestamp >> filename) {
            timestamps.push_back(timestamp);
            filenames.push_back(filename);
        }
    }

    return true;
}

std::vector<fs::path> TUMDataset::getRGBImages() {
    std::vector<fs::path> images;
    if (!fs::exists(rgb_dir_)) {
        return images;
    }

    for (const auto& entry : fs::directory_iterator(rgb_dir_)) {
        if (entry.path().extension() == ".png") {
            images.push_back(entry.path());
        }
    }

    std::sort(images.begin(), images.end());
    return images;
}

std::vector<fs::path> TUMDataset::getDepthImages() {
    std::vector<fs::path> images;
    if (!fs::exists(depth_dir_)) {
        return images;
    }

    for (const auto& entry : fs::directory_iterator(depth_dir_)) {
        if (entry.path().extension() == ".png") {
            images.push_back(entry.path());
        }
    }

    std::sort(images.begin(), images.end());
    return images;
}

size_t TUMDataset::size() const {
    if (!fileExists(groundtruth_file_)) {
        return 0;
    }

    std::ifstream file(groundtruth_file_);
    size_t count = 0;
    std::string line;
    
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] != '#') {
            count++;
        }
    }
    
    return count;
}

// ============================================================================
// EuRoC Dataset Implementation
// ============================================================================

EuRoCDataset::EuRoCDataset(const std::string& sequence_id,
                           const std::string& root_path)
    : sequence_id_(sequence_id)
{
    // Get root path from config if not provided
    if (root_path.empty()) {
        root_path_ = readRootFromConfig();
        if (root_path_.empty()) {
            throw std::runtime_error(
                "EuRoC root path not provided and not found in config.\n"
                "Please configure using: python -m evo.tools.dataset_settings set euroc /path/to/euroc"
            );
        }
    } else {
        root_path_ = root_path;
    }
    
    setupPaths();
}

std::string EuRoCDataset::readRootFromConfig() {
#ifdef _WIN32
    const char* user_profile = std::getenv("USERPROFILE");
    if (!user_profile) {
        return "";
    }
    std::string config_path = std::string(user_profile) + "\\.evo\\dataset_config.json";
#else
    const char* home = std::getenv("HOME");
    if (!home) {
        return "";
    }
    std::string config_path = std::string(home) + "/.evo/dataset_config.json";
#endif

    std::ifstream file(config_path);
    if (!file.is_open()) {
        return "";
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find("\"euroc\"");
        if (pos != std::string::npos) {
            size_t colon = line.find(":", pos);
            if (colon != std::string::npos) {
                size_t first_quote = line.find("\"", colon + 1);
                size_t second_quote = line.find("\"", first_quote + 1);
                if (first_quote != std::string::npos && second_quote != std::string::npos) {
                    std::string path = line.substr(first_quote + 1, second_quote - first_quote - 1);
                    for (size_t i = 0; i < path.length(); ++i) {
                        if (path[i] == '\\' && i + 1 < path.length() && path[i + 1] == '\\') {
                            path.replace(i, 2, "/");
                        }
                    }
                    return path;
                }
            }
        }
    }
    
    return "";
}

void EuRoCDataset::setupPaths() {
    fs::path root(root_path_);
    sequence_dir_ = root / sequence_id_;
    mav0_dir_ = sequence_dir_ / "mav0";
    groundtruth_file_ = mav0_dir_ / "state_groundtruth_estimate0" / "data.csv";
    cam0_csv_ = mav0_dir_ / "cam0" / "data.csv";
    cam1_csv_ = mav0_dir_ / "cam1" / "data.csv";
    imu_file_ = mav0_dir_ / "imu0" / "data.csv";
    cam0_data_ = mav0_dir_ / "cam0" / "data";
    cam1_data_ = mav0_dir_ / "cam1" / "data";
}

bool EuRoCDataset::fileExists(const fs::path& path) const {
    return fs::exists(path);
}

bool EuRoCDataset::readPoses(std::vector<double>& timestamps,
                             std::vector<Eigen::Matrix<double, 7, 1>>& poses) {
    if (!fileExists(groundtruth_file_)) {
        std::cerr << "Groundtruth file not found: " << groundtruth_file_ << std::endl;
        return false;
    }

    std::ifstream file(groundtruth_file_);
    if (!file.is_open()) {
        return false;
    }

    timestamps.clear();
    poses.clear();
    std::string line;
    
    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string token;
        std::vector<double> values;
        
        while (std::getline(iss, token, ',')) {
            values.push_back(std::stod(token));
        }
        
        if (values.size() >= 8) {
            double timestamp = values[0] / 1e9;  // Convert nanoseconds to seconds
            double px = values[1], py = values[2], pz = values[3];
            double qw = values[4], qx = values[5], qy = values[6], qz = values[7];
            
            timestamps.push_back(timestamp);
            
            Eigen::Matrix<double, 7, 1> pose;
            pose << px, py, pz, qx, qy, qz, qw;
            poses.push_back(pose);
        }
    }

    return true;
}

bool EuRoCDataset::readCam0Timestamps(std::vector<double>& timestamps,
                                      std::vector<std::string>& filenames) {
    if (!fileExists(cam0_csv_)) {
        std::cerr << "Cam0 CSV not found: " << cam0_csv_ << std::endl;
        return false;
    }

    std::ifstream file(cam0_csv_);
    timestamps.clear();
    filenames.clear();
    std::string line;
    
    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        size_t comma = line.find(',');
        if (comma != std::string::npos) {
            double timestamp = std::stod(line.substr(0, comma)) / 1e9;
            std::string filename = line.substr(comma + 1);
            
            timestamps.push_back(timestamp);
            filenames.push_back(filename);
        }
    }

    return true;
}

bool EuRoCDataset::readCam1Timestamps(std::vector<double>& timestamps,
                                      std::vector<std::string>& filenames) {
    if (!fileExists(cam1_csv_)) {
        std::cerr << "Cam1 CSV not found: " << cam1_csv_ << std::endl;
        return false;
    }

    std::ifstream file(cam1_csv_);
    timestamps.clear();
    filenames.clear();
    std::string line;
    
    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        size_t comma = line.find(',');
        if (comma != std::string::npos) {
            double timestamp = std::stod(line.substr(0, comma)) / 1e9;
            std::string filename = line.substr(comma + 1);
            
            timestamps.push_back(timestamp);
            filenames.push_back(filename);
        }
    }

    return true;
}

bool EuRoCDataset::readIMUData(std::vector<double>& timestamps,
                               std::vector<Eigen::Vector3d>& gyro,
                               std::vector<Eigen::Vector3d>& accel) {
    if (!fileExists(imu_file_)) {
        std::cerr << "IMU file not found: " << imu_file_ << std::endl;
        return false;
    }

    std::ifstream file(imu_file_);
    timestamps.clear();
    gyro.clear();
    accel.clear();
    std::string line;
    
    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string token;
        std::vector<double> values;
        
        while (std::getline(iss, token, ',')) {
            values.push_back(std::stod(token));
        }
        
        if (values.size() >= 7) {
            double timestamp = values[0] / 1e9;
            Eigen::Vector3d g(values[1], values[2], values[3]);
            Eigen::Vector3d a(values[4], values[5], values[6]);
            
            timestamps.push_back(timestamp);
            gyro.push_back(g);
            accel.push_back(a);
        }
    }

    return true;
}

std::vector<fs::path> EuRoCDataset::getCam0Images() {
    std::vector<fs::path> images;
    if (!fs::exists(cam0_data_)) {
        return images;
    }

    for (const auto& entry : fs::directory_iterator(cam0_data_)) {
        if (entry.path().extension() == ".png") {
            images.push_back(entry.path());
        }
    }

    std::sort(images.begin(), images.end());
    return images;
}

std::vector<fs::path> EuRoCDataset::getCam1Images() {
    std::vector<fs::path> images;
    if (!fs::exists(cam1_data_)) {
        return images;
    }

    for (const auto& entry : fs::directory_iterator(cam1_data_)) {
        if (entry.path().extension() == ".png") {
            images.push_back(entry.path());
        }
    }

    std::sort(images.begin(), images.end());
    return images;
}

size_t EuRoCDataset::size() const {
    if (!fileExists(groundtruth_file_)) {
        return 0;
    }

    std::ifstream file(groundtruth_file_);
    size_t count = 0;
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        if (!line.empty()) {
            count++;
        }
    }
    
    return count;
}