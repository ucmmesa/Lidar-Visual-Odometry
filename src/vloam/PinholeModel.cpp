#include <vloam/PinholeModel.h>

namespace vloam
{

PinholeModel::PinholeModel
    (int width, int height,
     float fx, float fy, float cx, float cy,
     float d0, float d1, float d2, float d3, float d4
    )
    :
    CameraModel(width, height),
    fx_(fx), fy_(fy), cx_(cx), cy_(cy),
    distortion_(fabs(d0) > 1e-8),
    undist_map1_(height_, width_, CV_16SC2),
    undist_map2_(height_, width_, CV_16SC2)
{
    std::cerr << "[PinholeModel]\t " << distortion_ << std::endl;
    d_[0] = d0;
    d_[1] = d1;
    d_[2] = d2;
    d_[3] = d3;
    d_[4] = d4;
    cvK_ = (cv::Mat_<float>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
    cvD_ = (cv::Mat_<float>(1, 5) << d_[0], d_[1], d_[2], d_[3], d_[4]);

    cv::initUndistortRectifyMap(cvK_, cvD_, cv::Mat_<double>::eye(3, 3), cvK_,
                                cv::Size(width_, height_), CV_16SC2, undist_map1_, undist_map2_);

    K_ << fx_, 0.0, cx_,
        0.0, fy_, cy_,
        0.0, 0.0, 1.0;

    K_inv_ = K_.inverse();
}

PinholeModel::~PinholeModel()
{

}

bool PinholeModel::is_in_image(const Eigen::Vector3f &point, int boundary)
{
    Eigen::Vector2f uv = xyz_to_uv(point);
    int u = static_cast<int> (uv(0));
    int v = static_cast<int> (uv(1));

    if (u > 0 + boundary && u < width_ - boundary && v > 0 + boundary && v < height_ - boundary && point(2) > 1.0) {
        return true;
    }
    else {
        return false;
    }
}

bool PinholeModel::is_in_image(const int u, const int v, int boundary)
{
    if (u > 0 + boundary && u < width_ - boundary && v > 0 + boundary && v < height_ - boundary) {
        return true;
    }
    else {
        return false;
    }
}

bool PinholeModel::is_in_image(const Eigen::Vector2f &uv, int boundary)
{
    int u = static_cast<int> (uv(0));
    int v = static_cast<int> (uv(1));

    if (u > 0 + boundary && u < width_ - boundary && v > 0 + boundary && v < height_ - boundary) {
        return true;
    }
    else {
        return false;
    }
}

bool PinholeModel::is_in_image(const Eigen::Vector2f &uv, int boundary, float scale)
{
    int u = static_cast<int> (uv(0) * scale);
    int v = static_cast<int> (uv(1) * scale);

    if (u > 0 + boundary && u < static_cast<int> (float(width_) * scale) - boundary && v > 0 + boundary
        && v < static_cast<int> (float(height_) * scale) - boundary) {
        return true;
    }
    else {
        return false;
    }
}

bool PinholeModel::is_in_image(const POINT &point, int boundary)
{

}

Eigen::Vector2f PinholeModel::xyz_to_uv(const Eigen::Vector3f &xyz)
{
    float x = fx_ * xyz(0) + cx_ * xyz(2);
    float y = fy_ * xyz(1) + cy_ * xyz(2);
    float z = xyz(2);
    Eigen::Vector2f uv(x / z, y / z);

    if (!distortion_) {
        return uv;
    }
    else {
        float xx = xyz(0) / xyz(2);
        float yy = xyz(1) / xyz(2);
        float r2 = xx * xx + yy * yy;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float a1 = 2 * xx * yy;
        float a2 = r2 + 2 * xx * xx;
        float a3 = r2 + 2 * yy * yy;
        float cdist = 1 + (float) d_[0] * r2 + (float) d_[1] * r4 + (float) d_[4] * r6;
        float xd = xx * cdist + (float) d_[2] * a1 + (float) d_[3] * a2;
        float yd = yy * cdist + (float) d_[2] * a3 + (float) d_[3] * a1;
        Eigen::Vector2f uv_undist(xd * (float) fx_ + (float) cx_, yd * (float) fy_ + (float) cy_);

        return uv_undist;
    }
}

Eigen::Vector2f PinholeModel::xyz_to_uv(const POINT &xyz)
{
    float x = fx_ * xyz.x + cx_ * xyz.z;
    float y = fy_ * xyz.y + cy_ * xyz.z;
    float z = xyz.z;

    Eigen::Vector2f uv(x / z, y / z);

    if (!distortion_) {
        return uv;
    }
    else {
        float xx = xyz.x / xyz.z;
        float yy = xyz.y / xyz.z;
        float r2 = xx * xx + yy * yy;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float a1 = 2 * xx * yy;
        float a2 = r2 + 2 * xx * xx;
        float a3 = r2 + 2 * yy * yy;
        float cdist = 1 + (float) d_[0] * r2 + (float) d_[1] * r4 + (float) d_[4] * r6;
        float xd = xx * cdist + (float) d_[2] * a1 + (float) d_[3] * a2;
        float yd = yy * cdist + (float) d_[2] * a3 + (float) d_[3] * a1;
        Eigen::Vector2f uv_undist(xd * (float) fx_ + (float) cx_, yd * (float) fy_ + (float) cy_);

        return uv_undist;
    }
}

vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>
PinholeModel::pointcloud_to_uv(const PointCloud &pc, float scale)
{
    vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> uv_set;

    for (auto point:pc) {
        Eigen::Vector3f xyz(point.x, point.y, point.z);

        float x = fx_ * xyz(0) + cx_ * xyz(2);
        float y = fy_ * xyz(1) + cy_ * xyz(2);
        float z = xyz(2);
        Eigen::Vector2f uv(x / z, y / z);

        if (!distortion_) {
            uv_set.push_back(uv * scale);
        }
        else {
            float xx = xyz(0) / xyz(2);
            float yy = xyz(1) / xyz(2);
            float r2 = xx * xx + yy * yy;
            float r4 = r2 * r2;
            float r6 = r4 * r2;
            float a1 = 2 * xx * yy;
            float a2 = r2 + 2 * xx * xx;
            float a3 = r2 + 2 * yy * yy;
            float cdist = 1 + (float) d_[0] * r2 + (float) d_[1] * r4 + (float) d_[4] * r6;
            float xd = xx * cdist + (float) d_[2] * a1 + (float) d_[3] * a2;
            float yd = yy * cdist + (float) d_[2] * a3 + (float) d_[3] * a1;
            Eigen::Vector2f uv_undist(xd * (float) fx_ + (float) cx_, yd * (float) fy_ + (float) cy_);

            uv_set.push_back(uv_undist * scale);
        }
    }

    return uv_set;
}

void PinholeModel::undistort_image(const cv::Mat &raw, cv::Mat &rectified)
{
    if (distortion_) {
        cv::remap(raw, rectified, undist_map1_, undist_map2_, CV_INTER_LINEAR);
        std::cerr << "[PinholeModel]\t " << "undistorted image" << std::endl;
    }
    else
        rectified = raw.clone();
}

}   // namespace dedvo
