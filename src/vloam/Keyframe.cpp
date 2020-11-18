//
// Created by ubuntu on 2020/3/9.
//

#include <vloam/Keyframe.h>

namespace vloam
{

Keyframe::Keyframe(Frame::Ptr frame, int id, CameraModel::Ptr camera)
    : id_(id), frame_(frame), first_connection_(true)
{
    camera_ = camera;
    pinhole_model_ = static_pointer_cast<PinholeModel>(camera_);
    point_sampling();
}

Keyframe::Keyframe(Frame::Ptr frame, CameraModel::Ptr camera)
    : id_(-1), frame_(frame), first_connection_(true)
{
    camera_ = camera;
    pinhole_model_ = static_pointer_cast<PinholeModel>(camera_);
    point_sampling();
}

Keyframe::~Keyframe()
{

}

// 这个主要是为了降低计算量
void Keyframe::point_sampling()
{
    int num_bucket_size = 10;
    //    vector<Point, Eigen::aligned_allocator<Eigen::Vector3f>> pointcloud = frame_->pointcloud();
    PointCloud pointcloud = frame_->pointcloud();

    vector<pair<float, POINT>> mag_point_bucket;
    mag_point_bucket.reserve(num_bucket_size);

    int num_out_points = 0;

    for (auto iter = pointcloud.begin(); iter != pointcloud.end(); ++iter) {

        //        if((*visibility_iter) == false) continue;

        Eigen::Vector2f uv = camera_->xyz_to_uv(*iter);

        if (frame_->camera()->is_in_image(uv, 4) && iter->z > 0 && iter->a != 0) {

            int u = static_cast<int> (uv(0));
            int v = static_cast<int> (uv(1));

            cv::Mat img = frame()->level(0);
            float dx = 0.5f * (img.at<float>(v, u + 1) - img.at<float>(v, u - 1));
            float dy = 0.5f * (img.at<float>(v + 1, u) - img.at<float>(v - 1, u));

            // if ( (dx*dx + dy*dy) > (25.0/ (255*255)))  // 400, 25,
            //     pointcloud_.push_back(xyz);
            pair<float, POINT> mag_point;
            mag_point = make_pair((dx * dx + dy * dy), (*iter));

            mag_point_bucket.push_back(mag_point);

            // 每当 bucket 存储的点超过10,需要从中选择梯度梯度响应值的最大的点压入到pointcloud_中
            if (mag_point_bucket.size() == num_bucket_size) {

                float max = -1;
                int idx;
                for (int i = 0; i < mag_point_bucket.size(); ++i) {

                    if (mag_point_bucket[i].first > max) {
                        max = mag_point_bucket[i].first;
                        idx = i;
                    }

                }

                if (max > (6.25 / (255.0 * 255.0)))  // 16.25
                    pointcloud_.push_back(mag_point_bucket[idx].second);

                mag_point_bucket.clear();
            }
        }
        else {
            num_out_points++;
        }

    }

    assert(num_out_points != 0 && "Points must be projected at all pyramid levels.");
    //    cerr << "[Keyframe]\t number of points in boudary : " << num_out_points << endl;
    //    frame_->pointcloud().clear();
}

// 计算两帧的共视程度, 通过将其他关键帧的3d点投影到当前帧,计算可以观察到的3d点与所有投影点的比率
float Keyframe::get_visible_ratio(const Keyframe::Ptr keyframe)
{
    Transformf Tij = frame_->Twc().inverse() * keyframe->frame()->Twc();

    int patch_halfsize_ = 2;
    const int border = patch_halfsize_ + 2;
    int current_level = 0;

    cv::Mat &current_img = frame_->level(current_level);

    const float scale = 1.0f / (1 << current_level);

    int visible_points = 0;

    for (auto iter = keyframe->pointcloud().begin(); iter != keyframe->pointcloud().end(); ++iter) {
        Eigen::Vector3f xyz_cur(iter->x, iter->y, iter->z);
        Eigen::Vector3f xyz_prev = Tij * xyz_cur;
        Eigen::Vector2f uv_prev;
        uv_prev.noalias() = camera_->xyz_to_uv(xyz_prev) * scale;

        const float u_prev_f = uv_prev(0);
        const float v_prev_f = uv_prev(1);
        const int u_prev_i = static_cast<int> (u_prev_f);
        const int v_prev_i = static_cast<int> (v_prev_f);

        if (u_prev_i - border < 0 || u_prev_i + border > current_img.cols || v_prev_i - border < 0
            || v_prev_i + border > current_img.rows || xyz_prev(2) <= 0)
            continue;

        visible_points++;

    }

    return static_cast<float> (visible_points) / static_cast<float> (keyframe->pointcloud().size());
}


// 可视化关键帧上的点云
void Keyframe::show_image_with_points(cv::Mat &img, size_t num_level)
{
    cv::Mat keyframe_with_points;// = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3);

    if (img.type() == CV_32FC1) {
        cvtColor(img, keyframe_with_points, cv::COLOR_GRAY2BGR);
    }
    else {
        //        keyframe_with_points = img;
        img.copyTo(keyframe_with_points);
    }

    //    cv::namedWindow("original_rgb",cv::WINDOW_NORMAL);
    //    cv::imshow("original_rgb",img_with_points);

    const float scale = 1.0f / (1 << num_level);

    int n = 0;
    for (auto iter = pointcloud_.begin(); iter != pointcloud_.end(); ++iter) {
        n++;
        if (n % 5 != 0) continue;

        Eigen::Vector3f xyz_ref(iter->x, iter->y, iter->z);
        Eigen::Vector2f uv_ref;
        uv_ref.noalias() = camera_->xyz_to_uv(xyz_ref) * scale;

        const float u_ref_f = uv_ref(0);
        const float v_ref_f = uv_ref(1);
        const int u_ref_i = static_cast<int> (u_ref_f);
        const int v_ref_i = static_cast<int> (v_ref_f);

        float v_min = 1.0;
        float v_max = 50.0;
        float dv = v_max - v_min;
        float v = xyz_ref(2);
        float r = 1.0;
        float g = 1.0;
        float b = 1.0;
        if (v < v_min) v = v_min;
        if (v > v_max) v = v_max;

        if (v < v_min + 0.25 * dv) {
            r = 0.0;
            g = 4 * (v - v_min) / dv;
        }
        else if (v < (v_min + 0.5 * dv)) {
            r = 0.0;
            b = 1 + 4 * (v_min + 0.25 * dv - v) / dv;
        }
        else if (v < (v_min + 0.75 * dv)) {
            r = 4 * (v - v_min - 0.5 * dv) / dv;
            b = 0.0;
        }
        else {
            g = 1 + 4 * (v_min + 0.75 * dv - v) / dv;
            b = 0.0;
        }

        //cv::circle(img_with_points, cv::Point(u_ref_i, v_ref_i), 0.1, cv::Scalar( static_cast<int> (r*255), static_cast<int> (g*255), static_cast<int> (b*255)), -1);
        cv::circle(keyframe_with_points, cv::Point(u_ref_i, v_ref_i), 3.5, cv::Scalar(r, g, b), -1);

    }

    cv::namedWindow("keyframe_with_points", cv::WINDOW_AUTOSIZE);
    // cv::imshow("keyframe_with_points", keyframe_with_points);
    //    cv::imwrite("img_with_points.png",img_with_points*255);
    cv::waitKey(1);
}

}   // namespace vloam
