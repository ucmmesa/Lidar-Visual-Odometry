//
// Created by ubuntu on 2020/3/9.
//

#include <vloam/KeyframeDB.h>

namespace vloam {

KeyframeDB::KeyframeDB()
{

}

KeyframeDB::~KeyframeDB()
{

}

void KeyframeDB::show_image_with_accum_points(size_t num_keyframe, size_t num_level)
{
    //    Keyframe::Ptr last_keyframe = (*(keyframeDB_.end()-1));
    Frame::Ptr last_frame = (*(keyframeDB_.end()-1))->frame();
    cv::Mat img_with_points = cv::Mat(cv::Size(last_frame->level(num_level).cols, last_frame->level(num_level).rows), CV_8UC3);
    cvtColor(last_frame->level(num_level), img_with_points, cv::COLOR_GRAY2BGR);

    const float scale = 1.0f/(1<<num_level);

    for(auto iter=keyframeDB_.end()-1; iter != keyframeDB_.end()-num_keyframe+1; --iter) {
        auto Twl = last_frame->Twc();
        auto Twi = (*iter)->frame()->Twc();
        auto Tli = Twl.inverse() * Twi;

        auto pc_i = (*iter)->frame()->pointcloud();

        for(auto pc_iter = pc_i.begin(); pc_iter != pc_i.end(); ++pc_iter) {
            Eigen::Vector3f xyz_i (pc_iter->x, pc_iter->y, pc_iter->z);
            Eigen::Vector3f xyz_l = Tli*xyz_i;

            if( last_frame->camera()->is_in_image(xyz_l, 2)) {
                Eigen::Vector2f uv_l;
                uv_l.noalias() = last_frame->camera()->xyz_to_uv(xyz_l) * scale;

                const int u_l_i = static_cast<int> (uv_l(0));
                const int v_l_i = static_cast<int> (uv_l(1));

                cv::circle(img_with_points, cv::Point(u_l_i, v_l_i), 1, cv::Scalar( 0.0, 0.0, 0.9 ), -1);
            }
        }

    }

    cv::namedWindow("image_with_accum_points",cv::WINDOW_NORMAL);
    //cv::imshow("image_with_accum_points",img_with_points);
    cv::waitKey(1);
}

}   // namespace vloam
