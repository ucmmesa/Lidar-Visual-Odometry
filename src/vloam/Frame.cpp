#include <gflags/gflags.h>
#include <glog/logging.h>

#include "vloam/Frame.h"
#include "vloam/Twist.h"
#include <aloam_velodyne/tic_toc.h>
#include <omp.h>

namespace vloam
{

    Frame::Frame(const cv::Mat &img)
    {
        initialize_img(img);
        initialize_pc();
    }

    Frame::Frame(const int64_t timestamp, const cv::Mat &img, PointCloud &pointcloud, CameraModel::Ptr camera)
        //    :utime_(timestamp), camera_(camera), num_levels_(Config::cfg()->num_levels())
        : utime_(timestamp), camera_(camera), pointcloud_(pointcloud), original_pointcloud_(pointcloud), num_levels_(Config::cfg()
                                                                                                                         ->num_levels()),
          max_level_(
              Config::cfg()->max_level())
    {
        initialize_img(img);
        initialize_pc();

        //    initialize_pc_OMP();
    }

    Frame::~Frame()
    {
        camera_ = nullptr;
    }

    Matrix3x4 Frame::Twlidar()
    {
        Matrix3x4 T_cam_liar = Config::cfg()->camlidar().extrinsic;

        Matrix3x4 T_w_liar;
        T_w_liar.block<3, 3>(0, 0) = Twc_.rotationMatrix() * T_cam_liar.block<3, 3>(0, 0);
        T_w_liar.block<3, 1>(0, 3) = Twc_.rotationMatrix() * T_cam_liar.block<3, 1>(0, 3) + Twc_.pos;

        return T_w_liar;
    }
    void Frame::Twc(const Transformf &Twc)
    {
        lock_guard<mutex> lg{Twc_mutex_};
        Twc_ = Twc;
    }

    cv::Mat &Frame::level(size_t idx)
    {
        return img_pyramid_[idx];
    }

    //vector<Point, Eigen::aligned_allocator<Eigen::Vector3f>>& Frame::pointcloud()
    //{
    //    return pointcloud_;
    //}
    pcl::PointCloud<pcl::PointXYZI> &Frame::xyzipc()
    {
        return xyzipc_;
    }
    PointCloud &Frame::pointcloud()
    {
        return pointcloud_;
    }

    PointCloud &Frame::original_pointcloud()
    {
        return original_pointcloud_;
    }

    void Frame::show_pointcloud()
    {
        for (size_t i = 0; i < pointcloud_.size(); i++)
        {
            cout << pointcloud_[i] << endl
                 << endl;
        }
    }

    void Frame::show_image_with_points(cv::Mat &img, size_t num_level)
    {
        cv::Mat img_with_points; // = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3);

        if (img.type() == CV_32FC1)
        {
            cvtColor(img, img_with_points, cv::COLOR_GRAY2BGR);
        }
        else
        {
            //        img_with_points = img;
            img.copyTo(img_with_points);
        }

        //    cv::namedWindow("original_rgb",cv::WINDOW_NORMAL);
        //    cv::imshow("original_rgb",img_with_points);

        const float scale = 1.0f / (1 << num_level);

        cerr << "Point size = " << pointcloud_.size() << endl;
        int n = 0;
        for (auto iter = pointcloud_.begin(); iter != pointcloud_.end(); ++iter)
        {
            n++;
            if (n % 5 != 0)
                continue;

            Eigen::Vector3f xyz_ref(iter->x, iter->y, iter->z);
            Eigen::Vector2f uv_ref;
            uv_ref.noalias() = camera_->xyz_to_uv(xyz_ref) * scale;

            const float u_ref_f = uv_ref(0);
            const float v_ref_f = uv_ref(1);
            const int u_ref_i = static_cast<int>(u_ref_f);
            const int v_ref_i = static_cast<int>(v_ref_f);

            float v_min = 1.0;
            float v_max = 50.0;
            float dv = v_max - v_min;
            float v = xyz_ref(2);
            float r = 1.0;
            float g = 1.0;
            float b = 1.0;
            if (v < v_min)
                v = v_min;
            if (v > v_max)
                v = v_max;

            if (v < v_min + 0.25 * dv)
            {
                r = 0.0;
                g = 4 * (v - v_min) / dv;
            }
            else if (v < (v_min + 0.5 * dv))
            {
                r = 0.0;
                b = 1 + 4 * (v_min + 0.25 * dv - v) / dv;
            }
            else if (v < (v_min + 0.75 * dv))
            {
                r = 4 * (v - v_min - 0.5 * dv) / dv;
                b = 0.0;
            }
            else
            {
                g = 1 + 4 * (v_min + 0.75 * dv - v) / dv;
                b = 0.0;
            }

            //        std::cout << "color: " << r << ", " << g << ", " << b << std::endl;
            //        iter->r = r;
            //        iter->g = g;
            //        iter->b = b;
            //cv::circle(img_with_points, cv::Point(u_ref_i, v_ref_i), 0.1, cv::Scalar( static_cast<int> (r*255), static_cast<int> (g*255), static_cast<int> (b*255)), -1);
            cv::circle(img_with_points, cv::Point(u_ref_i, v_ref_i), 3.5, cv::Scalar(r, g, b), -1);
        }

        cerr << "show iamge with points1" << endl;
        cv::namedWindow("image_with_points", cv::WINDOW_AUTOSIZE);
        cv::imshow("image_with_points", img_with_points);
        //    cv::imwrite("img_with_points.png",img_with_points*255);
         cv::waitKey(2);
        cerr << "show iamge with points2" << endl;
    }

    void Frame::save_image_with_points(size_t num_level, int id)
    {
        cv::Mat img_with_points; // = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3);

        if (original_img_.type() == CV_32FC1)
        {
            cvtColor(original_img_, img_with_points, cv::COLOR_GRAY2BGR);
        }
        else
        {
            img_with_points = original_img_;
        }

        //    cv::namedWindow("original_rgb",cv::WINDOW_NORMAL);
        //    cv::imshow("original_rgb",img_with_points);

        const float scale = 1.0f / (1 << num_level);

        int cnt = 0;

        for (auto iter = pointcloud_.begin(); iter != pointcloud_.end(); ++iter)
        {
            ++cnt;
            if (cnt % 4 > 0)
                continue;

            Eigen::Vector3f xyz_ref(iter->x, iter->y, iter->z);
            Eigen::Vector2f uv_ref;
            uv_ref.noalias() = camera_->xyz_to_uv(xyz_ref) * scale;

            const float u_ref_f = uv_ref(0);
            const float v_ref_f = uv_ref(1);
            const int u_ref_i = static_cast<int>(u_ref_f);
            const int v_ref_i = static_cast<int>(v_ref_f);

            float v_min = 1.0;
            float v_max = 15.0;
            float dv = v_max - v_min;
            float v = xyz_ref(2);
            float r = 1.0;
            float g = 1.0;
            float b = 1.0;
            if (v < v_min)
                v = v_min;
            if (v > v_max)
                v = v_max;

            if (v < v_min + 0.25 * dv)
            {
                r = 0.0;
                g = 4 * (v - v_min) / dv;
            }
            else if (v < (v_min + 0.5 * dv))
            {
                r = 0.0;
                b = 1 + 4 * (v_min + 0.25 * dv - v) / dv;
            }
            else if (v < (v_min + 0.75 * dv))
            {
                r = 4 * (v - v_min - 0.5 * dv) / dv;
                b = 0.0;
            }
            else
            {
                g = 1 + 4 * (v_min + 0.75 * dv - v) / dv;
                b = 0.0;
            }

            //cv::circle(img_with_points, cv::Point(u_ref_i, v_ref_i), 0.1, cv::Scalar( static_cast<int> (r*255), static_cast<int> (g*255), static_cast<int> (b*255)), -1);
            cv::circle(img_with_points, cv::Point(u_ref_i, v_ref_i), 2.0, cv::Scalar(r, g, b), -1);
        }

        //    cv::namedWindow("image_with_points",cv::WINDOW_NORMAL);
        cv::imshow("image_with_points", img_with_points);
        //    cv::imwrite("img_with_points.png",img_with_points*255);
        cv::waitKey(1);

        string path = "samsung/";
        string str_frame_id = to_string(id);
        string f_name = path + str_frame_id + ".png";
        cv::imwrite(f_name, img_with_points * 255);
    }

    void Frame::initialize_img(const cv::Mat img)
    {
        // init image
        if (img.empty())
        {
            throw std::runtime_error("[Frame]\t Input image have to CV_8UC1 type.");
        }

        cv::Mat gray;

        if (img.type() != CV_8UC1)
        {
            cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            original_img_ = move(img);
            //        img.copyTo(original_img_);
        }
        else
        {
            gray = img;
            cvtColor(img, original_img_, cv::COLOR_GRAY2BGR);
        }

        //    cv::Mat tmp;
        //    camera_->undistort_image(gray, tmp);
        //    gray = tmp;

        original_img_.convertTo(original_img_, CV_32FC3, 1.0 / 255);
        gray.convertTo(gray, CV_32FC1, 1.0 / 255);

        //    gray.convertTo(gray, CV_32FC1, 1.0);
        //    cv::namedWindow("org_img",cv::WINDOW_AUTOSIZE);
        //    cv::imshow("org_img",original_img_);

        create_image_pyramid(gray, num_levels_, img_pyramid_);
    }

    // 这一块应该可以硬件加速
    void Frame::initialize_pc()
    {
        //init point cloud
        float scale = 1.0f / (1 << max_level_);
        int border = 4;

        auto tic = TicToc();

        PointCloud pc;
        pc.reserve(pointcloud_.size());
        xyzipc_.clear();
        xyzipc_.reserve(pointcloud_.size());
        LOG(INFO) << "step1";
        for (auto iter = pointcloud_.begin(); iter != pointcloud_.end(); ++iter)
        {

            Eigen::Vector2f uv = camera_->xyz_to_uv(*iter);

            if (camera_->is_in_image(uv, border, scale))
            { // && iter->z < 5.0

                int u = static_cast<int>(uv(0));
                int v = static_cast<int>(uv(1));

                cv::Vec3f bgr = original_img_.at<cv::Vec3f>(v, u);

                iter->r = static_cast<uint8_t>(bgr[2] * 255.0);
                iter->g = static_cast<uint8_t>(bgr[1] * 255.0);
                iter->b = static_cast<uint8_t>(bgr[0] * 255.0);
                iter->a = 1.0;
            }
            else
            {
                iter->r = static_cast<uint8_t>(0.0);
                iter->g = static_cast<uint8_t>(255.0);
                iter->b = static_cast<uint8_t>(0.0);
                iter->a = 1;
            }

            if (iter->z > 0.0)
            {
                pc.push_back(*iter);
                pcl::PointXYZI point;
                point.intensity = iter->z;
                point.x = iter->x * 10.f / iter->z;
                point.y = iter->y * 10.f / iter->z;
                point.z = 10.f;
                xyzipc_.push_back(point);
            }
        }

        LOG(INFO) << "[Frame]\t Computation time of init_pc : " << tic.toc() << "ms";
        LOG(INFO) << "step2";
        LOG(INFO) << "the number of sparse depth point:" << pc.size();
        pcl::copyPointCloud(pc, pointcloud_);
        LOG(INFO) << "step3";

        //    visible_points_.resize(pointcloud_.size(), false);

        //    visible_points(7);
        show_image_with_points(original_img_, 0);
        original_img_.release();
        LOG(INFO) << "[Frame]:\t realse image memory";
    }

    void Frame::initialize_pc_OMP()
    {

        float scale = 1.0f / (1 << max_level_);
        int border = 4;

        auto tic = TicToc();

        PointCloud pc;

        omp_set_num_threads(6);
#pragma omp parallel
        {
            PointCloud pc_private;

#pragma omp for nowait schedule(static)
            //        for(auto iter=pointcloud_.begin(); iter!=pointcloud_.end(); ++iter) {
            for (size_t i = 0; i < pointcloud_.size(); ++i)
            {
                Eigen::Vector2f uv = camera_->xyz_to_uv(pointcloud_[i]);

                if (camera_->is_in_image(uv, border, scale) && pointcloud_[i].z > 0.0)
                {

                    int u = static_cast<int>(uv(0));
                    int v = static_cast<int>(uv(1));

                    cv::Vec3f bgr = original_img_.at<cv::Vec3f>(v, u);

                    pointcloud_[i].r = static_cast<uint8_t>(bgr[2] * 255.0);
                    pointcloud_[i].g = static_cast<uint8_t>(bgr[1] * 255.0);
                    pointcloud_[i].b = static_cast<uint8_t>(bgr[0] * 255.0);

                    pc_private.push_back(pointcloud_[i]);
                }
            }
            //        cerr << "[Frame]\t num_omp_thread : " << omp_get_num_threads() << endl;

#pragma omp for schedule(static) ordered
            for (int i = 0; i < omp_get_num_threads(); i++)
            {
#pragma omp ordered
                pc.insert(pc.end(), pc_private.begin(), pc_private.end());
            }
        }

        cerr << "[Frame]\t Computation time of init_pc : " << tic.toc() << " ms" << endl;

        pcl::copyPointCloud(pc, pointcloud_);

        original_img_.release();
    }

    template <typename T>
    static void pyrDownMeanSmooth(const cv::Mat &in, cv::Mat &out)
    {
        out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

        auto getPixel = [](const cv::Mat &img, int y, int x) -> T & {
            return ((T *)(img.data + img.step.p[0] * y))[x];
        };

        // #pragma omp parallel for collapse(2)
        for (int y = 0; y < out.rows; ++y)
        {
            for (int x = 0; x < out.cols; ++x)
            {
                int x0 = x * 2;
                int x1 = x0 + 1;
                int y0 = y * 2;
                int y1 = y0 + 1;

                // out.at<T>(y, x) = (T)((in.at<T>(y0, x0) + in.at<T>(y0, x1) + in.at<T>(y1, x0) + in.at<T>(y1, x1)) / 4.0f);

                getPixel(out, y, x) =
                    (T)(getPixel(in, y0, x0) + getPixel(in, y0, x1) + getPixel(in, y1, x0) + getPixel(in, y1, x1) / 4.0f);
            }
        }
    }

    void create_image_pyramid(const cv::Mat &img_level_0, int n_levels, ImgPyramid &pyramid)
    {
        pyramid.resize(n_levels);
        pyramid[0] = img_level_0;

        for (int i = 1; i < n_levels; ++i)
        {
            pyramid[i] = cv::Mat(pyramid[i - 1].rows / 2, pyramid[i - 1].cols / 2, CV_32FC1);
            pyrDownMeanSmooth<float>(pyramid[i - 1], pyramid[i]);
        }
    }

} // namespace vloam
