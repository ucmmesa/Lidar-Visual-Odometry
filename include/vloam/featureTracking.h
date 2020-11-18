//
// Created by ubuntu on 2020/4/2.
//

#ifndef FEATURETRACKING_H
#define FEATURETRACKING_H

#include <string>
#include <vector>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "vloam/Config.h"
#include "vloam/pointDefinition.h"
#include "vloam/Datatypes.h"

namespace vloam
{
typedef Eigen::unordered_map<int, ImagePoint> featureTrackResult;
} //namespace vloam
namespace vloam
{
class featureTracking
{
public:
    typedef std::shared_ptr<featureTracking> Ptr;

    featureTracking();
    ~featureTracking() = default;

    pcl::PointCloud<ImagePoint>::Ptr
    trackImage(double timestamp, const cv::Mat &imgdata);

    inline
    const featureTrackResult &featureTrackingResultCur() const
    {
        return imagePointCurMap_;
    }


private:
    inline
    double distance(cv::Point2f &pt1, cv::Point2f &pt2)
    {
        double dx = pt1.x - pt2.x;
        double dy = pt1.y - pt2.y;

        return std::sqrt(dx * dx + dy * dy);
    }

    inline
    bool inBorder(const cv::Point2f &pt)
    {
        const int BORDER_SIZE = 20;
        int img_x = cvRound(pt.x);
        int img_y = cvRound(pt.y);
        return BORDER_SIZE <= img_x && img_x < imageWidth_ - BORDER_SIZE && BORDER_SIZE <= img_y
            && img_y < imageHeight_ - BORDER_SIZE;
    }

    inline
    void reduceVector(std::vector<cv::Point2f> &v, const std::vector<uchar> &status)
    {
        int j = 0;
        for (size_t i = 0; i < v.size(); i++) {
            if (status[i])
                v[j++] = v[i];
        }
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:

    bool bsystemInited_ = false;
    bool bflowBack_ = true;
    int ishowSkipNum_ = 2;
    int ishowDSRate_ = 2;

    CameraModel::Ptr pcamera_;

    int imageWidth_, imageHeight_;
    cv::Size showSize_;
    cv::Mat mimageShow_;
    cv::Mat mharrisCur_;

    int imaxFeatureNumPerSubregion_ = 5;
    int ixSubregionNum_ = 28;
    int iySubregionNum_ = 6;
    int itotalSubregionNum_;
    int iMAXFEATURENUM_;
    int ixBoundary_ = 20;
    int iyBoundary_ = 20;
    double dsubregionWidth_;
    double dsubregionHeight_;
    double dmaxTrackDis_ = 255;

    int ifeaturesIndFromStart_ = 0;

    // feature tracking result
    cv::Mat mimageLast_, mimageCur_;
    double dtimeLast_, dtimeCur_;
    std::vector<cv::Point2f> vfeaturesLast_;
    std::vector<cv::Point2f> vfeaturesCur_;
    std::vector<int> vfeaturesInd_;
    std::vector<int> vsubregionFeatureNum_;
    int itotalFeatureNum_ = 0;
    pcl::PointCloud<ImagePoint>::Ptr pimagePointsLast_;
    pcl::PointCloud<ImagePoint>::Ptr pimagePointsCur_;

    featureTrackResult imagePointLastMap_;
    featureTrackResult imagePointCurMap_;
    // visualize image
    cv::Mat mimage_feature_;
};

} //namespace vloam

#endif //FEATURETRACKING_H
