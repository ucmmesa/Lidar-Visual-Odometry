//
// Created by ubuntu on 2020/4/2.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "vloam/featureTracking.h"

namespace vloam
{
    featureTracking::featureTracking()
    {
        // step1: set parameters
        const auto &featuretrackingcfg = Config::cfg()->featuretracking();

        bflowBack_ = featuretrackingcfg.bflowBack;
        imageWidth_ = featuretrackingcfg.imageWidth;
        imageHeight_ = featuretrackingcfg.imageHeight;

        ishowSkipNum_ = featuretrackingcfg.ishowSkipNum;
        ishowDSRate_ = featuretrackingcfg.ishowDSRate;

        showSize_ = cv::Size(imageWidth_ / ishowDSRate_, imageHeight_ / ishowDSRate_);

        imaxFeatureNumPerSubregion_ = featuretrackingcfg.imaxFeatureNumPerSubregion;

        ixSubregionNum_ = featuretrackingcfg.ixSubregionNum;
        iySubregionNum_ = featuretrackingcfg.iySubregionNum;

        itotalSubregionNum_ = ixSubregionNum_ * iySubregionNum_;

        int maxfeaturenum = imaxFeatureNumPerSubregion_ * itotalSubregionNum_;

        vfeaturesCur_.resize(maxfeaturenum);
        vfeaturesLast_.resize(maxfeaturenum);
        vfeaturesInd_.resize(maxfeaturenum);
        vsubregionFeatureNum_.resize(itotalSubregionNum_);

        ixBoundary_ = featuretrackingcfg.ixBoundary;
        iyBoundary_ = featuretrackingcfg.iyBoundary;

        dsubregionWidth_ = (double)(imageWidth_ - 2 * ixBoundary_) / (double)ixSubregionNum_;
        dsubregionHeight_ = (double)(imageHeight_ - 2 * iyBoundary_) / (double)iySubregionNum_;
        dmaxTrackDis_ = featuretrackingcfg.dmaxTrackDis;

        // step2 获取相机的配置文件
        pcamera_ = Config::cfg()->camera();

        pimagePointsLast_.reset(new pcl::PointCloud<ImagePoint>());
        pimagePointsCur_.reset(new pcl::PointCloud<ImagePoint>());

        // step3 print Config
        LOG(INFO) << "\n    ------------------------------\n"
                  << "          [featuretracking]        \n"
                  << "        bflowBack_: " << (bflowBack_ ? "true" : "false \n") << "\n"
                  << "        image shape: (" << imageHeight_ << " , " << imageWidth_ << " )\n"
                  << "        showSkipNum: " << ishowSkipNum_ << "\n"
                  << "        showDSRate: " << ishowDSRate_ << "\n"
                  << "        showSize: (" << showSize_.height << " , " << showSize_.width << " )\n"
                  << "        maxFeatureNumPerSubregion: " << imaxFeatureNumPerSubregion_ << "\n"
                  << "        xSubregionNum: " << ixSubregionNum_ << "\n"
                  << "        ySubregionNum: " << iySubregionNum_ << "\n"
                  << "        totalSubregionNum: " << itotalSubregionNum_ << "\n"
                  << "        maxfeaturenum: " << maxfeaturenum << "\n"
                  << "        xBoundary: " << ixBoundary_ << "\n"
                  << "        yBoundary: " << iyBoundary_ << "\n"
                  << "        subregionWidth: " << dsubregionWidth_ << "\n"
                  << "        subregionHeight: " << dsubregionHeight_ << "\n"
                  << "        maxTrackDis: " << dmaxTrackDis_ << "\n"
                  << "        fx: " << pcamera_->fx() << " fy: " << pcamera_->fy() << "\n"
                  << "        cx: " << pcamera_->cx() << " cy: " << pcamera_->cy() << "\n"
                  << "    ------------------------------";
    }

    // TODO: 实现追踪特征的可视化, 同时需要发布追踪的结果
    pcl::PointCloud<ImagePoint>::Ptr
    featureTracking::trackImage(double timestamp, const cv::Mat &imgdata)
    {
        LOG(INFO) << "[featureTracking] trackImage";
        if (imgdata.empty())
        {
            LOG(ERROR) << "[featureTracking] cannot load empty image";
            return nullptr;
        }

        //step1 swap and set timestamp
        dtimeLast_ = dtimeCur_;
        dtimeCur_ = timestamp;

        //step2 swap and set image data
        cv::swap(mimageLast_, mimageCur_);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(0.4);
        clahe->apply(imgdata, mimageCur_);
        pcamera_->undistort_image(mimageCur_, mimageCur_);
        // cv::imshow("rectifyimage", mimageCur_);
        // cv::waitKey(1);
        //step3 set visual image
        cv::resize(mimageCur_, mimageShow_, showSize_);
        //step4 detect harris core
        cv::cornerHarris(mimageShow_, mharrisCur_, 3, 3, 0.03);
        //step5 swap OpenCV and PCL format feature point
        vfeaturesLast_.swap(vfeaturesCur_);
        pimagePointsLast_.swap(pimagePointsCur_);

        pimagePointsCur_->clear();

        imagePointLastMap_.swap(imagePointCurMap_);
        imagePointCurMap_.clear();

        // first image detect feature
        if (!bsystemInited_)
        {
            bsystemInited_ = true;

            // 清空每个区域的subregion追踪的特征点
            for (int i = 0; i < itotalSubregionNum_; i++)
            {
                vsubregionFeatureNum_[i] = 0;
            }

            ImagePoint point;
            // int recordFeatureNum = itotalFeatureNum_;
            //　添加特征点, 是在上一阵添加特征点
            for (int i = 0; i < iySubregionNum_; i++)
            {
                for (int j = 0; j < ixSubregionNum_; j++)
                {
                    int ind = ixSubregionNum_ * i + j;
                    int numToFind = imaxFeatureNumPerSubregion_ - vsubregionFeatureNum_[ind];

                    if (numToFind > 0)
                    {
                        int subregionLeft = ixBoundary_ + (int)(dsubregionWidth_ * j);
                        int subregionTop = iyBoundary_ + (int)(dsubregionHeight_ * i);
                        auto subregion =
                            cv::Rect(subregionLeft, subregionTop, (int)dsubregionWidth_, (int)dsubregionHeight_);

                        cv::Mat image_roi = mimageCur_(subregion);

                        std::vector<cv::Point2f> corners;
                        std::vector<cv::KeyPoint> kpt1;
                        corners.clear();
                        kpt1.clear();
                        cv::FAST(image_roi, kpt1, 30);
                        if (kpt1.size() < 2)
                        {
                            kpt1.clear();
                            cv::FAST(image_roi, kpt1, 15);
                        }
                        std::sort(kpt1.begin(), kpt1.end(),
                                  [](const cv::KeyPoint &a, const cv::KeyPoint &b) -> bool {
                                      return a.response > b.response;
                                  });
                        for (size_t i = 0; i < kpt1.size() && i < numToFind; i++)
                        {
                            corners.push_back(kpt1[i].pt);
                        }

                        numToFind = corners.size();
                        for (size_t t = 0; t < corners.size(); t++)
                        {
                            vfeaturesCur_[itotalFeatureNum_ + t] = corners[t];
                        }
                        // 如果提取到特征点，将会
                        int numFound = 0;
                        for (int k = 0; k < numToFind; k++)
                        {
                            vfeaturesCur_[itotalFeatureNum_ + k].x += subregionLeft;
                            vfeaturesCur_[itotalFeatureNum_ + k].y += subregionTop;
                            int xInd = (vfeaturesCur_[itotalFeatureNum_ + k].x + 0.5) / ishowDSRate_;
                            int yInd = (vfeaturesCur_[itotalFeatureNum_ + k].y + 0.5) / ishowDSRate_;

                            // harris角点特征值的检测
                            // if (((float *)(mharrisCur_.data + mharrisCur_.elemSize() * yInd))[xInd] > 1e-6)
                            if (((float *)(mharrisCur_.data + mharrisCur_.elemSize() * yInd))[xInd] > 0.0)
                            {
                                vfeaturesCur_[itotalFeatureNum_ + numFound] = vfeaturesCur_[itotalFeatureNum_ + k];
                                vfeaturesInd_[itotalFeatureNum_ + numFound] = ifeaturesIndFromStart_;

                                point.u = (vfeaturesCur_[itotalFeatureNum_ + numFound].x - pcamera_->cx()) / pcamera_->fx();
                                point.v = (vfeaturesCur_[itotalFeatureNum_ + numFound].y - pcamera_->cy()) / pcamera_->fy();
                                point.ind = vfeaturesInd_[itotalFeatureNum_ + numFound];

                                pimagePointsCur_->push_back(point);
                                imagePointCurMap_[point.ind] = point;

                                numFound++;               // 表示新添加的特征点
                                ifeaturesIndFromStart_++; // 新特征点的id号
                            }
                        } // add feature end

                        itotalFeatureNum_ += numFound;
                        vsubregionFeatureNum_[ind] += numFound;
                    } // deal with condition where need extract feature
                }
            }
        }
        else // tracking and detect
        {
            std::vector<uchar> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(mimageLast_,
                                     mimageCur_,
                                     vfeaturesLast_,
                                     vfeaturesCur_,
                                     status,
                                     err,
                                     cv::Size(25, 25),
                                     4,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
            // begin flowBack
            if (bflowBack_)
            {
                std::vector<uchar> reverse_status;
                std::vector<cv::Point2f> reverse_pts = vfeaturesLast_;
                cv::calcOpticalFlowPyrLK(mimageCur_,
                                         mimageLast_,
                                         vfeaturesCur_,
                                         reverse_pts,
                                         reverse_status,
                                         err,
                                         cv::Size(28, 28),
                                         2,
                                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

                for (size_t i = 0; i < status.size(); i++)
                {
                    if (status[i] && reverse_status[i] && distance(vfeaturesLast_[i], reverse_pts[i]) <= 1.0)
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0;
                }

            } // end flowback

            for (size_t i = 0; i < vfeaturesCur_.size(); i++)
            {
                if (status[i] && !inBorder(vfeaturesCur_[i]))
                    status[i] = 0;
            }

            // 清空每个区域的subregion追踪的特征点
            for (int i = 0; i < itotalSubregionNum_; i++)
            {
                vsubregionFeatureNum_[i] = 0;
            }

            ImagePoint point;     // temp variable
            int featureCount = 0; // the counter of sucessfully tracked feature
            cv::cvtColor(mimageCur_, mimage_feature_, cv::COLOR_GRAY2BGR);

            for (int i = 0; i < itotalFeatureNum_; i++)
            {
                double trackDist =
                    (vfeaturesCur_[i].x - vfeaturesLast_[i].x) * (vfeaturesCur_[i].x - vfeaturesLast_[i].x) +
                    (vfeaturesCur_[i].y - vfeaturesLast_[i].y) * (vfeaturesCur_[i].y - vfeaturesLast_[i].y);

                // determine track result is valid, 追踪的结果需要在图像边界内
                if (!(vfeaturesCur_[i].x < ixBoundary_ || vfeaturesCur_[i].x > imageWidth_ - ixBoundary_ || vfeaturesCur_[i].y < iyBoundary_ || vfeaturesCur_[i].y > imageHeight_ - iyBoundary_ || status[i] == 0))
                {

                    int xInd = (int)((vfeaturesCur_[i].x - ixBoundary_) / dsubregionWidth_);
                    int yInd = (int)((vfeaturesCur_[i].y - iyBoundary_) / dsubregionHeight_);
                    int ind = ixSubregionNum_ * yInd + xInd;

                    if (vsubregionFeatureNum_[ind] < imaxFeatureNumPerSubregion_)
                    {
                        vfeaturesLast_[featureCount] = vfeaturesLast_[i];
                        vfeaturesCur_[featureCount] = vfeaturesCur_[i];
                        vfeaturesInd_[featureCount] = vfeaturesInd_[i];
                        // TODO: visualize featuretracking result

                        cv::circle(mimage_feature_,
                                   vfeaturesCur_[featureCount],
                                   6,
                                   cv::Scalar(0, 0, 255));
                        cv::line(mimage_feature_,
                                 vfeaturesLast_[featureCount],
                                 vfeaturesCur_[featureCount],
                                 cv::Scalar(0, 255, 0),
                                 2);
                        // 输出结果
                        point.u = (vfeaturesCur_[featureCount].x - pcamera_->cx()) / pcamera_->fx();
                        point.v = (vfeaturesCur_[featureCount].y - pcamera_->cy()) / pcamera_->fy();
                        point.ind = vfeaturesInd_[featureCount];
                        pimagePointsCur_->push_back(point);
                        imagePointCurMap_[point.ind] = point;

                        featureCount++;
                        vsubregionFeatureNum_[ind]++;
                    }
                } // determine track result is valid end
            }     // add all tracked successfully feature end

            itotalFeatureNum_ = featureCount; // reset feature count
            LOG(INFO) << "itotalFeatureNum: " << itotalFeatureNum_;
            for (int i = 0; i < iySubregionNum_; i++)
            {
                for (int j = 0; j < ixSubregionNum_; j++)
                {

                    int ind = ixSubregionNum_ * i + j;
                    // LOG(INFO) << "( " << i << " , " << j << " ) " << vsubregionFeatureNum_[ind];
                    int numToFind = imaxFeatureNumPerSubregion_ - vsubregionFeatureNum_[ind];

                    if (numToFind > 0)
                    {
                        int subregionLeft = ixBoundary_ + (int)(dsubregionWidth_ * j);
                        int subregionTop = iyBoundary_ + (int)(dsubregionHeight_ * i);
                        auto subregion =
                            cv::Rect(subregionLeft, subregionTop, (int)dsubregionWidth_, (int)dsubregionHeight_);

                        cv::Mat image_roi = mimageCur_(subregion);

                        std::vector<cv::Point2f> corners;
                        std::vector<cv::KeyPoint> kpt1;

                        corners.clear();
                        kpt1.clear();

                        cv::FAST(image_roi, kpt1, 30);
                        if (kpt1.size() < 2)
                        {
                            kpt1.clear();
                            cv::FAST(image_roi, kpt1, 10);
                        }

                        std::sort(kpt1.begin(), kpt1.end(),
                                  [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
                                      return a.response > b.response;
                                  });
                        for (size_t i = 0; i < kpt1.size() && i < numToFind; i++)
                        {
                            corners.push_back(kpt1[i].pt);
                        }

                        for (size_t t = 0; t < corners.size(); t++)
                        {
                            vfeaturesCur_[itotalFeatureNum_ + t] = corners[t];
                        }
                        numToFind = corners.size();
                        // LOG(INFO) << "numToFind: " << numToFind;
                        int numFound = 0;
                        // begin add subregion feature
                        for (int k = 0; k < numToFind; k++)
                        {
                            vfeaturesCur_[itotalFeatureNum_ + k].x += subregionLeft;
                            vfeaturesCur_[itotalFeatureNum_ + k].y += subregionTop;

                            int xInd = (vfeaturesCur_[itotalFeatureNum_ + k].x + 0.5) / ishowDSRate_;
                            int yInd = (vfeaturesCur_[itotalFeatureNum_ + k].y + 0.5) / ishowDSRate_;

                            // harris core evaluation
                            // harris角点特征值的检测
                            // if (((float *)(mharrisCur_.data + mharrisCur_.elemSize() * yInd))[xInd] > 1.0e-5)
                            if (((float *)(mharrisCur_.data + mharrisCur_.elemSize() * yInd))[xInd] > 0.0)
                            {
                                vfeaturesCur_[itotalFeatureNum_ + numFound] = vfeaturesCur_[itotalFeatureNum_ + k];
                                vfeaturesInd_[itotalFeatureNum_ + numFound] = ifeaturesIndFromStart_;
                                // output add new feature
                                // 输出结果
                                point.u = (vfeaturesCur_[itotalFeatureNum_ + numFound].x - pcamera_->cx()) / pcamera_->fx();
                                point.v = (vfeaturesCur_[itotalFeatureNum_ + numFound].y - pcamera_->cy()) / pcamera_->fy();
                                point.ind = vfeaturesInd_[itotalFeatureNum_ + numFound];

                                pimagePointsCur_->push_back(point);
                                imagePointCurMap_[point.ind] = point;

                                cv::circle(mimage_feature_,
                                           vfeaturesCur_[itotalFeatureNum_ + numFound],
                                           6,
                                           cv::Scalar(255, 0, 0));
                                numFound++;               // 表示新添加的特征点
                                ifeaturesIndFromStart_++; // 新特征点的id号
                            }
                        } // add region feature end
                        // LOG(INFO) << "numFound: " << numFound;
                        itotalFeatureNum_ += numFound;
                        vsubregionFeatureNum_[ind] += numFound;
                    }
                }
            } // add all feature end

            cv::imshow("featuretracking result", mimage_feature_);
            cv::waitKey(1);
        } //session continues track feature

        LOG(INFO) << "[featureTracking] trackImage successfully";
        return pimagePointsCur_;
    } // function trackImage

} //namespace vloam