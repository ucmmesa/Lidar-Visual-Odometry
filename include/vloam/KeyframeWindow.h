//
// Created by ubuntu on 2020/5/19.
//

#ifndef KEYFRAMEWINDOW_H
#define KEYFRAMEWINDOW_H

#include <vector>
#include <Eigen/Core>

#include <vloam/Keyframe.h>

namespace vloam
{
class KeyframeWindow
{
public:
    typedef std::shared_ptr<KeyframeWindow> Ptr;

    KeyframeWindow(int num_keyframe);
    ~KeyframeWindow();

    void add(Keyframe::Ptr keyframe);

    int size()
    { return kf_window_.size(); }

    std::vector<Keyframe::Ptr>::iterator begin()
    { return kf_window_.begin(); }
    std::vector<Keyframe::Ptr>::iterator end()
    { return kf_window_.end(); }

    std::vector<Keyframe::Ptr> &frames()
    { return kf_window_; }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    int num_keyframe_;
    std::vector<Keyframe::Ptr> kf_window_;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> H_;
};
} //namespace vloam

#endif //KEYFRAMEWINDOW_H
