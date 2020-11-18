//
// Created by ubuntu on 2020/5/19.
//

#include "vloam/KeyframeWindow.h"

namespace vloam
{
KeyframeWindow::KeyframeWindow(int num_keyframe)
    : num_keyframe_(num_keyframe)
{
    kf_window_.reserve(num_keyframe_);

    H_.resize(6 * num_keyframe_, 6 * num_keyframe_);
    H_.setZero(6 * num_keyframe_, 6 * num_keyframe_);
}

KeyframeWindow::~KeyframeWindow()
{

}

void KeyframeWindow::add(Keyframe::Ptr keyframe)
{
    if (kf_window_.size() < num_keyframe_)
        kf_window_.push_back(keyframe);
    else {
        kf_window_.erase(kf_window_.begin());
        kf_window_.push_back(keyframe);
    }

}
} // namespace vloam