#pragma once
#include <thread>
#include <mutex>
#include "Keyframe.h"

namespace vloam
{

class KeyframeDB
{
public:
    typedef shared_ptr<KeyframeDB> Ptr;

    KeyframeDB();
    ~KeyframeDB();
    void add(Keyframe::Ptr keyframe)
    {
        unique_lock<mutex> ulock{mtx_DB};
        keyframeDB_.push_back(keyframe);
        ulock.unlock();
        //        string path = "samsung/";
        //        string str_frame_id = to_string(keyframe->id());
        //        string f_name = path+str_frame_id + ".png";
        //        cv::imwrite(f_name, keyframe->frame()->original_img()*255);
        //        usleep(10);
        //        keyframe->frame()->original_img().release();

        //        keyframe->frame()->save_image_with_points(0, keyframe->id());
        usleep(10);
        //        keyframe->frame()->original_img().release();
    }

    Keyframe::Ptr latest_keyframe()
    {
        return *(keyframeDB_.end() - 1);
    }

    void latest_keyframe(vector<Keyframe::Ptr> &keyframe_window, int n)
    {

        for (int i = n; i > 0; --i)
        {
            keyframe_window.push_back(*(keyframeDB_.end() - i));
        }
    }

    void keyframe_set(vector<Keyframe::Ptr> &keyframe_window, int idx, int n)
    {

        for (int i = n; i > 0; --i)
        {
            keyframe_window.push_back(keyframeDB_[idx - i]);
        }
    }

    void connected_keyframe(vector<Keyframe::Ptr> &connected_keyframe, int n)
    {

        for (int i = n; i > 1; --i)
        {
            connected_keyframe.push_back(*(keyframeDB_.end() - i));
        }
    }

    int size()
    {
        return keyframeDB_.size();
    }

    vector<Keyframe::Ptr>::iterator begin() { return keyframeDB_.begin(); }
    vector<Keyframe::Ptr>::iterator end() { return keyframeDB_.end(); }
    vector<Keyframe::Ptr> &keyframeDB() { return keyframeDB_; }
    void show_image_with_accum_points(size_t num_keyframe, size_t num_level);

    mutex mtx_DB;

private:
    vector<Keyframe::Ptr> keyframeDB_;
}; //class keyframeDB

} // namespace vloam
