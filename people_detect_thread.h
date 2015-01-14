#include <opencv2/opencv.hpp>
#include <vector>

namespace people {
void start_thread_people_detect();
void stop_thread_people_detect();
void put_frame(const cv::Mat& f);
void get_result(std::vector<cv::Rect>& r);
}
