#pragma once
#include<vector>
#include<cmath>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include  <iostream>

namespace Panaroma{
	class FeatureTracker
	{
	public:
		FeatureTracker();
		~FeatureTracker();
		static cv::Mat HarrisCorner(cv::Mat src, double alpha, double taur);
		static std::vector<cv::Point> HarrisCornerPoints(cv::Mat src, double alpha, double taur);
		static void nonMaximaSuppression(const cv::Mat& src, const int sz, cv::Mat& dst, const cv::Mat mask);
		static std::pair<std::vector<cv::Point>, std::vector<bool>> KLTtracker(std::vector<cv::Point> _inputFeaturePoint, cv::Mat _frame_t0, cv::Mat _frame_t1);
		static cv::Mat bilinearWindow(cv::Mat reference, double x, double y, int windowSize); // Cordination for the top left corner in window
	};
}


