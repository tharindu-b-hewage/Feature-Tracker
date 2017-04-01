#pragma once
#include<vector>
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
		cv::Mat HarrisCorner(cv::Mat src, double alpha, double taur);
		void nonMaximaSuppression(const cv::Mat& src, const int sz, cv::Mat& dst, const cv::Mat mask);
	};
}


