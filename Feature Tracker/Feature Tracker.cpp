// Feature Tracker.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "FeatureTracker.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

int main()
{
	//Panaroma::FeatureTracker Tracker; 

	//cv::Mat image = cv::imread("../../cornerTest3.png", CV_LOAD_IMAGE_COLOR);

	cv::Mat frame0 = cv::imread("../../hotel.seq0.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat frame1 = cv::imread("../../hotel.seq4.png", CV_LOAD_IMAGE_COLOR);

	//std::cout << "*****: " << (float)frame0.at<uchar>(1.2,2.1)<< "| " << (float)frame0.at<uchar>(cv::Point(2, 1)) << std::endl; std::cin.get();

	if (!frame0.data || !frame1.data)
	{
		std::cout << "Image loading failed..." << std::endl;
		std::cin.get();
		return -1;
	}

	std::cout << "Please Wait..." << std::endl;

	std::vector<cv::Point> _Corners = Panaroma::FeatureTracker::HarrisCornerPoints(frame0, 0.03, 0.08);

	std::cout << "Corners passed..\n";

	std::pair<std::vector<cv::Point>, std::vector<bool>> trackedCorners = Panaroma::FeatureTracker::KLTtracker(_Corners, frame0, frame1);

	
	cv::Mat image = frame0;
	cv::Mat image1 = frame1;

	std::vector<cv::Point> nextCorners = trackedCorners.first;
	std::vector<bool> index = trackedCorners.second;

	for (int i = 0; i < _Corners.size(); i++)
	{
		std::cout << "i:"<<i<<" " << _Corners.size() << " " << nextCorners.size() << " " << index.size() << std::endl;
		cv::circle(image1, _Corners[i], 2, cv::Scalar(0, 0, 255), 3);
		cv::circle(image, _Corners[i], 2, cv::Scalar(0, 0, 255), 3);

		if(index[i])
			cv::line(image1, _Corners[i], nextCorners[i], cv::Scalar(0, 255, 0), 2);

	}

	std::cout << "Processing Done..." << std::endl;

	cv::imshow("frame1", image1);
	cv::imshow("frame0", image);

	cv::waitKey(0); // sfdfdd
	cv::destroyAllWindows();

    return 0;
}

