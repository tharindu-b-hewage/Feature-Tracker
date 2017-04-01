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
	Panaroma::FeatureTracker Tracker; 

	cv::Mat image = cv::imread("../../cornerTest3.png", CV_LOAD_IMAGE_COLOR);

	if (!image.data)
	{
		std::cout << "Image loading failed..." << std::endl;
		std::cin.get();
		return -1;
	}

	/*cv::imshow("temp", image); 
	cv::waitKey(0);// std::cin.get();*/
	std::cout << "Please Wait..." << std::endl;

	cv::Mat Corners = Tracker.HarrisCorner(image, 0.03, 0.08);


    return 0;
}

