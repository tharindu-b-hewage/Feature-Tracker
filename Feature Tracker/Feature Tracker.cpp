// Feature Tracker.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

int main()
{
	cv::Mat image = cv::imread("dog.jpg");
	cv::imshow("image", image);
	cv::waitKey(0);
    return 0;
}

