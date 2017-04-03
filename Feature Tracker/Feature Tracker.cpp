// Feature Tracker.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "FeatureTracker.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <string>

int main()
{
	//Panaroma::FeatureTracker Tracker; 

	//cv::Mat image = cv::imread("../../cornerTest3.png", CV_LOAD_IMAGE_COLOR);

	cv::Mat frame0 = cv::imread("../../hotel.seq0.png", CV_LOAD_IMAGE_COLOR);
	if (!frame0.data)
	{
		std::cout << "Image loading failed..." << std::endl;
		std::cin.get();
		return -1;
	}

	cv::Mat frame1;
	cv::Mat image = frame0;

	std::cout << "please wait..." << std::endl;

	//Locate Corners
	std::vector<cv::Point> _Corners = Panaroma::FeatureTracker::HarrisCornerPoints(frame0, 0.03, 0.08);

	for (int i = 1; i < 50; i++)
	{
		//Load next image
		std::cout << "Processing Image Number: " << i <<"out of "<<50<< std::endl;
		frame1 = cv::imread("../../hotel.seq"+std::to_string(i)+".png", CV_LOAD_IMAGE_COLOR);
		cv::Mat imageSP = frame1;
		if (!frame1.data)
		{
			std::cout << "Image loading failed..." << std::endl;
			std::cin.get();
			return -1;
		}

		//Track next frame
		std::pair<std::vector<cv::Point>, std::vector<bool>> trackedCorners = Panaroma::FeatureTracker::KLTtracker(_Corners, frame0, frame1);
		
		std::vector<cv::Point> nextCorners = trackedCorners.first;
		std::vector<bool> index = trackedCorners.second;

		//draw results
		for (int i = 0; i < _Corners.size(); i++)
		{
			//std::cout << "i:" << i << " " << _Corners.size() << " " << nextCorners.size() << " " << index.size() << std::endl;
			//cv::circle(image, _Corners[i], 5, cv::Scalar(0, 0, 255), 2);

			if (index[i])
			{
				cv::line(image, _Corners[i], nextCorners[i], cv::Scalar(0, 255, 0), 2);

				cv::circle(imageSP, nextCorners[i], 4, cv::Scalar(0, 0, 255), 1);
			}		

		}

		//Prepare for the next round
		_Corners = nextCorners;

		//Imtermediate results showing
		cv::imshow("intermediate results", imageSP);
		cv::waitKey(300);
		//cv::destroyAllWindows();
	}

	std::cout << "Processing Done..." << std::endl;

	//cv::imshow("frame1", image1);
	cv::imshow("frame0", image);

	cv::waitKey(0); // sfdfdd
	cv::destroyAllWindows();

    return 0;
}

