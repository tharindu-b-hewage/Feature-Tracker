// Feature Tracker.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "FeatureTracker.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <string>

using namespace cv;

int main()
{
	/**
	* @function cornerHarris_Demo.cpp
	* @brief Demo code for detecting corners using Harris-Stephens method
	* @author OpenCV team
	*/

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

	using namespace cv;
	using namespace std;

	/*/-- Webcam--//
	cv::VideoCapture webcamStream(0);

	if (!webcamStream.isOpened())
	{
		std::cout << "Camera failure!\n";
		return -1;
	}

	while (true) // infinite loop 
	{
		cv::Mat frame0; webcamStream.read(frame0);

		//Locate Corners
		std::vector<cv::Point> _Corners = Panaroma::FeatureTracker::HarrisCornerPoints(frame0, 0.03, 0.5);

		for (int i = 1; i < 5000; i++)
		{
			std::cout << "tracking.. : " << i << std::endl;
			//Load next image
			cv::Mat frame1; webcamStream.read(frame1);
			cv::Mat imageSP = frame1;

			//Track next frame
			std::pair<std::vector<cv::Point>, std::vector<bool>> trackedCorners = Panaroma::FeatureTracker::KLTtracker(_Corners, frame0, frame1);

			std::vector<cv::Point> nextCorners = trackedCorners.first;
			std::vector<bool> index = trackedCorners.second;

			//draw results
			for (int i = 0; i < _Corners.size(); i++)
			{

				if (index[i])
				{
					//cv::line(image, _Corners[i], nextCorners[i], cv::Scalar(0, 255, 0), 2);

					cv::circle(imageSP, nextCorners[i], 4, cv::Scalar(0, 0, 255), 1);
				}

			}

			//Prepare for the next round
			_Corners = nextCorners;

			//Imtermediate results showing
			cv::imshow("intermediate results", imageSP);
			cv::waitKey(40);
		}
	}
	
	*///-- Webcam--//

	//--Test images
	//while (true) // infinite loop 
	//{
		cv::Mat frame0 = cv::imread("../../hotel.seq0.png", CV_LOAD_IMAGE_COLOR);
		//cv::Mat frame0 = cv::imread("../../images 2/0.png", CV_LOAD_IMAGE_COLOR);
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
		std::vector<cv::Point> _Corners = Panaroma::FeatureTracker::HarrisCornerPoints(frame0, 0.03, 1.5);

		for (int i = 1; i < 50; i++)
		{
			//Load next image
			std::cout << "Processing Image Number: " << i << "out of " << 50 << std::endl;
			frame1 = cv::imread("../../hotel.seq" + std::to_string(i) + ".png", CV_LOAD_IMAGE_COLOR);
			//frame1 = cv::imread("../../images 2/" + std::to_string(i) + ".png", CV_LOAD_IMAGE_COLOR);
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

				if (index[i])
				{
					cv::line(image, _Corners[i], nextCorners[i], cv::Scalar(0, 255, 0), 2);

					cv::circle(imageSP, nextCorners[i], 4, cv::Scalar(0, 0, 255), 1);
				}

			}

			//Prepare for the next round
			_Corners = nextCorners;

			//Imtermediate results showing
			if (/*i == 49 || i==1*/1)
			{
				cv::imshow("intermediate results"/*+std::to_string(i)*/, imageSP);
				cv:imwrite("../../KLTout/KLToutput_"+std::to_string(i)+".png",imageSP);
				cv::waitKey(40);
			}
		}

		cv::destroyAllWindows();
	//}
	//std::cout << "Processing Done..." << std::endl;

	//cv::imshow("frame1", image1);
	//cv::imshow("frame0", image);

	//cv::waitKey(0); // sfdfdd
	//cv::destroyAllWindows();
	
    
	/*cv::Mat frame0 = cv::imread("../../hotel.seq0.png", CV_LOAD_IMAGE_GRAYSCALE); 		//std::cout << "here\n";
	cv::Mat nonmaxima = Panaroma::FeatureTracker::nonMaximaSuppression(frame0, 5);
	cv::imshow("original", frame0);
	cv::imshow("supp", nonmaxima);
	cv::waitKey(0);*/
	return 0;
}

