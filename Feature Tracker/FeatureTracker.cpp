#include "FeatureTracker.h"
#define KLT_ACCURACY_THRESHOLD 0.08

using namespace cv;

namespace Panaroma {

	FeatureTracker::FeatureTracker()
	{
	}

	FeatureTracker::~FeatureTracker()
	{
	}

	cv::Mat FeatureTracker::HarrisCorner(cv::Mat src, double alpha, double taur)
	{
		Mat src_gray_temp, src_gray; cvtColor(src, src_gray_temp, CV_BGR2GRAY); src_gray_temp.convertTo(src_gray, CV_64FC1, 1/255.0); //std::cout << (double)src_gray.at<double>(0, 0) << "   |Here***\n"; std::cin.get();
		Mat Ix; Sobel(src_gray, Ix, -1, 1, 0); 
		Mat Iy; Sobel(src_gray, Iy, -1, 0, 1);
		Mat IxIy = Ix.mul(Iy), Ix2 = Ix.mul(Ix), Iy2 = Iy.mul(Iy);
		Mat G; GaussianBlur(src_gray, G, Size(3, 3), 5, 5);
		Mat G_Ix2 = Ix2.mul(G), G_Iy2 = Iy2.mul(G), G_IxIy = IxIy.mul(G);
		//cv::imshow("temp", G_Iy2); cv::waitKey(0); //std::cin.get();
		int WIN_SIZE = 5;
		Mat Corners = Mat::zeros(src.size(), CV_8UC1); 

		//std::cout << "ROI limits: " << (src_gray.cols - (WIN_SIZE / 2)) << " and " << (src_gray.rows - (WIN_SIZE / 2)) << std::endl; std::cin.get();

		for (int y = 2; y < (src_gray.cols-(WIN_SIZE/2)); y++)
		{
			for (int x = 2; x < (src_gray.rows - (WIN_SIZE / 2)); x++) {

				Rect ROI = Rect(y - 2, x - 2, WIN_SIZE, WIN_SIZE); //std::cout << x << "," << y << std::endl;
				double sum_G_Ix2 = sum(G_Ix2(ROI))[0], sum_G_Iy2 = sum(G_Iy2(ROI))[0], sum_G_IxIy = sum(G_IxIy(ROI))[0];

				//cv::imshow("temp", G_Ix2(ROI)); cv::waitKey(0);
				// Problem with ROI!!!
				
				double deretminant = sum_G_Ix2*sum_G_Iy2 - sum_G_IxIy*sum_G_IxIy;
				double trace = sum_G_Ix2 + sum_G_Iy2;

				/*/Testing
				if (sum_G_Ix2 != 0 || sum_G_Iy2 != 0 || sum_G_IxIy != 0) {
					//continue;
					//std::cout << sum_G_Ix2 << "," << sum_G_Iy2 << "," << sum_G_IxIy << std::endl;
					//cv::imshow("temp", G_Ix2(ROI)); cv::waitKey(0); cv::destroyAllWindows();//std::cin.get();
					//std::cout << "  |det,trace = " << deretminant << "," << trace <<"	|| criteria value: "<< deretminant - alpha * trace * trace <<" compared with "<<taur<< std::endl; //std::cin.get();
				}
				else {
					//continue;
					//std::cout << (src_gray.cols - (WIN_SIZE / 2)) -y << "," << (src_gray.rows - (WIN_SIZE / 2)) -x << std::endl;
				}
				*///Testing	

				if ((deretminant - alpha * trace * trace) < taur)
				{
					Corners.at<uchar>(x, y) = 0;
				}
				else
				{
					Corners.at<uchar>(x, y) = 255;
				}

				//std::cout << (int)Corners.at<uchar>(x, y) << std::endl << std::endl << std::endl;
			}
		}

		//cv::imshow("corners", Corners); cv::waitKey(0);

		Mat supressedCorners, mask;
		nonMaximaSuppression(Corners, 5, supressedCorners, mask);
		return supressedCorners;
	}


	std::vector<cv::Point> FeatureTracker::HarrisCornerPoints(cv::Mat src, double alpha, double taur)
	{
		Mat src_gray_temp, src_gray; cvtColor(src, src_gray_temp, CV_BGR2GRAY); src_gray_temp.convertTo(src_gray, CV_64FC1, 1 / 255.0); //std::cout << (double)src_gray.at<double>(0, 0) << "   |Here***\n"; std::cin.get();
		Mat Ix; Sobel(src_gray, Ix, -1, 1, 0);
		Mat Iy; Sobel(src_gray, Iy, -1, 0, 1);
		Mat IxIy = Ix.mul(Iy), Ix2 = Ix.mul(Ix), Iy2 = Iy.mul(Iy);
		Mat G; GaussianBlur(src_gray, G, Size(3, 3), 5, 5);
		Mat G_Ix2 = Ix2.mul(G), G_Iy2 = Iy2.mul(G), G_IxIy = IxIy.mul(G);
		//cv::imshow("temp", G_Iy2); cv::waitKey(0); //std::cin.get();
		int WIN_SIZE = 5;
		Mat Corners = Mat::zeros(src.size(), CV_8UC1);

		std::vector<cv::Point> output;
		//std::cout << "ROI limits: " << (src_gray.cols - (WIN_SIZE / 2)) << " and " << (src_gray.rows - (WIN_SIZE / 2)) << std::endl; std::cin.get();

		for (int y = 2; y < (src_gray.cols - (WIN_SIZE / 2)); y++)
		{
			for (int x = 2; x < (src_gray.rows - (WIN_SIZE / 2)); x++) {

				Rect ROI = Rect(y - 2, x - 2, WIN_SIZE, WIN_SIZE); //std::cout << x << "," << y << std::endl;
				double sum_G_Ix2 = sum(G_Ix2(ROI))[0], sum_G_Iy2 = sum(G_Iy2(ROI))[0], sum_G_IxIy = sum(G_IxIy(ROI))[0];

				//cv::imshow("temp", G_Ix2(ROI)); cv::waitKey(0);
				// Problem with ROI!!!

				double deretminant = sum_G_Ix2*sum_G_Iy2 - sum_G_IxIy*sum_G_IxIy;
				double trace = sum_G_Ix2 + sum_G_Iy2;

				/*/Testing
				if (sum_G_Ix2 != 0 || sum_G_Iy2 != 0 || sum_G_IxIy != 0) {
				//continue;
				//std::cout << sum_G_Ix2 << "," << sum_G_Iy2 << "," << sum_G_IxIy << std::endl;
				//cv::imshow("temp", G_Ix2(ROI)); cv::waitKey(0); cv::destroyAllWindows();//std::cin.get();
				//std::cout << "  |det,trace = " << deretminant << "," << trace <<"	|| criteria value: "<< deretminant - alpha * trace * trace <<" compared with "<<taur<< std::endl; //std::cin.get();
				}
				else {
				//continue;
				//std::cout << (src_gray.cols - (WIN_SIZE / 2)) -y << "," << (src_gray.rows - (WIN_SIZE / 2)) -x << std::endl;
				}
				*///Testing	

				if ((deretminant - alpha * trace * trace) < taur)
				{
					Corners.at<uchar>(x, y) = 0;
				}
				else
				{
					Corners.at<uchar>(x, y) = 255;
				}

				//std::cout << (int)Corners.at<uchar>(x, y) << std::endl << std::endl << std::endl;
			}
		}

		//cv::imshow("corners", Corners); cv::waitKey(0);

		Mat supressedCorners, mask;
		nonMaximaSuppression(Corners, 5, supressedCorners, mask);
		
		for (int y = 2; y < (src_gray.cols - (WIN_SIZE / 2)); y++)
		{
			for (int x = 2; x < (src_gray.rows - (WIN_SIZE / 2)); x++) {
				if ( (int)supressedCorners.at<uchar>(x, y) > 0) {
					output.push_back(cv::Point(y, x));
				}
			}
		}

		return output;
	}

	void FeatureTracker::nonMaximaSuppression(const Mat& src, const int sz, Mat& dst, const Mat mask) {

		// initialise the block mask and destination
		const int M = src.rows;
		const int N = src.cols;
		const bool masked = !mask.empty();
		Mat block = 255 * Mat_<uint8_t>::ones(Size(2 * sz + 1, 2 * sz + 1));
		dst = Mat_<uint8_t>::zeros(src.size());

		// iterate over image blocks
		for (int m = 0; m < M; m += sz + 1) {
			for (int n = 0; n < N; n += sz + 1) {
				Point  ijmax;
				double vcmax, vnmax;

				// get the maximal candidate within the block
				Range ic(m, min(m + sz + 1, M));
				Range jc(n, min(n + sz + 1, N));
				minMaxLoc(src(ic, jc), NULL, &vcmax, NULL, &ijmax, masked ? mask(ic, jc) : noArray());
				Point cc = ijmax + Point(jc.start, ic.start);

				// search the neighbours centered around the candidate for the true maxima
				Range in(max(cc.y - sz, 0), min(cc.y + sz + 1, M));
				Range jn(max(cc.x - sz, 0), min(cc.x + sz + 1, N));

				// mask out the block whose maxima we already know
				Mat_<uint8_t> blockmask;
				block(Range(0, in.size()), Range(0, jn.size())).copyTo(blockmask);
				Range iis(ic.start - in.start, min(ic.start - in.start + sz + 1, in.size()));
				Range jis(jc.start - jn.start, min(jc.start - jn.start + sz + 1, jn.size()));
				blockmask(iis, jis) = Mat_<uint8_t>::zeros(Size(jis.size(), iis.size()));

				minMaxLoc(src(in, jn), NULL, &vnmax, NULL, &ijmax, masked ? mask(in, jn).mul(blockmask) : blockmask);
				Point cn = ijmax + Point(jn.start, in.start);

				// if the block centre is also the neighbour centre, then it's a local maxima
				if (vcmax > vnmax) {
					dst.at<uint8_t>(cc.y, cc.x) = 255;
				}
			}
		}
	}

	std::pair<std::vector<cv::Point>,std::vector<bool>> FeatureTracker::KLTtracker(std::vector<cv::Point> _inputFeaturePoint, cv::Mat _frame_t0, cv::Mat _frame_t1)
	{
		//Convert to grayscale
		cvtColor(_frame_t0, _frame_t0, CV_BGR2GRAY);
		cvtColor(_frame_t1, _frame_t1, CV_BGR2GRAY);

		//double convertion
		_frame_t0.convertTo(_frame_t0, CV_64FC1, 1 / 255.0);
		_frame_t1.convertTo(_frame_t1, CV_64FC1, 1 / 255.0);

		//Calculate Derivatives for the second moment matrix
		Mat Ix; Sobel(_frame_t0, Ix, -1, 1, 0);
		Mat Iy; Sobel(_frame_t0, Iy, -1, 0, 1);
		Mat IxIy = Ix.mul(Iy), Ix2 = Ix.mul(Ix), Iy2 = Iy.mul(Iy); 

		int WindowSize = 15;
		std::vector<cv::Point> outputArray;
		std::vector<bool> index;

		int r = 0;
		for (cv::Point p : _inputFeaturePoint)
		{
			if ((p.x - WindowSize / 2) < 0 || (p.x + WindowSize / 2) > _frame_t0.cols || (p.y - WindowSize / 2) < 0 || (p.y + WindowSize / 2) > _frame_t0.rows)
			{
				outputArray.push_back(cv::Point(0, 0));
				index.push_back(false);
				continue;
			}
				
			//std::cout << p.x << "," << p.y << "| Remaining: "<< _inputFeaturePoint.size() - r << "\n";
			Rect ROI = Rect((p.x - WindowSize / 2), (p.y - WindowSize / 2), WindowSize, WindowSize); //Mat temp = _frame_t0; rectangle(temp, ROI, Scalar(0, 0, 255), 10); imshow("temp", temp); waitKey(0);
			double sum_Ix2 = sum(Ix2(ROI))[0], sum_Iy2 = sum(Iy2(ROI))[0], sum_IxIy = sum(IxIy(ROI))[0]; 
			double _deretminant = sum_Ix2*sum_Iy2 - sum_IxIy*sum_IxIy;
			double  u = 0, v = 0, _x = p.x, _y = p.y;

			bool success = false; 

			// Iterate untill convergence
			while (true) {
				//Next window position 
				_x = _x + u; _y = _y + v;

				//validity check
				if (((int)(_x) - WindowSize / 2) < 0 || ((int)(_x) + WindowSize / 2) > _frame_t0.cols || ((int)(_y) - WindowSize / 2) < 0 || ((int)(_y) + WindowSize / 2) > _frame_t0.rows)
					break;
				if ((ceil(_x)-WindowSize / 2) < 0 || (ceil(_x)+WindowSize / 2) > _frame_t0.cols || (ceil(_y) - WindowSize / 2) < 0 || (ceil(_y) + WindowSize / 2) > _frame_t0.rows)
					break;
				
				//Calculate It
				Mat It = bilinearWindow(_frame_t1, _x - WindowSize / 2, _y - WindowSize / 2, WindowSize) - _frame_t0(ROI);
				
				//Calculate u,v
				double sum_IxIt = sum( It.mul( Ix(ROI)))[0];
				double sum_IyIt = sum( It.mul( Iy(ROI)))[0];

				double _u = u, _v = v;

				u = (-sum_Iy2 * sum_IxIt + sum_IxIy * sum_IyIt) / _deretminant;
				v = (-sum_Ix2 * sum_IyIt + sum_IxIy * sum_IxIt) / _deretminant;
							//std::cout << "u,v-> " << u << "," << v << /*"  | float Value:  " << (-sum_Iy2 * sum_IxIt + sum_IxIy * sum_IyIt) / _deretminant <<*/ std::endl;
				//Compare values
				if ( (abs(u - _u)/_u)<KLT_ACCURACY_THRESHOLD && (abs(v - _v) / _v)<KLT_ACCURACY_THRESHOLD)
				{
					success = true;
					break;
				}


			}

			if (success)
			{
				outputArray.push_back( cv::Point(p.x + u, p.y + v));
				index.push_back(true);
			}
			else
			{
				outputArray.push_back(cv::Point(0, 0));
				index.push_back(false);
			}
			r++;
		}
		//std::cin.get();
		return std::pair<std::vector<cv::Point>, std::vector<bool>>(outputArray,index);
	}

	cv::Mat FeatureTracker::bilinearWindow(cv::Mat reference, double __x, double __y, int windowSize)
	{
		cv::Mat bilinearWindow = cv::Mat(Size(windowSize, windowSize), CV_64FC1);
		double x, y;
		for (int X = 0; X < windowSize; X++) 
		{
			for (int Y = 0; Y < windowSize; Y++)
			{
				x = __x + X; 
				y = __y + Y;

				int x_low = (int)x, y_low = (int)y, x_high = ceil(x), y_high = ceil(y);
				double _u = x - x_low, _y = y - y_low;

				double p1 = reference.at<double>(y_low, x_low);
				double p2 = reference.at<double>(y_high, x_low);
				double p3 = reference.at<double>(y_low, x_high);
				double p4 = reference.at<double>(y_high, x_high);

				double I1 = (p2 * _u + p1 * (1 - _u));
				double I2 = (p4 * _u + p3 * (1 - _u));

				double interpolatedValue = (I2 * _y + I1 * (1 - _y));
				bilinearWindow.at<double>(Y, X) = interpolatedValue;
			}
		}

		return bilinearWindow;
	}
}