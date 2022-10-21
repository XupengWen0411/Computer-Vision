#include "pch.h"
#include "TrackingTest.h"

void ImageFeature::lbp_demo(Mat& image) {
	int height = image.rows;
	int width = image.cols;

	double t0 = cv::getTickCount();
	uchar code;
	uchar* mat_ptr = image.data;
	Mat LBP_dst = Mat::zeros(height - 2, width - 2, CV_8UC1);
	for (size_t h = 1; h < height - 1; h++) {
		uchar* lbp_ptr = LBP_dst.ptr<uchar>(h - 1);
		for (size_t w = 1; w < width - 1; w++) {
			uchar center = *(mat_ptr + h * width + w);
			code = 0;
			code |= (*(mat_ptr + (h - 1) * width + w - 1) > center) << 7;
			code |= (*(mat_ptr + (h - 1) * width + w) > center) << 6;
			code |= (*(mat_ptr + (h - 1) * width + w + 1) > center) << 5;
			code |= (*(mat_ptr + (h)* width + w + 1) > center) << 4;
			code |= (*(mat_ptr + (h + 1) * width + w + 1) > center) << 3;
			code |= (*(mat_ptr + (h + 1) * width + w) > center) << 2;
			code |= (*(mat_ptr + (h + 1) * width + w - 1) > center) << 1;
			code |= (*(mat_ptr + (h)* width + w - 1) > center) << 0;
			lbp_ptr[w - 1] = code;
		}
	}
	double t1 = cv::getTickCount();
	cout << "Total cost is :" << ((t1 - t0) / cv::getTickFrequency()) << endl;
	imshow("LBP_dst", LBP_dst);
}

static void elbp_change(int radius, void*userdata) {
	if (radius < 1) {
		radius = 1;
	}
	cout << "radius = " << radius << endl;

	Mat src = *((Mat*)userdata);
	int height = src.rows;
	int width = src.cols;
	int offset = radius * 2;
	Mat elbpImg = Mat::zeros(height - offset, width - offset, CV_8UC1);
	int neighbors = 8;
	for (size_t n = 0; n < neighbors; n++) {
		float tmp = 2.0 * CV_PI * n / static_cast<float>(neighbors);
		float x = static_cast<float>(-radius) * sin(tmp);
		float y = static_cast<float>(radius) * cos(tmp);

		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));

		float tx = x - fx;
		float ty = y - fy;

		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) *ty;
		float w4 = tx * ty;

		for (int i = radius; i < height - radius; i++) {
			for (int j = radius; j < width - radius; j++) {
				float t = w1 * src.at<uchar>(i + fy, j + fx)
					+ w2 * src.at<uchar>(i + fy, j + cx)
					+ w3 * src.at<uchar>(i + cy, j + fx)
					+ w4 * src.at<uchar>(i + cy, j + cx);
				elbpImg.at<uchar>(i - radius, j - radius) += (
					(t > src.at<uchar>(i, j)) &&
					(std::abs(t - src.at<uchar>(i, j)) >
						std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
	imshow("elbp_result", elbpImg);
	imwrite("LBPimage.jpg", elbpImg);
}


void ImageFeature::elbp_demo(Mat& image) {
	namedWindow("elbp_result", WINDOW_AUTOSIZE);
	int current_radius = 3;
	int max_count = 20;
	createTrackbar("elbp_radius ", "elbp_result", &current_radius, max_count, elbp_change, &image);
	elbp_change(current_radius, &image);
}


