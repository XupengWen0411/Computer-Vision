#pragma once
#include "pch.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


class ImageFeature {
public:
	void lbp_demo(Mat& image);
	void elbp_demo(Mat& image);

};

