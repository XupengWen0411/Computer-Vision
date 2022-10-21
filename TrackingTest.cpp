// ShipTrackingTest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include "TrackingTest.h"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>


void canny_feature();
void LBP_Feature();
void Hog_frature();
void SIFT_feature();

int main()
{

	//canny_feature();
	//LBP_Feature();
	//Hog_frature();
	SIFT_feature();

	return 0;
}

void canny_feature()
{
	//canny边缘检测
	Mat srcImage, grayImage;
	srcImage = imread("C:\\Users\\JACK\\Source\\repos\\ShipTrackingTest\\ShipTrackingTest\\68.jpg");
	Mat srcImage1 = srcImage.clone();
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	Mat dstImage, edge;

	blur(grayImage, grayImage, Size(3, 3));
	Canny(grayImage, edge, 150, 100, 3);

	dstImage.create(srcImage1.size(), srcImage1.type());
	dstImage = Scalar::all(0);
	srcImage1.copyTo(dstImage, edge);
	imwrite("canny.jpg", dstImage);
}


void LBP_Feature()
{
	//LBP特征提取检测
	const char* img_path = "C:\\Users\\JACK\\Source\\repos\\ShipTrackingTest\\ShipTrackingTest\\68.jpg";
	Mat image = imread(img_path, IMREAD_GRAYSCALE);   //灰度图读入
	if (image.empty()) {
		cout << "图像数据为空，读取文件失败！" << endl;
	}
	ImageFeature imgfeature;
	imgfeature.elbp_demo(image);
	imgfeature.lbp_demo(image);

	imshow("image", image);
	imwrite("grey.jpg", image);

	waitKey(0);
	destroyAllWindows();
	
}

using namespace std;
using namespace cv;

// HOGDescriptor visual_imagealizer
// adapted for arbitrary size of feature sets and training images
Mat get_hogdescriptor_visual_image(Mat& origImg,
	vector<float>& descriptorValues,//hog特征向量
	Size winSize,//图片窗口大小
	Size cellSize,
	int scaleFactor,//缩放背景图像的比例
	double viz_factor)//缩放hog特征的线长比例
{
	Mat visual_image;//最后可视化的图像大小
	resize(origImg, visual_image, Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));

	int gradientBinSize = 9;
	// dividing 180° into 9 bins, how large (in rad) is one bin?
	float radRangeForOneBin = 3.14 / (float)gradientBinSize; //pi=3.14对应180°

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;//x方向上的cell个数
	int cells_in_y_dir = winSize.height / cellSize.height;//y方向上的cell个数
	int totalnrofcells = cells_in_x_dir * cells_in_y_dir;//cell的总个数
	//注意此处三维数组的定义格式
	//int ***b;
	//int a[2][3][4];
	//int (*b)[3][4] = a;
	//gradientStrengths[cells_in_y_dir][cells_in_x_dir][9]
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y < cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x < cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin < gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;//把每个cell的9个bin对应的梯度强度都初始化为0
		}
	}
	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	//相当于blockstride = (8,8)
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx < blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky < blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr < 4; cellNr++)
			{
				// compute corresponding cell nr
				int cellx = blockx;
				int celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}
				for (int bin = 0; bin < gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;//因为C是按行存储

				} // for (all bins)


				// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;//由于block之间有重叠，所以要记录哪些cell被多次计算了

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)
	// compute average gradient strengths
	for (int celly = 0; celly < cells_in_y_dir; celly++)
	{
		for (int cellx = 0; cellx < cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin < gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}
	cout << "winSize = " << winSize << endl;
	cout << "cellSize = " << cellSize << endl;
	cout << "blockSize = " << cellSize * 2 << endl;
	cout << "blockNum = " << blocks_in_x_dir << "×" << blocks_in_y_dir << endl;
	cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

	// draw cells
	for (int celly = 0; celly < cells_in_y_dir; celly++)
	{
		for (int cellx = 0; cellx < cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize.width;
			int drawY = celly * cellSize.height;

			int mx = drawX + cellSize.width / 2;
			int my = drawY + cellSize.height / 2;

			rectangle(visual_image,
				Point(drawX*scaleFactor, drawY*scaleFactor),
				Point((drawX + cellSize.width)*scaleFactor,
				(drawY + cellSize.height)*scaleFactor),
				CV_RGB(0, 0, 0),//cell框线的颜色
				1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin < gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;//取每个bin里的中间值，如10°,30°,...,170°.

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = cellSize.width / 2;
				float scale = viz_factor; // just a visual_imagealization scale,
				// to see the lines better

				// compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visual_imagealization
				line(visual_image,
					Point(x1*scaleFactor, y1*scaleFactor),
					Point(x2*scaleFactor, y2*scaleFactor),
					CV_RGB(255, 255, 255),//HOG可视化的cell的颜色
					1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	// don't forget to free memory allocated by helper data structures!
	for (int y = 0; y < cells_in_y_dir; y++)
	{
		for (int x = 0; x < cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visual_image;//返回最终的HOG可视化图像

}

void Hog_frature()
{
	HOGDescriptor hog;//使用的是默认的hog参数
	/*
	HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16), Size block_stride=Size(8, 8),
	Size cell_size=Size(8, 8), int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA(DEFAULT_WIN_SIGMA=-1),
	double threshold_L2hys=0.2, bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS)
	Parameters:
	win_size – Detection window size. Align to block size and block stride.
	block_size – Block size in pixels. Align to cell size. Only (16,16) is supported for now.
	block_stride – Block stride. It must be a multiple of cell size.
	cell_size – Cell size. Only (8, 8) is supported for now.
	nbins – Number of bins. Only 9 bins per cell are supported for now.
	win_sigma – Gaussian smoothing window parameter.
	threshold_L2hys – L2-Hys normalization method shrinkage.
	gamma_correction – Flag to specify whether the gamma correction preprocessing is required or not.
	nlevels – Maximum number of detection window increases.
	*/
	//对于128*80的图片，blockstride = 8,15*9的block，2*2*9*15*9 = 4860

	Mat src = imread("C:\\Users\\JACK\\Source\\repos\\ShipTrackingTest\\ShipTrackingTest\\68.jpg");//注意这里边的双斜杠！！！！！！！！！！
	int src_width = src.cols;
	int src_height = src.rows;
	int width = src_width;
	int height = src_height;
	hog.winSize = Size(width, height);
	vector<float> des;//HOG特征向量

	Mat dst;
	resize(src, dst, Size(width, height));//规范图像尺寸
	imshow("src", src);
	hog.compute(dst, des);//计算hog特征
	Mat background = Mat::zeros(Size(width, height), CV_8UC1);//设置黑色背景图，因为要用白色绘制hog特征

	Mat d = get_hogdescriptor_visual_image(background, des, hog.winSize, hog.cellSize, 3, 2.5);
	imshow("dst", d);
	imwrite("hogvisualize.jpg", d);
	waitKey();
}

void SIFT_feature()
{

	int64 t1, t2;
	double tkpt, tdes, tmatch_bf, tmatch_knn;

	// 1. 读取图片
	const cv::Mat image1 = cv::imread("C:\\Users\\JACK\\Source\\repos\\ShipTrackingTest\\ShipTrackingTest\\68.jpg", 0); //Load as grayscale
	const cv::Mat image2 = cv::imread("C:\\Users\\JACK\\Source\\repos\\ShipTrackingTest\\ShipTrackingTest\\68.jpg", 0); //Load as grayscale
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;

	int desNum = 100;
	Ptr<SiftFeatureDetector> sift = SiftFeatureDetector::create("68.jpg");
	// 2. 计算特征点
	t1 = cv::getTickCount();
	sift->detect(image1, keypoints1);
	t2 = cv::getTickCount();
	tkpt = 1000.0*(t2 - t1) / cv::getTickFrequency();
	sift->detect(image2, keypoints2);


	// 3. 计算特征描述符
	cv::Mat descriptors1, descriptors2;
	t1 = cv::getTickCount();
	sift->compute(image1, keypoints1, descriptors1);
	t2 = cv::getTickCount();
	tdes = 1000.0*(t2 - t1) / cv::getTickFrequency();
	sift->compute(image2, keypoints2, descriptors2);

	cv::Mat output;
	cv::drawKeypoints(image1, keypoints1, output);
	cv::imwrite("sift_image1_keypoints.jpg", output);
	cv::drawKeypoints(image2, keypoints2, output);
	cv::imwrite("sift_image2_keypoints.jpg", output);

	//// 4. 特征匹配
	//cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
	//// cv::BFMatcher matcher(cv::NORM_L2);

	//// (1) 直接暴力匹配
	//std::vector<cv::DMatch> matches;
	//t1 = cv::getTickCount();
	//matcher->match(descriptors1, descriptors2, matches);
	//t2 = cv::getTickCount();
	//tmatch_bf = 1000.0*(t2 - t1) / cv::getTickFrequency();
	//// 画匹配图
	//cv::Mat img_matches_bf;
	//drawMatches(image1, keypoints1, image2, keypoints2, matches, img_matches_bf);
	//imshow("bf_matches", img_matches_bf);

	//// (2) KNN-NNDR匹配法
	//std::vector<std::vector<cv::DMatch> > knn_matches;
	//const float ratio_thresh = 0.7f;
	//std::vector<cv::DMatch> good_matches;
	//t1 = cv::getTickCount();
	//matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	//for (auto & knn_matche : knn_matches) {
	//	if (knn_matche[0].distance < ratio_thresh * knn_matche[1].distance) {
	//		good_matches.push_back(knn_matche[0]);
	//	}
	//}
	//t2 = cv::getTickCount();
	//tmatch_knn = 1000.0*(t2 - t1) / cv::getTickFrequency();

	//// 画匹配图
	//cv::Mat img_matches_knn;
	//drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches_knn, cv::Scalar::all(-1),
	//	cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//cv::imshow("knn_matches", img_matches_knn);
	//cv::waitKey(0);


	//std::cout << "图1特征点检测耗时(ms)：" << tkpt << std::endl;
	//std::cout << "图1特征描述符耗时(ms)：" << tdes << std::endl;
	//std::cout << "BF特征匹配耗时(ms)：" << tmatch_bf << std::endl;
	//std::cout << "KNN-NNDR特征匹配耗时(ms)：" << tmatch_knn << std::endl;
	
}
