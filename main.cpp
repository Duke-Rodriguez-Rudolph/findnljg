#include<iostream>
#include <opencv2/opencv.hpp> 
#include<math.h>

using namespace std;
using namespace cv;


/**
* @brief RGB转HSV，并转化为二值图
*
* @param img 待转换的原图
* @return Mat mask 转换好的二值图
*/
Mat getTwoValue(Mat img) {
	//imgHsv存储HSV图，mask储存二值图
	Mat imgHsv, mask;
	cvtColor(img, imgHsv, COLOR_BGR2HSV);
	inRange(imgHsv, Scalar(100, 43, 46), Scalar(124, 255, 255), mask);
	return mask;
}


/**
* @brief 通过腐蚀与膨胀清除白点
*
* @param img 待处理的原图
* @param erode_interation 腐蚀迭代次数
* @param dial_interation 膨胀迭代次数
* @return Mat imgDial 处理好后的图像
*/
Mat clearWhitePoint(Mat img, int erode_interation, int dial_interation) {
	//imgDial储存膨胀后的图，imgErode储存腐蚀后的图，element储存核
	Mat imgDial, imgErode, element = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(img, imgErode, element, Point(-1, -1), erode_interation);
	dilate(imgErode, imgDial, element, Point(-1, -1), dial_interation);
	
	return imgDial;

}


/**
* @brief 通过膨胀与腐蚀清除黑点
*
* @param img 待处理的原图
* @param dial_interation 膨胀迭代次数
* @param erode_interation 腐蚀迭代次数
* @return Mat imgErode 处理好后的图像
*/
Mat clearBlackPoint(Mat img, int dial_interation, int erode_interation) {
	//imgDial储存膨胀后的图，imgErode储存腐蚀后的图，element储存核
	Mat imgDial, imgErode, element = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(img, imgDial, element, Point(-1, -1), dial_interation);
	erode(imgDial, imgErode, element, Point(-1, -1), erode_interation);
	return imgErode;

}


/**
* @brief 将轮廓从Mat类型转为可画出的Point型
*
* @param box 待转换的轮廓
* @return vector<Point> result 可画出来的轮廓
*/
vector<Point> turnToContours(Mat box) {
	//可画出来的轮廓的内部
	vector<Point> result;
	for (int i = 0; i < box.rows; i++) {
		float* ptr = box.ptr<float>(i);
		result.push_back(Point(int(ptr[0]), int(ptr[1])));
	}
	return result;
}


/**
* @brief 对一帧的完整操作
*
* @param img 输入一帧的画面
*/
void onceTime(Mat img) {
	//存储二值图
	Mat mask;
	Mat result;
	vector<Point2f> mids;
	//存储查找轮廓的内容
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	//图像预处理
	mask = getTwoValue(img);
	mask = clearWhitePoint(mask, 1, 2);

	floodFill(mask, Point(0, 0), Scalar(255));
	bitwise_not(mask,mask);
	
	result = clearBlackPoint(mask, 9, 1);

	findContours(result, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point());
	for (int i = 0; i < contours.size(); i++) {
		vector<Point> cnt = contours[i];
		float area = contourArea(cnt);
		if (area > 1000 && area < 2500) {
			RotatedRect rect = minAreaRect(cnt);
			Mat box;
			boxPoints(rect, box);
			//粗轮廓的中点
			Point2f mid = Point2f((box.at<float>(0, 0) + box.at<float>(1, 0) + box.at<float>(2, 0) + box.at<float>(3, 0)) / 4,
				(box.at<float>(0, 1) + box.at<float>(1, 1) + box.at<float>(2, 1) + box.at<float>(3, 1)) / 4);
			mids.push_back(mid);
		}
	}


	findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point());
	for (int i = 0; i < contours.size(); i++) {
		vector<Point> cnt = contours[i];
		for (int j = 0; j < mids.size(); j++) {
			int flag = pointPolygonTest(cnt, mids[j], false);
			if (flag == 1) {
				RotatedRect rect = minAreaRect(cnt);
				Mat box;
				boxPoints(rect, box);
				vector<vector<Point>> contour;
				contour.push_back(turnToContours(box));
				drawContours(img,contour,-1,Scalar(0,0,255),3);

				//真实轮廓的中点
				Point mid = Point(Point2f((box.at<float>(0, 0) + box.at<float>(1, 0) + box.at<float>(2, 0) + box.at<float>(3, 0)) / 4,
					(box.at<float>(0, 1) + box.at<float>(1, 1) + box.at<float>(2, 1) + box.at<float>(3, 1)) / 4));
				circle(img, mid, 1, Scalar(0, 0, 255), 5);
			}
		}
	}
	imshow("img", img);
}

int main() {
	//实例化相机
	VideoCapture capture;
	//帧
	Mat frame;
	//读取视频
	frame = capture.open("1.avi");

	//对每一帧进行操作
	while (capture.isOpened()) {
		capture.read(frame);
		onceTime(frame);
		waitKey(25);
	}

	destroyAllWindows();
	return 0;
}