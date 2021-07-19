#include<iostream>
#include <opencv2/opencv.hpp> 
#include<math.h>

using namespace std;
using namespace cv;


/**
* @brief RGBתHSV����ת��Ϊ��ֵͼ
*
* @param img ��ת����ԭͼ
* @return Mat mask ת���õĶ�ֵͼ
*/
Mat getTwoValue(Mat img) {
	//imgHsv�洢HSVͼ��mask�����ֵͼ
	Mat imgHsv, mask;
	cvtColor(img, imgHsv, COLOR_BGR2HSV);
	inRange(imgHsv, Scalar(100, 43, 46), Scalar(124, 255, 255), mask);
	return mask;
}


/**
* @brief ͨ����ʴ����������׵�
*
* @param img �������ԭͼ
* @param erode_interation ��ʴ��������
* @param dial_interation ���͵�������
* @return Mat imgDial ����ú��ͼ��
*/
Mat clearWhitePoint(Mat img, int erode_interation, int dial_interation) {
	//imgDial�������ͺ��ͼ��imgErode���港ʴ���ͼ��element�����
	Mat imgDial, imgErode, element = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(img, imgErode, element, Point(-1, -1), erode_interation);
	dilate(imgErode, imgDial, element, Point(-1, -1), dial_interation);
	
	return imgDial;

}


/**
* @brief ͨ�������븯ʴ����ڵ�
*
* @param img �������ԭͼ
* @param dial_interation ���͵�������
* @param erode_interation ��ʴ��������
* @return Mat imgErode ����ú��ͼ��
*/
Mat clearBlackPoint(Mat img, int dial_interation, int erode_interation) {
	//imgDial�������ͺ��ͼ��imgErode���港ʴ���ͼ��element�����
	Mat imgDial, imgErode, element = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(img, imgDial, element, Point(-1, -1), dial_interation);
	erode(imgDial, imgErode, element, Point(-1, -1), erode_interation);
	return imgErode;

}


/**
* @brief ��������Mat����תΪ�ɻ�����Point��
*
* @param box ��ת��������
* @return vector<Point> result �ɻ�����������
*/
vector<Point> turnToContours(Mat box) {
	//�ɻ��������������ڲ�
	vector<Point> result;
	for (int i = 0; i < box.rows; i++) {
		float* ptr = box.ptr<float>(i);
		result.push_back(Point(int(ptr[0]), int(ptr[1])));
	}
	return result;
}


/**
* @brief ��һ֡����������
*
* @param img ����һ֡�Ļ���
*/
void onceTime(Mat img) {
	//�洢��ֵͼ
	Mat mask;
	Mat result;
	vector<Point2f> mids;
	//�洢��������������
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	//ͼ��Ԥ����
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
			//���������е�
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

				//��ʵ�������е�
				Point mid = Point(Point2f((box.at<float>(0, 0) + box.at<float>(1, 0) + box.at<float>(2, 0) + box.at<float>(3, 0)) / 4,
					(box.at<float>(0, 1) + box.at<float>(1, 1) + box.at<float>(2, 1) + box.at<float>(3, 1)) / 4));
				circle(img, mid, 1, Scalar(0, 0, 255), 5);
			}
		}
	}
	imshow("img", img);
}

int main() {
	//ʵ�������
	VideoCapture capture;
	//֡
	Mat frame;
	//��ȡ��Ƶ
	frame = capture.open("1.avi");

	//��ÿһ֡���в���
	while (capture.isOpened()) {
		capture.read(frame);
		onceTime(frame);
		waitKey(25);
	}

	destroyAllWindows();
	return 0;
}