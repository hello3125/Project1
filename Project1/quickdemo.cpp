#include <quickopencv.h>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;

void QuickDemo::colorSpace_Demo(Mat &image)
{
	Mat gray, hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	// H 0 ~ 180, S, V 
	cvtColor(image, gray, COLOR_BGR2GRAY);
	namedWindow("HSV", WINDOW_FREERATIO);
	namedWindow("�Ҷ�", WINDOW_FREERATIO);
	imshow("HSV", hsv);
	imshow("�Ҷ�", gray);
	imwrite("D:/test/hsv.png", hsv);
	imwrite("D:/test/gray.png", gray);
}
