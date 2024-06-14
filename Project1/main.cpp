#include <iostream>
#include <opencv2/opencv.hpp>
#include <quickopencv.h>
using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
	//Mat img = imread("D:/test/images/OIP-C.jpg");
	Mat img = imread("C:/Users/Lenovo/Pictures/IMG_00000008.jpg");
	//Mat img = imread("C:/Users/Lenovo/Pictures/ly.png");
	if (img.empty())
	{
		cout << "could not load image" << endl;
		return -1;
	}
	namedWindow("输入窗口",WINDOW_FREERATIO);
	imshow("输入窗口", img);//显示图片

	QuickDemo qd;
	/*qd.colorSpace_Demo(img);*/

	/*qd.mat_creation_demo(img);*/

	/*qd.pixel_visit_demo(img);*/

	//qd.operators_demo(img);

	//qd.key_demo(img);

	//qd.color_style_demo(img);
	
	//qd.bitwise_demo(img);

	//qd.channels_demo(img);

	//qd.inrange_demo(img);

	//qd.pixel_statistic_demo(img);

	//qd.bifilter_demo(img);
	qd.face_detection_demo();

	waitKey(0);//等待按键
	destroyAllWindows();

	
	return 0;
}