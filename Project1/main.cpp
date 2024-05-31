#include <iostream>
#include <opencv2/opencv.hpp>
#include <quickopencv.h>
using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
	Mat img = imread("D:/test/school.jpg");
	if (img.empty())
	{
		cout << "could not load image" << endl;
		return -1;
	}
	namedWindow("输入窗口",WINDOW_FREERATIO);
	imshow("输入窗口", img);//显示图片

	QuickDemo qd;
	/*qd.colorSpace_Demo(img);*/

	qd.mat_creation_demo(img);

	waitKey(0);//等待按键
	destroyAllWindows();

	
	return 0;
}