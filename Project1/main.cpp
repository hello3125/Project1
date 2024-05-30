#include <iostream>
#include <opencv2/opencv.hpp>
#include <quickopencv.h>
using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
	Mat img = imread("C:/Users/Lenovo/Pictures/4c194b7df1a241789d49338ec423804d.jpg");
	if (img.empty())
	{
		cout << "could not load image" << endl;
		return -1;
	}
	namedWindow("输入窗口",WINDOW_FREERATIO);
	imshow("输入窗口", img);//显示图片

	QuickDemo qd;
	qd.colorSpace_Demo(img);

	waitKey(0);//等待按键
	destroyAllWindows();//销毁前面出纳宫颈癌
	
	return 0;
}