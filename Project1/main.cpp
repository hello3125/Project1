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
	namedWindow("���봰��",WINDOW_FREERATIO);
	imshow("���봰��", img);//��ʾͼƬ

	QuickDemo qd;
	/*qd.colorSpace_Demo(img);*/

	qd.mat_creation_demo(img);

	waitKey(0);//�ȴ�����
	destroyAllWindows();

	
	return 0;
}