#include <quickopencv.h>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;

void QuickDemo::colorSpace_Demo(Mat &image)
{  
	Mat gray, hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	// H 0 ~ 180, S, V 0-100
	cvtColor(image, gray, COLOR_BGR2GRAY);
	namedWindow("HSV", WINDOW_FREERATIO);
	namedWindow("灰度", WINDOW_FREERATIO);
	imshow("HSV", hsv);
	imshow("灰度", gray);
	imwrite("D:/test/hsv.png", hsv);
	imwrite("D:/test/gray.png", gray);
}

void QuickDemo::mat_creation_demo(Mat &image)
{
	Mat m1, m2;
	m1 = image.clone();
	image.copyTo(m2);

	//创建空白图像
	Mat m3 = Mat::ones(Size(80, 80), CV_8UC3);//创建80*80的CV8位的无符号的n通道的unsigned char

	
	std::cout << "width:" << m3.cols << "height" << m3.rows << "channels" << m3.channels() << std::endl;

	m3 = Scalar(255, 0, 0);//给像素三通道分别赋值255,0,0  单通道赋值方法 m3 = 127;
	cout << m3 << endl;
	//Mat重载+ - * /
	
	m3 += Scalar(0, 255, 255);
	cout << m3 << endl;  //每个像素变成 255,255,255

	Mat m4 = m3.clone();//赋值M4就是M3 M4改变了,M3也改变了，没有产生新的自我(M4与M3同体)
	//M4为M3的克隆，M3还是原来的颜色，不会改变。(M4与M3不同体，各自是各自的颜色)
	//m3.copyTo(m4);//把M3赋值给M4，M4就是蓝色
	m4 = Scalar(0, 255, 255);//改变m4的颜色为黄色 ,m4也改变
	std::cout << m4 << std::endl;
	imshow("图像3", m3);//标题和图像名称   显示图像m3 纯蓝色
	imshow("图像4", m4);//标题和图像名称
}