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
	namedWindow("�Ҷ�", WINDOW_FREERATIO);
	imshow("HSV", hsv);
	imshow("�Ҷ�", gray);
	imwrite("D:/test/hsv.png", hsv);
	imwrite("D:/test/gray.png", gray);
}

void QuickDemo::mat_creation_demo(Mat &image)
{
	Mat m1, m2;
	m1 = image.clone();
	image.copyTo(m2);

	//�����հ�ͼ��
	Mat m3 = Mat::ones(Size(80, 80), CV_8UC3);//����80*80��CV8λ���޷��ŵ�nͨ����unsigned char

	
	std::cout << "width:" << m3.cols << "height" << m3.rows << "channels" << m3.channels() << std::endl;

	m3 = Scalar(255, 0, 0);//��������ͨ���ֱ�ֵ255,0,0  ��ͨ����ֵ���� m3 = 127;
	cout << m3 << endl;
	//Mat����+ - * /
	
	m3 += Scalar(0, 255, 255);
	cout << m3 << endl;  //ÿ�����ر�� 255,255,255

	Mat m4 = m3.clone();//��ֵM4����M3 M4�ı���,M3Ҳ�ı��ˣ�û�в����µ�����(M4��M3ͬ��)
	//M4ΪM3�Ŀ�¡��M3����ԭ������ɫ������ı䡣(M4��M3��ͬ�壬�����Ǹ��Ե���ɫ)
	//m3.copyTo(m4);//��M3��ֵ��M4��M4������ɫ
	m4 = Scalar(0, 255, 255);//�ı�m4����ɫΪ��ɫ ,m4Ҳ�ı�
	std::cout << m4 << std::endl;
	imshow("ͼ��3", m3);//�����ͼ������   ��ʾͼ��m3 ����ɫ
	imshow("ͼ��4", m4);//�����ͼ������
}