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

void QuickDemo::pixel_visit_demo(Mat &image)
{
	int dims = image.channels();
	int h = image.rows;
	int w = image.cols;
	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++)
		{
			if (dims == 1) //��ͨ���ĻҶ�ͼ��
			{
				int pv = image.at<uchar>(row, col);//�õ�����ֵ
				image.at<uchar>(row, col) = 255 - pv;//������ֵ���¸�ֵ

			}
			if (dims == 3) //��ͨ���Ĳ�ɫͼ��
			{
				Vec3b bgr = image.at<Vec3b>(row, col); //opencv�ض������ͣ���ȡ��ά��ɫ��3��ֵ
				image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
				image.at<Vec3b>(row, col)[1] = 255 - bgr[1];
				image.at<Vec3b>(row, col)[2] = 255 - bgr[2];//�Բ�ɫͼ���ȡ��������ֵ�����Ҷ�����ֵ���и�д��
			}
		}
	}
	namedWindow("���ض�д��ʾ", WINDOW_FREERATIO);
	imshow("���ض�д��ʾ", image);
}


void QuickDemo::operators_demo(Mat &image)
{
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	dst = image - Scalar(50, 50, 50);
	m = Scalar(50, 50, 50);
	multiply(image, m, dst);//�˷�����
	namedWindow("�˷�����", WINDOW_FREERATIO);
	imshow("�˷�����", dst);
	add(image, m, dst);//�ӷ�����
	namedWindow("�ӷ�����", WINDOW_FREERATIO);
	imshow("�ӷ�����", dst);
	subtract(image, m, dst);//��������
	namedWindow("��������", WINDOW_FREERATIO);
	imshow("��������", dst);
	divide(image, m, dst);//��������
	namedWindow("��������", WINDOW_FREERATIO);
	imshow("��������", dst);
	//�ӷ������ײ�
	int dims = image.channels();
	int h = image.rows;
	int w = image.cols;
	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++)
		{
			Vec3b p1 = image.at<Vec3b>(row, col); //opencv�ض������ͣ���ȡ��ά��ɫ��3��ֵ
			Vec3b p2 = m.at<Vec3b>(row, col);
			dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(p1[0] + p2[0]);//saturate_cast����������С��0����0������255����255
			dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(p1[1] + p2[1]);
			dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(p1[2] + p2[2]);//�Բ�ɫͼ���ȡ��������ֵ�����Ҷ�����ֵ���и�д��
		}
	}
	namedWindow("�ӷ�����2", WINDOW_FREERATIO);
	imshow("�ӷ�����2", dst);
}

static void on_lightness(int b, void* userData)
{
	Mat image = *((Mat*)userData);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	m = Scalar(b, b, b);//�����������ȵ���ֵ
	addWeighted(image, 1.0, m, 0, b, dst);//�ں�����ͼ
	imshow("������Աȶȵ���", dst);//��ʾ��������֮���ͼƬ
}

static void on_contrast(int b, void* userData)
{
	Mat image = *((Mat*)userData);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	double contrast = b / 100.0;
	addWeighted(image, contrast, m, 0.0, 0, dst);//�ں�����ͼ
	imshow("������Աȶȵ���", dst);//��ʾ��������֮���ͼƬ
}

void QuickDemo::tracking_bar_demo(Mat &image)
{
	namedWindow("������Աȶȵ���", WINDOW_FREERATIO);
	Mat dst = Mat::zeros(image.size(), image.type());//ͼƬ�ĳ�ʼ������һ����image��С��ȣ�������ͬ��ͼ��
	Mat m = Mat::zeros(image.size(), image.type());//ͼƬ�ĳ�ʼ������һ����image��С��ȣ�������ͬ��ͼ��
	Mat src = image;//��src��ֵ
	int lightness = 50;
	int max_value = 100;//�������ֵΪ100
	int contrast_value = 100;//�Աȶ�
	createTrackbar("Value Bar:", "������Աȶȵ���", &lightness, max_value, on_lightness, (void*)(&image));//���ú������ȵ��ڹ��ܡ�
	createTrackbar("Contrast Bar:", "������Աȶȵ���", &contrast_value, 200, on_contrast, (void*)(&image));//�Աȶ�

	on_lightness(50, &image);
}

void QuickDemo::key_demo(Mat &image)
{
	Mat dst = Mat::zeros(image.size(), image.type());
	while (true)
	{
		char c = waitKey(300);
		if (c == 27)
		{
			break;
		}

		if (c == 49)// key 1
		{
			std::cout << "you enter key 1" << std::endl;
			cvtColor(image, dst, COLOR_BGR2GRAY);
		}
		if (c == 50)// key 2
		{
			std::cout << "you enter key 2" << std::endl;
			cvtColor(image, dst, COLOR_BGR2HSV);
		}
		if (c == 51)// key 1
		{
			std::cout << "you enter key 3" << std::endl;
			dst = Scalar(50, 50, 50);
			add(image, dst, dst);
		}
		imshow("������Ӧ", dst);
	}
}

void QuickDemo::color_style_demo(Mat &image)
{
	int colormap[] = {
		COLORMAP_AUTUMN ,
		COLORMAP_BONE,
		COLORMAP_CIVIDIS,
		COLORMAP_DEEPGREEN,
		COLORMAP_HOT,
		COLORMAP_HSV,
		COLORMAP_INFERNO,
		COLORMAP_JET,
		COLORMAP_MAGMA,
		COLORMAP_OCEAN,
		COLORMAP_PINK,
		COLORMAP_PARULA,
		COLORMAP_RAINBOW,
		COLORMAP_SPRING,
		COLORMAP_TWILIGHT,
		COLORMAP_TURBO,
		COLORMAP_TWILIGHT,
		COLORMAP_VIRIDIS,
		COLORMAP_TWILIGHT_SHIFTED,
		COLORMAP_WINTER
	};

	Mat dst;
	int index = 0;
	while (true)
	{
		char c = waitKey(3000);
		if (c == 27)
		{
			break;
		}
		applyColorMap(image, dst, colormap[index % 19]);
		index++;
		imshow("ѭ������", dst);
	}
}

void QuickDemo::bitwise_demo(Mat &image)
{
	Mat m1 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);
	rectangle(m1, Rect(100, 100, 80, 80), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(m2, Rect(150, 150, 80, 80), Scalar(0, 255, 255), -1, LINE_8, 0);
	imshow("m1", m1);
	imshow("m2", m2);

	Mat dst;
	bitwise_and(m1, m2, dst);
	imshow("����λ��", dst);
	bitwise_or(m1, m2, dst);//λ������
	imshow("����λ��", dst);
	bitwise_not(image, dst);//ȡ������
	imshow("����λ��", dst);
	bitwise_xor(m1, m2, dst);//������
	imshow("����λ���", dst);
	
}

void QuickDemo::channels_demo(Mat &image)
{
	std::vector<Mat> mv;
	split(image, mv);
	imshow("��ɫ", mv[0]);
	imshow("��ɫ", mv[1]);
	imshow("��ɫ", mv[2]);

	//��ͨ���ϲ���ʹ�� merge
	Mat	dst = Mat::zeros(image.size(),image.type());
	mv[0] = 0;
	mv[1] = 0;
	merge(mv, dst);
	imshow("��ɫͼ��", dst);

	//mv[0] = 0;
	//mv[2] = 0;
	//merge(mv, dst);
	//imshow("��ɫͼ��", dst);

	int from_to[] = { 0,2,1,1,2,0 }; //BGRͼ��ת��ΪRGBͼ��
	//��ͨ���໥��������0->��2����һ->��һ���ڶ�->��0
	mixChannels(&image, 1, &dst, 1, from_to, 3);//3��ʾ3��ͨ��
	//����1ָ������ͼ��->����2���õ�dst
	imshow("ͨ�����", dst);
}

void QuickDemo::inrange_demo(Mat &image)
{
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);
	//35,43,46����ͼƬ����ɫ�����ȷ����Сֵ��
	//77,255,255 ��ȡ
	//����1�ͷ�Χ������2�߷�Χ
	//��hsv�е��ɵ͵��ߵ����ص���ȡ�������Ҵ洢��mask���С�
	imshow("mask", hsv);
	Mat redback = Mat::zeros(image.size(), image.type());
	redback = Scalar(40, 40, 200);
	bitwise_not(mask, mask);
	imshow("mask", mask);
	image.copyTo(redback, mask);//��redback���Ƶ�mask��maskͨ��inRange�õ���
	imshow("roi ������ȡ ", redback);
}

void QuickDemo::pixel_statistic_demo(Mat &image)
{
	double minv, maxv;//������ֵ
	Point minLoc, maxLoc;//������ֵ��ַ
	std::vector<Mat>mv;//mv��һ��Mat���͵����� װ�����������
	split(image, mv);
	for (int i = 0; i < mv.size(); i++)
	{
		//�ֱ��ӡ����ͨ������ֵ
		minMaxLoc(mv[i], &minv, &maxv, &minLoc, &maxLoc, Mat());//���ͼ������ֵ����Сֵ��
		std::cout << "No.channels:" << i << "min value:" << minv << "max value:" << maxv << std::endl;
	}
	Mat mean, stddev;//ƽ��ֵ������
	meanStdDev(image, mean, stddev);//���ͼ��ľ�ֵ�ͷ���
	std::cout << "mean:" << mean << std::endl;
	std::cout << "stddev:" << stddev << std::endl;
}

void QuickDemo::drawing_demo(Mat &image)
{
	Rect rect;
	rect.x = 200;
	rect.y = 200;
	rect.width = 100;
	rect.height = 100;
	rectangle(image, rect, Scalar(0, 0, 255), 1, 8, 0);
	//����1Ϊ��ͼ�ĵ�ͼ���߻������ƣ�����2λͼƬ����ʼ����ȣ��߶�
	//����3���������ɫ������4����0����С��0�����
	//����5��ʾ������䣬����6Ĭ��ֵΪ0
	
	//��Բ
	circle(image, Point(300, 300), 150, Scalar(1, 99, 66), 1, LINE_AA, 0);//LINE_AA �����
	//��ֱ��
	line(image, Point(100, 100), Point(300, 300), Scalar(0, 255, 0), 2, 8, 0);
	//����Բ
	RotatedRect rtt;
	rtt.center = Point(200, 200);
	rtt.size = Size(100, 200);
	rtt.angle = 0.0;
	ellipse(image, rtt, Scalar(0, 0, 255), 2, 8);
	imshow("������ʾ", image);
}

void QuickDemo::random_drawing()
{
	Mat canvas = Mat::zeros(Size(600, 600), CV_8UC3);
	int w = canvas.cols;
	int h = canvas.rows;
	RNG rng(12345);
	while (true)
	{
		int c = waitKey(10);
		if (c == 27)
		{
			break;
		}
		int x1 = rng.uniform(0, w);
		int y1 = rng.uniform(0, h);
		int x2 = rng.uniform(0, canvas.cols);
		int y2 = rng.uniform(0, h);
		int b = rng.uniform(0, 255);
		int g = rng.uniform(0, 255);
		int r = rng.uniform(0, 255);
		//canvas = Scalar(0, 0, 0);
		line(canvas, Point(x1, y1), Point(x2, y2), Scalar(b, g, r), 8, LINE_AA, 0);//line_AA��ʾȥ�����	
		imshow("���������ʾ", canvas);
	}
}

void QuickDemo::polyline_drawing_demo()
{
	Mat canvas = Mat::zeros(Size(521, 521), CV_8UC3);
	Point p1(100, 100);
	Point p2(350, 100);
	Point p3(450, 280);
	Point p4(320, 450);
	Point p5(80, 400);
	std::vector<Point>pts;//��5����װ��һ�������ڡ�
	pts.push_back(p1);//δ��ʼ������������ֻ����pushback����
	//�����ʼ���������������±������
	pts.push_back(p2);
	pts.push_back(p3);
	pts.push_back(p4);
	pts.push_back(p5);
	//���ƶ����
	polylines(canvas, pts, true, Scalar(0, 0, 255), 2, LINE_AA, 0);
	//�������
	fillPoly(canvas, pts, Scalar(122, 155, 255), LINE_AA, 0);
	imshow("���ƽ��1",canvas);
	//����API�㶨ͼƬ�Ļ������
	std::vector<std::vector<Point>>contours;
	contours.push_back(pts);
	canvas = Scalar(0, 0, 0);
	drawContours(canvas, contours, -1, Scalar(0, 0, 255), -1);
	//����2��ʾ�������ƣ�����3Ϊ����ʾ���еڼ�������εĻ��ƣ�Ϊ����ʾ������
	//����5��ʾ�Ƿ���䣬-1��䣬���������
	imshow("���ƽ��2", canvas);
}

//����1��ʾ����¼���
Point sp(-1, -1);//���Ŀ�ʼ��λ��
Point ep(-1, -1);//��������λ��
Mat temp;
static void on_draw(int event, int x, int y, int flags, void *userdata)
{
	Mat image = *((Mat*)userdata);
	if (event == EVENT_LBUTTONDOWN)//��������������
	{
		sp.x = x;
		sp.y = y;
		std::cout << "start point" << sp << std::endl;
	}
	else if (event == EVENT_LBUTTONUP)
	{
		ep.x = x;
		ep.y = y;
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;
		if (dx > 0 && dy > 0)
		{
			Rect box(sp.x, sp.y, dx, dy);
			imshow("ROI����", image(box));
			rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
			imshow("������", image);
			sp.x = -1;
			sp.y = -1;//��λ��Ϊ��һ����׼��
		}
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		if (sp.x > 0 && sp.y > 0)
		{
			ep.x = x;
			ep.y = y;
			int dx = ep.x - sp.x;
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0)
			{
				Rect box(sp.x, sp.y, dx, dy);
				temp.copyTo(image);
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
				imshow("������", image);
			}
		}
	}
}
void QuickDemo::mouse_drawing_demo(Mat &image)
{
	namedWindow("������", WINDOW_AUTOSIZE);
	setMouseCallback("������", on_draw, (void*)(&image));
	//���ô��ڵĻص�����������1��ʾ���ƣ�����2��ʾ����on_draw
	imshow("������", image);
	temp = image.clone();
}

void QuickDemo::norm_demo(Mat &image)
{
	Mat dst;//����һ����Ϊdst�Ķ�ֵ�����͵�����
	std::cout << image.type() << std::endl;//��ӡ����ͼƬ������
	image.convertTo(image, CV_32F);//��dst����ת���ɸ�����float32λ���ݡ�
	std::cout << image.type() << std::endl;//�ٴδ�ӡת�������������
	normalize(image, dst, 0, 1.0, NORM_MINMAX);//���й�һ����������һ����ķ�Χ0 - 1.0
	std::cout << dst.type() << std::endl;//��ӡ��һ������֮�������
	imshow("ͼ��Ĺ�һ��", dst);//��ʾ��һ����ͼ��
	//CV_8UC3 ,CV_32FC3  //3ͨ��ÿ��ͨ��8λ��UC����
	//ת���� 3ͨ�� ÿ��ͨ��32λ�ĸ�����
}

void QuickDemo::resize_demo(Mat &image)
{
	Mat zoomin, zoomout;
	int h = image.rows;
	int w = image.cols;
	resize(image, zoomin, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);
	//���Բ�ֵ������
	imshow("zoomin", zoomin);;
	resize(image, zoomout, Size(w*1.5, h*1.5), 0, 0, INTER_LINEAR);
	imshow("zoomout", zoomout);
}

void QuickDemo::flip_demo(Mat &image)
{
	Mat dst;
	flip(image, dst, 0);
	imshow("��ת1", dst);

	flip(image, dst, -1);
	imshow("��ת2", dst);
}

void QuickDemo::rotate_demo(Mat &image)
{
	Mat dst, M;
	int h = image.rows;//����ͼƬ�ĸ߶�
	int w = image.cols;//����ͼƬ�Ŀ��
	// ������ת������ת����Ϊͼ�����ģ���ת�Ƕ�Ϊ45�ȣ���������Ϊ1.0
	M = getRotationMatrix2D(Point(w / 2, h / 2), 45, 1.0);
	// ������ת��ͼ��Ŀ�Ⱥ͸߶�
	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));
	int nw = cos * w + sin * h;
	int nh = sin * w + cos * h;
	// ������ת�����ƽ�Ʋ��֣�ȷ��ͼ�����Ĳ���
	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	// ʹ��warpAffine�������з���任
   // ������ԭͼ�����ͼ�񡢱任�������ͼ���С����ֵ�������߽紦��ʽ���߽������ɫ
	warpAffine(image, dst, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(200, 200, 255));
	imshow("��ת��ʾ", dst);
}

void QuickDemo::video_demo(Mat &image)
{
	VideoCapture capture(0);

	Mat frame;
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);//��ȡ��Ƶ�Ŀ��
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);//��ȡ��Ƶ�ĸ߶�
	// fps �Ǻ���������Ƶ������
	double fps = capture.get(CAP_PROP_FPS);
	if (fps <= 0)
	{
		fps = 30;
	}
	std::cout << "frame width" << frame_width << std::endl;
	std::cout << "frame height" << frame_height << std::endl;
	std::cout << "frame FPS" << fps << std::endl;
	int codec = cv::VideoWriter::fourcc('a','v','c','1'); // ʹ��avc1������
	//int codec = cv::VideoWriter::fourcc('H', '2', '6', '4'); // ʹ��H.264������
	VideoWriter writer("D:/test.mp4", codec, fps, Size(frame_width, frame_height), true);
	namedWindow("frame", WINDOW_FREERATIO);
	while (true)
	{
		capture.read(frame);
		if (frame.empty())
		{
			break;
		}
		imshow("frame", frame);
		writer.write(frame);
		int c = waitKey(10);
		if (c == 27) {
			break;
		}
	}
	capture.release();
	writer.release();
}

void QuickDemo::histogram_demo(Mat &image) {
	// ��ͨ������
	std::vector<Mat> bgr_plane;
	split(image, bgr_plane);
	// �����������
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };
	Mat b_hist;
	Mat g_hist;
	Mat r_hist;
	// ����Blue, Green, red ͨ��ֱ��ͼ
	calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);

	// ��ʾֱ��ͼ
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / bins[0]);
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
	// ��һ��ֱ��ͼ����
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	// ����ֱ��ͼ����
	for (int i = 1; i < bins[0]; i++) {
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}
	// ��ʾֱ��ͼ
	namedWindow("Histogram Demo", WINDOW_AUTOSIZE);
	imshow("Histogram Demo", histImage);
}

void QuickDemo::histogram_2d_demo(Mat &image) {
	// 2D ֱ��ͼ
	Mat hsv, hs_hist;

	// ��ͼ���BGR��ɫ�ռ�ת��ΪHSV��ɫ�ռ�
	cvtColor(image, hsv, COLOR_BGR2HSV);

	// ����ɫ���ͱ��Ͷȵ�bin��
	int hbins = 30, sbins = 32;
	int hist_bins[] = { hbins, sbins };

	// ����ɫ���ͱ��Ͷȵķ�Χ
	float h_range[] = { 0, 180 };
	float s_range[] = { 0, 256 };
	const float* hs_ranges[] = { h_range, s_range };

	// �������ֱ��ͼ��ͨ��
	int hs_channels[] = { 0, 1 };

	// �����άֱ��ͼ
	calcHist(&hsv, 1, hs_channels, Mat(), hs_hist, 2, hist_bins, hs_ranges, true, false);

	// ��ȡֱ��ͼ�е����ֵ
	double maxVal = 0;
	minMaxLoc(hs_hist, 0, &maxVal, 0, 0);

	// �������ű���
	int scale = 10;
	Mat hist2d_image = Mat::zeros(sbins * scale, hbins * scale, CV_8UC3);

	// ���ƶ�άֱ��ͼ
	for (int h = 0; h < hbins; h++) {
		for (int s = 0; s < sbins; s++) {
			float binVal = hs_hist.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / maxVal);
			rectangle(hist2d_image, Point(h * scale, s * scale),
				Point((h + 1) * scale - 1, (s + 1) * scale - 1),
				Scalar::all(intensity),
				-1);
		}
	}

	// Ӧ����ɫӳ�䣬ʹֱ��ͼ�������Ӿ���
	applyColorMap(hist2d_image, hist2d_image, COLORMAP_JET);

	// ��ʾ��������ͼ��
	imshow("H-S Histogram", hist2d_image);
	imwrite("D:/hist_2d.png", hist2d_image);
}


void QuickDemo::histogram_eq_demo(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("�Ҷ�ͼ��", gray);
	Mat dst;
	equalizeHist(gray, dst);
	imshow("ֱ��ͼ���⻯��ʾ", dst);
}

void QuickDemo::blur_demo(Mat &image)
{
	Mat dst;
	blur(image, dst, Size(15, 15), Point(-1, -1));
	//����1ԭʼͼ�񣬲���2���֮���ͼ�񣬲���3����ľ����С��֧�ֵ��л��ߵ��еľ������������4�������ʼ�㡣
	imshow("ͼ��������", dst);
}

void QuickDemo::gaussian_blur_demo(Mat &image) {
	Mat dst;
	GaussianBlur(image, dst, Size(0, 0), 15);
	imshow("��˹ģ��", dst);
}

void QuickDemo::bifilter_demo(Mat &image) {
	Mat dst;
	bilateralFilter(image, dst, 0, 100, 10);
	imshow("˫��ģ��", dst);
}

void QuickDemo::face_detection_demo()
{
	//����Ԥѵ����TensorFlowģ��
	dnn::Net net = dnn::readNetFromTensorflow("D:/opencv/opencv/sources/samples/dnn/opencv_face_detector_uint8.pb", "D:/opencv/opencv/sources/samples/dnn/opencv_face_detector.pbtxt");
	VideoCapture capture(0);
	Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		//������ͼ��ת��Ϊ����������ĸ�ʽ,���ű���1.0��Ŀ��ͼ���С300*300�� ��ֵ104 177 123
		Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
		//��Ԥ������ͼ����Ϊ��������
		net.setInput(blob);// NCHW
		//��ȡ�����
		Mat probs = net.forward();
		//�������ת��Ϊ����
		Mat detectionMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());
		// �������
		for (int i = 0; i < detectionMat.rows; i++) {
			float confidence = detectionMat.at<float>(i, 2);
			//ֻ�������Ŷȴ���0.5�ļ����
			if (confidence > 0.5) {
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3)*frame.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4)*frame.rows);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5)*frame.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6)*frame.rows);
				Rect box(x1, y1, x2 - x1, y2 - y1);
				rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
		imshow("���������ʾ", frame);
		int c = waitKey(1);
		if (c == 27) { // �˳�
			break;
		}
	}
}