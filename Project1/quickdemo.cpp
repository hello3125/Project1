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

void QuickDemo::pixel_visit_demo(Mat &image)
{
	int dims = image.channels();
	int h = image.rows;
	int w = image.cols;
	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++)
		{
			if (dims == 1) //单通道的灰度图像
			{
				int pv = image.at<uchar>(row, col);//得到像素值
				image.at<uchar>(row, col) = 255 - pv;//给像素值重新赋值

			}
			if (dims == 3) //三通道的彩色图像
			{
				Vec3b bgr = image.at<Vec3b>(row, col); //opencv特定的类型，获取三维颜色，3个值
				image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
				image.at<Vec3b>(row, col)[1] = 255 - bgr[1];
				image.at<Vec3b>(row, col)[2] = 255 - bgr[2];//对彩色图像读取它的像素值，并且对像素值进行改写。
			}
		}
	}
	namedWindow("像素读写演示", WINDOW_FREERATIO);
	imshow("像素读写演示", image);
}


void QuickDemo::operators_demo(Mat &image)
{
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	dst = image - Scalar(50, 50, 50);
	m = Scalar(50, 50, 50);
	multiply(image, m, dst);//乘法操作
	namedWindow("乘法操作", WINDOW_FREERATIO);
	imshow("乘法操作", dst);
	add(image, m, dst);//加法操作
	namedWindow("加法操作", WINDOW_FREERATIO);
	imshow("加法操作", dst);
	subtract(image, m, dst);//减法操作
	namedWindow("减法操作", WINDOW_FREERATIO);
	imshow("减法操作", dst);
	divide(image, m, dst);//除法操作
	namedWindow("除法操作", WINDOW_FREERATIO);
	imshow("除法操作", dst);
	//加法操作底层
	int dims = image.channels();
	int h = image.rows;
	int w = image.cols;
	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++)
		{
			Vec3b p1 = image.at<Vec3b>(row, col); //opencv特定的类型，获取三维颜色，3个值
			Vec3b p2 = m.at<Vec3b>(row, col);
			dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(p1[0] + p2[0]);//saturate_cast用来防爆，小于0就是0，大于255就是255
			dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(p1[1] + p2[1]);
			dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(p1[2] + p2[2]);//对彩色图像读取它的像素值，并且对像素值进行改写。
		}
	}
	namedWindow("加法操作2", WINDOW_FREERATIO);
	imshow("加法操作2", dst);
}

static void on_lightness(int b, void* userData)
{
	Mat image = *((Mat*)userData);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	m = Scalar(b, b, b);//创建调整亮度的数值
	addWeighted(image, 1.0, m, 0, b, dst);//融合两张图
	imshow("亮度与对比度调整", dst);//显示调整亮度之后的图片
}

static void on_contrast(int b, void* userData)
{
	Mat image = *((Mat*)userData);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	double contrast = b / 100.0;
	addWeighted(image, contrast, m, 0.0, 0, dst);//融合两张图
	imshow("亮度与对比度调整", dst);//显示调整亮度之后的图片
}

void QuickDemo::tracking_bar_demo(Mat &image)
{
	namedWindow("亮度与对比度调整", WINDOW_FREERATIO);
	Mat dst = Mat::zeros(image.size(), image.type());//图片的初始化创建一个和image大小相等，种类相同的图像
	Mat m = Mat::zeros(image.size(), image.type());//图片的初始化创建一个和image大小相等，种类相同的图像
	Mat src = image;//给src赋值
	int lightness = 50;
	int max_value = 100;//定义最大值为100
	int contrast_value = 100;//对比度
	createTrackbar("Value Bar:", "亮度与对比度调整", &lightness, max_value, on_lightness, (void*)(&image));//调用函数亮度调节功能。
	createTrackbar("Contrast Bar:", "亮度与对比度调整", &contrast_value, 200, on_contrast, (void*)(&image));//对比度

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
		imshow("键盘响应", dst);
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
		imshow("循环播放", dst);
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
	imshow("像素位与", dst);
	bitwise_or(m1, m2, dst);//位操作或
	imshow("像素位或", dst);
	bitwise_not(image, dst);//取反操作
	imshow("像素位非", dst);
	bitwise_xor(m1, m2, dst);//异或操作
	imshow("像素位异或", dst);
	
}

void QuickDemo::channels_demo(Mat &image)
{
	std::vector<Mat> mv;
	split(image, mv);
	imshow("蓝色", mv[0]);
	imshow("绿色", mv[1]);
	imshow("红色", mv[2]);

	//将通道合并，使用 merge
	Mat	dst = Mat::zeros(image.size(),image.type());
	mv[0] = 0;
	mv[1] = 0;
	merge(mv, dst);
	imshow("红色图像", dst);

	//mv[0] = 0;
	//mv[2] = 0;
	//merge(mv, dst);
	//imshow("绿色图像", dst);

	int from_to[] = { 0,2,1,1,2,0 }; //BGR图像转换为RGB图像
	//把通道相互交换，第0->第2，第一->第一，第二->第0
	mixChannels(&image, 1, &dst, 1, from_to, 3);//3表示3个通道
	//参数1指针引用图像->参数2引用到dst
	imshow("通道混合", dst);
}

void QuickDemo::inrange_demo(Mat &image)
{
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);
	//35,43,46根据图片中绿色最低来确定最小值。
	//77,255,255 提取
	//参数1低范围，参数2高范围
	//将hsv中的由低到高的像素点提取出来并且存储到mask当中。
	imshow("mask", hsv);
	Mat redback = Mat::zeros(image.size(), image.type());
	redback = Scalar(40, 40, 200);
	bitwise_not(mask, mask);
	imshow("mask", mask);
	image.copyTo(redback, mask);//把redback复制到mask，mask通过inRange得到。
	imshow("roi 区域提取 ", redback);
}

void QuickDemo::pixel_statistic_demo(Mat &image)
{
	double minv, maxv;//定义最值
	Point minLoc, maxLoc;//定义最值地址
	std::vector<Mat>mv;//mv是一个Mat类型的容器 装在这个容器内
	split(image, mv);
	for (int i = 0; i < mv.size(); i++)
	{
		//分别打印各个通道的数值
		minMaxLoc(mv[i], &minv, &maxv, &minLoc, &maxLoc, Mat());//求出图像的最大值和最小值。
		std::cout << "No.channels:" << i << "min value:" << minv << "max value:" << maxv << std::endl;
	}
	Mat mean, stddev;//平均值、方差
	meanStdDev(image, mean, stddev);//求出图像的均值和方差
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
	//参数1为绘图的底图或者画布名称，参数2位图片的起始，宽度，高度
	//参数3代表填充颜色。参数4大于0是线小于0是填充
	//参数5表示邻域填充，参数6默认值为0
	
	//画圆
	circle(image, Point(300, 300), 150, Scalar(1, 99, 66), 1, LINE_AA, 0);//LINE_AA 抗锯齿
	//画直线
	line(image, Point(100, 100), Point(300, 300), Scalar(0, 255, 0), 2, 8, 0);
	//画椭圆
	RotatedRect rtt;
	rtt.center = Point(200, 200);
	rtt.size = Size(100, 200);
	rtt.angle = 0.0;
	ellipse(image, rtt, Scalar(0, 0, 255), 2, 8);
	imshow("绘制演示", image);
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
		line(canvas, Point(x1, y1), Point(x2, y2), Scalar(b, g, r), 8, LINE_AA, 0);//line_AA表示去掉锯齿	
		imshow("随机绘制演示", canvas);
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
	std::vector<Point>pts;//将5个点装入一个容器内。
	pts.push_back(p1);//未初始化数组容量，只能用pushback操作
	//如果初始化，可以用数组下标操作。
	pts.push_back(p2);
	pts.push_back(p3);
	pts.push_back(p4);
	pts.push_back(p5);
	//绘制多边形
	polylines(canvas, pts, true, Scalar(0, 0, 255), 2, LINE_AA, 0);
	//填充多边形
	fillPoly(canvas, pts, Scalar(122, 155, 255), LINE_AA, 0);
	imshow("绘制结果1",canvas);
	//单个API搞定图片的绘制填充
	std::vector<std::vector<Point>>contours;
	contours.push_back(pts);
	canvas = Scalar(0, 0, 0);
	drawContours(canvas, contours, -1, Scalar(0, 0, 255), -1);
	//参数2表示容器名称，参数3为正表示进行第几个多边形的绘制，为负表示都绘制
	//参数5表示是否填充，-1填充，正数不填充
	imshow("绘制结果2", canvas);
}

//参数1表示鼠标事件。
Point sp(-1, -1);//鼠标的开始的位置
Point ep(-1, -1);//鼠标结束的位置
Mat temp;
static void on_draw(int event, int x, int y, int flags, void *userdata)
{
	Mat image = *((Mat*)userdata);
	if (event == EVENT_LBUTTONDOWN)//如果鼠标的左键按下
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
			imshow("ROI区域", image(box));
			rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
			imshow("鼠标绘制", image);
			sp.x = -1;
			sp.y = -1;//复位，为下一次做准备
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
				imshow("鼠标绘制", image);
			}
		}
	}
}
void QuickDemo::mouse_drawing_demo(Mat &image)
{
	namedWindow("鼠标绘制", WINDOW_AUTOSIZE);
	setMouseCallback("鼠标绘制", on_draw, (void*)(&image));
	//设置窗口的回调函数。参数1表示名称，参数2表示调用on_draw
	imshow("鼠标绘制", image);
	temp = image.clone();
}

void QuickDemo::norm_demo(Mat &image)
{
	Mat dst;//定义一个名为dst的二值化类型的数据
	std::cout << image.type() << std::endl;//打印出来图片的类型
	image.convertTo(image, CV_32F);//将dst数据转换成浮点型float32位数据。
	std::cout << image.type() << std::endl;//再次打印转换后的数据类型
	normalize(image, dst, 0, 1.0, NORM_MINMAX);//进行归一化操作，归一化后的范围0 - 1.0
	std::cout << dst.type() << std::endl;//打印归一化操作之后的数据
	imshow("图像的归一化", dst);//显示归一化的图像
	//CV_8UC3 ,CV_32FC3  //3通道每个通道8位的UC类型
	//转换后 3通道 每个通道32位的浮点数
}

void QuickDemo::resize_demo(Mat &image)
{
	Mat zoomin, zoomout;
	int h = image.rows;
	int w = image.cols;
	resize(image, zoomin, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);
	//线性差值操作。
	imshow("zoomin", zoomin);;
	resize(image, zoomout, Size(w*1.5, h*1.5), 0, 0, INTER_LINEAR);
	imshow("zoomout", zoomout);
}

void QuickDemo::flip_demo(Mat &image)
{
	Mat dst;
	flip(image, dst, 0);
	imshow("翻转1", dst);

	flip(image, dst, -1);
	imshow("翻转2", dst);
}

void QuickDemo::rotate_demo(Mat &image)
{
	Mat dst, M;
	int h = image.rows;//定义图片的高度
	int w = image.cols;//定义图片的宽度
	// 生成旋转矩阵，旋转中心为图像中心，旋转角度为45度，缩放因子为1.0
	M = getRotationMatrix2D(Point(w / 2, h / 2), 45, 1.0);
	// 计算旋转后图像的宽度和高度
	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));
	int nw = cos * w + sin * h;
	int nh = sin * w + cos * h;
	// 调整旋转矩阵的平移部分，确保图像中心不变
	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	// 使用warpAffine函数进行仿射变换
   // 参数：原图像、输出图像、变换矩阵、输出图像大小、插值方法、边界处理方式、边界填充颜色
	warpAffine(image, dst, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(200, 200, 255));
	imshow("旋转演示", dst);
}

void QuickDemo::video_demo(Mat &image)
{
	VideoCapture capture(0);

	Mat frame;
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);//获取视频的宽度
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);//获取视频的高度
	// fps 是衡量处理视频的能力
	double fps = capture.get(CAP_PROP_FPS);
	if (fps <= 0)
	{
		fps = 30;
	}
	std::cout << "frame width" << frame_width << std::endl;
	std::cout << "frame height" << frame_height << std::endl;
	std::cout << "frame FPS" << fps << std::endl;
	int codec = cv::VideoWriter::fourcc('a','v','c','1'); // 使用avc1编码器
	//int codec = cv::VideoWriter::fourcc('H', '2', '6', '4'); // 使用H.264编码器
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
	// 三通道分离
	std::vector<Mat> bgr_plane;
	split(image, bgr_plane);
	// 定义参数变量
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };
	Mat b_hist;
	Mat g_hist;
	Mat r_hist;
	// 计算Blue, Green, red 通道直方图
	calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);

	// 显示直方图
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / bins[0]);
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
	// 归一化直方图数据
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	// 绘制直方图曲线
	for (int i = 1; i < bins[0]; i++) {
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}
	// 显示直方图
	namedWindow("Histogram Demo", WINDOW_AUTOSIZE);
	imshow("Histogram Demo", histImage);
}

void QuickDemo::histogram_2d_demo(Mat &image) {
	// 2D 直方图
	Mat hsv, hs_hist;

	// 将图像从BGR颜色空间转换为HSV颜色空间
	cvtColor(image, hsv, COLOR_BGR2HSV);

	// 定义色调和饱和度的bin数
	int hbins = 30, sbins = 32;
	int hist_bins[] = { hbins, sbins };

	// 定义色调和饱和度的范围
	float h_range[] = { 0, 180 };
	float s_range[] = { 0, 256 };
	const float* hs_ranges[] = { h_range, s_range };

	// 定义计算直方图的通道
	int hs_channels[] = { 0, 1 };

	// 计算二维直方图
	calcHist(&hsv, 1, hs_channels, Mat(), hs_hist, 2, hist_bins, hs_ranges, true, false);

	// 获取直方图中的最大值
	double maxVal = 0;
	minMaxLoc(hs_hist, 0, &maxVal, 0, 0);

	// 设置缩放比例
	int scale = 10;
	Mat hist2d_image = Mat::zeros(sbins * scale, hbins * scale, CV_8UC3);

	// 绘制二维直方图
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

	// 应用颜色映射，使直方图更易于视觉化
	applyColorMap(hist2d_image, hist2d_image, COLORMAP_JET);

	// 显示并保存结果图像
	imshow("H-S Histogram", hist2d_image);
	imwrite("D:/hist_2d.png", hist2d_image);
}


void QuickDemo::histogram_eq_demo(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("灰度图像", gray);
	Mat dst;
	equalizeHist(gray, dst);
	imshow("直方图均衡化演示", dst);
}

void QuickDemo::blur_demo(Mat &image)
{
	Mat dst;
	blur(image, dst, Size(15, 15), Point(-1, -1));
	//参数1原始图像，参数2卷积之后的图像，参数3卷积的矩阵大小，支持单行或者单列的卷积操作，参数4卷积的起始点。
	imshow("图像卷积操作", dst);
}

void QuickDemo::gaussian_blur_demo(Mat &image) {
	Mat dst;
	GaussianBlur(image, dst, Size(0, 0), 15);
	imshow("高斯模糊", dst);
}

void QuickDemo::bifilter_demo(Mat &image) {
	Mat dst;
	bilateralFilter(image, dst, 0, 100, 10);
	imshow("双边模糊", dst);
}

void QuickDemo::face_detection_demo()
{
	//加载预训练的TensorFlow模型
	dnn::Net net = dnn::readNetFromTensorflow("D:/opencv/opencv/sources/samples/dnn/opencv_face_detector_uint8.pb", "D:/opencv/opencv/sources/samples/dnn/opencv_face_detector.pbtxt");
	VideoCapture capture(0);
	Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		//将输入图像转换为神经网络所需的格式,缩放比例1.0，目标图像大小300*300， 均值104 177 123
		Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
		//将预处理后的图像作为网络输入
		net.setInput(blob);// NCHW
		//获取检测结果
		Mat probs = net.forward();
		//将检测结果转换为矩阵
		Mat detectionMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());
		// 解析结果
		for (int i = 0; i < detectionMat.rows; i++) {
			float confidence = detectionMat.at<float>(i, 2);
			//只考虑置信度大于0.5的检测结果
			if (confidence > 0.5) {
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3)*frame.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4)*frame.rows);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5)*frame.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6)*frame.rows);
				Rect box(x1, y1, x2 - x1, y2 - y1);
				rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
		imshow("人脸检测演示", frame);
		int c = waitKey(1);
		if (c == 27) { // 退出
			break;
		}
	}
}