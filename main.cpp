#include<iostream>

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat Findcolor(Mat frame);
Mat Getmaxcontour(Mat frame);
Mat Getcircle(Mat frame);

const int iLowH = 220/ 2;
const int iHighH = 250/ 2;

const int iLowS = 40 * 255 / 100;
const int iHighS = 100 * 255 / 100;

const int iLowV = 20* 255 / 100;
const int iHighV = 100* 255 / 100;

int main(int argc, char ** argv)
{
	VideoCapture cap(1);
	
	if (!cap.isOpened())
	{
		cout << "error: open the cap failed" << endl;
		return -1;
	}
	else cout <<"the camera4 open successful"<< endl;


	namedWindow("Camera1");
	while (1)
	{
		Mat frame;
		cap >> frame;
		//frame=Findcolor(frame);
		//frame = Getcircle(frame);					//寻找到合适的圆形
		//frame=Getmaxcontour(frame);				//寻找二值图像最大的连通域
		imshow("Camera1", frame);                 //显示图像
		cout<<"successful"<<endl;
		if (waitKey(30) >= 0)
		{
			break;
		}
	}

	cap.release();

	return 1;

}

Mat Findcolor(Mat frame)
{
	Mat HSV_image;

	cvtColor(frame, HSV_image, COLOR_RGB2HSV);  //转换成HSV格式
	GaussianBlur(frame, frame, Size(5, 5), 0, 0);  //高斯滤波

	Mat Thre_image;
	inRange(HSV_image, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), Thre_image);

	//开闭操作
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(Thre_image, Thre_image, MORPH_OPEN, element);

	morphologyEx(Thre_image, Thre_image, MORPH_CLOSE, element);

	return Thre_image;
}

Mat Getcircle(Mat frame)          //�ҵ�Բ
{
	vector<Vec3f> circles;

	HoughCircles(frame, circles, CV_HOUGH_GRADIENT, 1.5, 20, 150, 80, 0, 0); //ͼ����Ѱ��Բ

	Mat result; //= Mat::zeros(frame.rows, frame.cols, CV_8UC3);
	cvtColor(frame, result, CV_GRAY2BGR);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(round(circles[i][0]), round(circles[i][1]));
		int radius = round(circles[i][2]);

		circle(result, center, radius, Scalar(155, 50, 255), 3, 8, 0);
	}

	return result;
}




Mat Getmaxcontour(Mat frame)        //Ѱ�������е����ֵ
{
	vector< vector<Point> > contours;   //����

	Mat frametemp;
	frame.copyTo(frametemp);

	findContours(frametemp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);  //Ѱ������

	double maxArea = 0;
	vector<Point> maxContour;
	size_t maxContourIndx = 0;

	for (size_t i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > maxArea)
		{
			maxArea = area;
			maxContour = contours[i];  //Ѱ���������
			maxContourIndx = i;
		}
	}

	Mat result = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	if(contours.size()>0)
		drawContours(result, contours, maxContourIndx, Scalar(255), CV_FILLED);
	
	/*frame.copyTo(result);

	Rect maxRect = boundingRect(maxContour);
	rectangle(result, maxRect, cv::Scalar(255));*/   //���ֽ��ok����

	return result;
}



