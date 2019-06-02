#include <string>  
#include <list>  
#include <vector>  
#include <map>  
#include <stack>

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

#include<iostream>
#include<ctime>


using namespace std;
using namespace cv;

void DrawSelectLines( Mat& dst, vector<Vec2f> lines);
Mat ROImask(Mat src, vector<Vec2f> prunedLines);
Mat OstuSeg(Mat src);
Mat SobelSeg(Mat src);
Mat Morph(Mat src);
vector<Mat> EdgeSegHSV(Mat src_H, Mat src_S, Mat src_V, vector<Vec2f> lines);
Mat VergeHSV(Mat src_H, Mat src_S, Mat src_V);
void bwLabel(const Mat& imgBw, Mat& imgLabeled);
Mat LargestArea(Mat src);
Point2f findCentroid(Mat src);
void clearLeastArea(Mat& src);


int main(int argc, char** argv) {
/*-------------------计算程序的运行时间------------------*/
	clock_t startTime,endTime;
	startTime = clock();
/*--------------------图像读取和预处理-------------------*/
	Mat img_src = imread("1.jpg");

	if (img_src.empty()) {
		cout << "the picture can not be loaded" << endl;
		return 0;
	}
	else {
		//cvtColor(img_src, img_src, CV_BGR2RGB);
	}

	Mat img_src_cprs;
	resize(img_src, img_src_cprs, Size(400, 300)); //高斯金字塔
	
	Mat img_riverroc;
	img_src_cprs.copyTo(img_riverroc);

	Mat img_riverroc_3u;
	img_riverroc.copyTo(img_riverroc_3u);  //河岸线检测的时候保存的RGB副本

	Mat img_riverroc_rslt;
	img_src_cprs.copyTo(img_riverroc_rslt); //河岸检测线

	//imshow("tmp",img_riverroc);


/*----------------------河岸线检测----------------------*/
    cvtColor(img_riverroc, img_riverroc, CV_RGB2GRAY); //图片黑白处理
	//equalizeHist(img_riverroc, img_riverroc);
	GaussianBlur(img_riverroc, img_riverroc, Size(5, 5), 0, 0, BORDER_DEFAULT); //高斯降噪

	//imshow("GaussianBlurAndGrayAndEqual", img_riverroc);

	
	//边缘检测
	Canny(img_riverroc, img_riverroc, 30, 80 ,3); //canny边缘算法
	//threshold(img_riverroc, img_riverroc, 130, 255, CV_THRESH_OTSU);

	//bitwise_not(img_riverroc, img_riverroc); //颜色反转

    //面积最小的轮廓处理
	//clearLeastArea(img_riverroc);
	//imshow("canny", img_riverroc);    //显示Canny边缘算法结果内容

	//霍夫变换
	vector<Vec2f> lines;     //检测到的水岸线
	vector<Vec2f> prunedLines;  //筛选过后的角度
	HoughLines(img_riverroc, lines, 1, CV_PI / 90, 110, 0, 0); //霍夫变换
	cout << "lines:" << lines.size() << endl;
	
	//筛选直线角度
	
	for (int i = 1; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		if (1) {
			prunedLines.push_back(lines[i]);
		}
	}

	//绘制直线
	DrawSelectLines(img_riverroc_3u, prunedLines);

	cout <<"prunedLines:"<<prunedLines.size() << endl;
	//imshow("lines", img_riverroc_3u);

	//ROI区域的提取
	Mat mask;
	mask = ROImask(img_riverroc_3u, prunedLines);
	cvtColor(mask, mask, CV_GRAY2BGR);
	bitwise_and(img_riverroc_rslt, mask, img_riverroc_rslt);

	//imshow("riverroc_rslt", img_riverroc_rslt);    //显示相关内容
/*-----------------------目标检测-----------------------*/
	//准备阶段定义主要图片
	Mat target_roc;
	img_riverroc_rslt.copyTo(target_roc);
	//GaussianBlur(target_roc, target_roc, Size(3, 3), 0, 0, BORDER_DEFAULT); //高斯降噪
	//resize(target_roc, target_roc, Size(200,150));   //重新定义图片大小
	//imshow("target_roc", target_roc);
	

	//HSV颜色分离
	//equalizeHist(target_roc, target_roc);
	Mat target_roc_hsv;
	cvtColor(target_roc, target_roc_hsv, CV_BGR2HSV_FULL);     //图像颜色转换

	vector<Mat> channel_hsv;

	split(target_roc, channel_hsv);

	//显示三个通道颜色
	Mat target_H = channel_hsv.at(0);
	Mat target_S = channel_hsv.at(1);
	Mat target_V = channel_hsv.at(2);

	//imshow("H", target_H);
	//imshow("S", target_S);
	//imshow("V", target_V);


	//进行各个通道的分离后进行边缘信息筛查
	vector<Mat> target_edges = EdgeSegHSV(target_H, target_S, target_V, prunedLines);

	Mat target_H_edge = target_edges.at(0);
	Mat target_S_edge = target_edges.at(1);
	Mat target_V_edge = target_edges.at(2);

	//imshow("edge_H", target_H_edge);
	//imshow("edge_S", target_S_edge);
	//imshow("edge_V", target_V_edge);

	//DS边缘信息融合
	Mat target_verge = VergeHSV(target_H_edge, target_S_edge, target_V_edge);
	

	//形态学降噪
	target_verge = Morph(target_verge);
	//imshow("verge", target_verge);

	//连通域标记法
	Mat target_label;
	bwLabel(target_verge, target_label);

	//保留最大连通域
	target_label=LargestArea(target_label);
	//imshow("label", target_label);

	//寻找图像重心
	Point2f centroid;
	centroid=findCentroid(target_label);
	
	cout << centroid.x << " " << centroid.y << endl;

	//绘制图像重心
	Mat target_rslt;
	img_src_cprs.copyTo(target_rslt);
	circle(target_rslt, centroid, 2, Scalar(0,125,245), 2, CV_AA);//绘制图形中
	//imshow("result", target_rslt);
	//waitKey();

	//结束计时
	endTime = clock();
	cout << "the run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" <<endl;

	return 1;
}


//直线绘制与筛选
void DrawSelectLines(Mat& dst, vector<Vec2f> lines) {

	for (size_t i = 0; i < lines.size(); i++)
	{
		if (i >= lines.size() || i < 0) { cout << "vetcor下标越界" << endl; break; }

		float rho = lines[i][0], theta = lines[i][1];
		//筛选角度
		//rho = rho * (855 / 240);
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		//line(img_src, pt1, pt2, Scalar(55, 100, 195), 1, CV_AA); 

		if (dst.step1(1) == 3) {
			line(dst, pt1, pt2, Scalar(55, 100, 195), 1, CV_AA); //绘制线条
		}
		else if (dst.step1(1) == 1) {
			line(dst, pt1, pt2, 0, 15, CV_AA); //绘制线条
		}
		
	}
}

//生成掩膜
Mat ROImask(Mat src,vector<Vec2f> prunedLines) {
	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	bitwise_not(mask, mask);   //反色处理

	for (size_t i = 0; i < prunedLines.size(); i++) {
		if (i >= prunedLines.size() || i < 0) { cout << "vetcor下标越界" << endl; break; }
		float rho = prunedLines[i][0], theta = prunedLines[i][1];
		Point pt1, pt2;

		vector< vector<Point> > mask_contour;
		vector<Point> points;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;

		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));

		points.push_back(pt1);
		points.push_back(pt2);
		points.push_back(Point(src.size().width, 0));
		points.push_back(Point(0, 0));

		mask_contour.push_back(points);
		drawContours(mask, mask_contour, 0, Scalar(0, 0, 0), -1);

		//作于操作
		//bitwise_and(img_riverroc_rslt, roi, img_riverroc_rslt);

	}

	return mask.clone();
}

//最大熵分割
Mat EntropySeg(Mat src,int &T)
{
	int tbHist[256] = { 0 };                                          //每个像素值个数
	int index = 0;                                                  //最大熵对应的灰度
	double Property = 0.0;                                          //像素所占概率
	double maxEntropy = -1.0;                                       //最大熵
	double frontEntropy = 0.0;                                      //前景熵
	double backEntropy = 0.0;                                       //背景熵
	//纳入计算的总像素数
	int TotalPixel = 0;
	int nCol = src.cols * src.channels();                           //每行的像素个数
	for (int i = 0; i < src.rows; i++)
	{
		uchar* pData = src.ptr<uchar>(i);
		for (int j = 0; j < nCol; ++j)
		{
			++TotalPixel;
			tbHist[pData[j]] += 1;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		//计算背景像素数
		double backTotal = 0;
		for (int j = 0; j < i; j++)
		{
			backTotal += tbHist[j];
		}

		//背景熵
		for (int j = 0; j < i; j++)
		{
			if (tbHist[j] != 0)
			{
				Property = tbHist[j] / backTotal;
				backEntropy += -Property * logf((float)Property);
			}
		}
		//前景熵
		for (int k = i; k < 256; k++)
		{
			if (tbHist[k] != 0)
			{
				Property = tbHist[k] / (TotalPixel - backTotal);
				frontEntropy += -Property * logf((float)Property);
			}
		}

		if (frontEntropy + backEntropy > maxEntropy)    //得到最大熵
		{
			maxEntropy = frontEntropy + backEntropy;
			index = i;
		}
		//清空本次计算熵值
		frontEntropy = 0.0;
		backEntropy = 0.0;
	}
	Mat dst;
	//index += 20;
	cv::threshold(src, dst, index, 255, 0);             //进行阈值分割
	return dst.clone();
}

//Otus分隔
Mat OstuSeg(Mat src)
{
	int tbHist[256] = { 0 };                      //直方图数组
	double average = 0.0;                       //平均像素值
	double cov = 0.0;                           //方差
	double maxcov = 0.0;                        //方差最大值
	int index = 0;                              //分割像素值
	Mat dst;
	int nCol = src.cols * src.channels();       //每行的像素个数
	for (int i = 0; i < src.rows; i++)
	{
		uchar* pData = src.ptr<uchar>(i);
		for (int j = 0; j < nCol; ++j)
		{
			tbHist[pData[j]] += 1;
		}
	}

	int sum = 0;
	for (int i = 0; i < 256; ++i)
		sum += tbHist[i];

	double w0 = 0.0, w1 = 0.0, u0 = 0.0, u1 = 0.0;
	int count0 = 0;
	for (int i = 0; i < 255; ++i)
	{
		u0 = 0;
		count0 = 0;
		for (int j = 0; j <= i; ++j)
		{
			u0 += j * tbHist[j];
			count0 += tbHist[j];
		}
		u0 = u0 / count0;
		w0 = (float)count0 / sum;

		u1 = 0;
		for (int j = i + 1; j < 256; ++j)
			u1 += j * tbHist[j];

		u1 = u1 / (sum - count0);
		w1 = 1 - w0;
		cov = w0 * w1 * (u1 - u0) * (u1 - u0);
		if (cov > maxcov)
		{
			maxcov = cov;
			index = i;
		}
	}
	cv::threshold(src, dst, index, 255, 0);    //进行阈值分割
	return dst.clone();
}

//sobel算子
Mat SobelSeg(Mat src) {

	Mat xdst;
	Mat ydst;
	Mat dst;

	Sobel(src, xdst, -1, 1, 0);
	Sobel(src, ydst, -1, 0, 1);
	
	addWeighted(xdst, 0.5, ydst, 0.5, 1, dst);
	//equalizeHist(dst, dst);
	
	return dst.clone();
}

//形态学降噪
Mat Morph(Mat src) {

	Mat dst;
	Mat element = getStructuringElement(MORPH_RECT, Size(1, 1)); //开闭操作要运用到的元素
	morphologyEx(src, dst, MORPH_OPEN, element);    //开运算

	Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3)); //开闭操作要运用到的元素
    morphologyEx(dst, dst, MORPH_CLOSE, element2);    //闭运算

	return dst.clone();
}

//分离进行信息筛选
vector<Mat> EdgeSegHSV(Mat src_H, Mat src_S, Mat src_V, vector<Vec2f> lines) {

	vector<Mat> dst;
	Mat dst_H;
	Mat dst_S;
	Mat dst_V;

	dst_H = SobelSeg(src_H);
	DrawSelectLines(dst_H, lines);
	dst_H = OstuSeg(dst_H);                     //最大间方差分割

	//target_S = EntropySeg(target_S);              //最大熵分割；
	dst_S = SobelSeg(src_S);
	DrawSelectLines(dst_S, lines);
	dst_S = OstuSeg(dst_S);                     //最大间方差分割；

	//target_V = EntropySeg(target_V);              //最大熵分割；
	dst_V = SobelSeg(src_V);
	DrawSelectLines(dst_V, lines);
	dst_V = OstuSeg(dst_V);                     //最大间方差分割；

	dst.push_back(dst_H);
	dst.push_back(dst_S);
	dst.push_back(dst_V);

	return dst;
}

//信息融合
Mat VergeHSV(Mat src_H, Mat src_S, Mat src_V) {

	Mat dst = Mat::zeros(src_H.size(), CV_8UC1);

	int cols = src_H.cols;
	int rows = src_H.rows;

	for (int i = 0; i < rows; ++i) {
		
		const uchar* indata_H = src_H.ptr<uchar>(i);
		const uchar* indata_S = src_S.ptr<uchar>(i);
		const uchar* indata_V = src_V.ptr<uchar>(i);

		uchar* outdata = dst.ptr<uchar>(i);

		for (int j = 0; j < cols; ++j) {
			uchar flag = 0;
			
	
			if (indata_H[j] > 0) {
				flag++;
			}
			if (indata_S[j] > 0) {
				flag++;
			}
			if (indata_V[j] > 0) {
				flag++;
			}
			if (flag==3) {
				outdata[j] = 255;
			}
		}
		
	}
	
	return dst.clone();
}

//连通域标记算法
void bwLabel(const Mat& imgBw, Mat& imgLabeled)
{
	// 对图像周围扩充一格
	Mat imgClone = Mat(imgBw.rows + 1, imgBw.cols + 1, imgBw.type(), Scalar(0));
	imgBw.copyTo(imgClone(Rect(1, 1, imgBw.cols, imgBw.rows)));

	imgLabeled.create(imgClone.size(), imgClone.type());
	imgLabeled.setTo(Scalar::all(0));

	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(imgClone, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	vector<int> contoursLabel(contours.size(), 0);
	int numlab = 1;
	// 标记外围轮廓
	for (vector< vector<Point> >::size_type i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][3] >= 0) // 有父轮廓
		{
			continue;
		}
		for (vector<Point>::size_type k = 0; k != contours[i].size(); k++)
		{
			imgLabeled.at<uchar>(contours[i][k].y, contours[i][k].x) = numlab;
		}
		contoursLabel[i] = numlab++;
	}
	// 标记内轮廓
	for (vector< vector<Point> >::size_type i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][3] < 0)
		{
			continue;
		}
		for (vector<Point>::size_type k = 0; k != contours[i].size(); k++)
		{
			imgLabeled.at<uchar>(contours[i][k].y, contours[i][k].x) = contoursLabel[hierarchy[i][3]];
		}
	}
	// 非轮廓像素的标记
	for (int i = 0; i < imgLabeled.rows; i++)
	{
		for (int j = 0; j < imgLabeled.cols; j++)
		{
			if (imgClone.at<uchar>(i, j) != 0 && imgLabeled.at<uchar>(i, j) == 0)
			{
				imgLabeled.at<uchar>(i, j) = imgLabeled.at<uchar>(i, j - 1);
			}
		}
	}
	imgLabeled = imgLabeled(Rect(1, 1, imgBw.cols, imgBw.rows)).clone(); // 将边界裁剪掉1像素
}

//提取寻找最大连通域
Mat LargestArea(Mat src) {
	Mat dst=src.clone();  //输出函数
	map<uchar, size_t> index; //每个连通域及其面积

	int cols = src.cols;
	int rows = src.rows;

	for (int i = 0; i < rows; ++i) {

		const uchar* indata = src.ptr<uchar>(i);
		for (int j = 0; j < cols; ++j) {
			uchar label = indata[j];
			if (label == 0) {
				continue;
			}
			else if(index.count(label)){
				index[label] = ++index[label];
			}
			else if (!index.count(label)) {
				index[label] = 1;
			}
		}

	}

	//寻找最大连通域以及其标号
	uchar largeIndex;
	size_t largeArea=0;
	for (map<uchar, size_t>::iterator iter = index.begin(); iter != index.end();iter++) {
		cout<<"index:"<<int(iter->first)<<"area:"<<iter->second<<endl;
		if (iter->second > largeArea) {
			largeArea = iter->second;
			largeIndex = iter->first;
		}
	}

	for (int i = 0; i < rows; ++i) {

		uchar* indata = dst.ptr<uchar>(i);
		for (int j = 0; j < cols; ++j) {
			if (indata[j] == largeIndex) {
				indata[j] = 255;
			}
			else {
				indata[j] = 0;
			}

		}

	}
	
	return dst;
}

//通过图像矩阵得到物体的标记
Point2f findCentroid(Mat src) {
	Point2f centroid;

	//求图像矩
	Moments m;
	m = moments(src, true);
	
	centroid = Point2f(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));

	return centroid;
}

//删除最小轮廓
void clearLeastArea(Mat& src) {
	vector< vector<Point> > contours;
	vector< vector<Point> > prunedContours;
	Mat img_tmp;
	int maxArea = 10;
	src.copyTo(img_tmp);

	findContours(img_tmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);  //寻找轮廓

	for (size_t i = 0; i < contours.size(); i++)
	{
		if (i >= contours.size() || i < 0) { cout << "vetcor下标越界" << endl; break; }
		if (contourArea(contours[i]) < maxArea)
		{
			prunedContours.push_back(contours[i]);
		}
	}
	for (size_t i = 0; i < prunedContours.size(); i++)
	{
		if (i >= prunedContours.size() || i < 0) { cout << "vetcor下标越界" << endl; break; }
		drawContours(src, prunedContours, i, 0, 1, 8);
	}										//删除面积过小的轮廓

	cout << prunedContours.size() << endl;
}