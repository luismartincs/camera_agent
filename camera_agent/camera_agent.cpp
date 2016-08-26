/**
 * Simple shape detector program.
 * It loads an image and tries to find simple shapes (rectangle, triangle, circle, etc) in it.
 * This program is a modified version of `squares.cpp` found in the OpenCV sample dir.
 */
 /*
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>

using namespace std;
using namespace cv;

int isIPCam = 0;

static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	double scale = 0.4;
	int thickness = 1;
	int baseline = 0;

	cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
	cv::Rect r = cv::boundingRect(contour);

	cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
	cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
	cv::putText(im, label, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}

int main()
{

	IplImage* img; 
	Mat src;

	CvCapture* capture;
	VideoCapture vcap;

	//Se define la IP de la camara remota
	
	if(isIPCam) {

		const std::string videoStreamAddress = "http://192.168.15.90:8080/videofeed?dummy=param.mjpg";
		std::cout << "Conectando..." << std::endl;
		if(!vcap.open(videoStreamAddress)) {
			std::cout << "Error opening video stream or file" << std::endl;
			return -1;
		}
		std::cout << "¡Conexion exitosa!" << std::endl;
	} else {
		capture = cvCaptureFromCAM(0);
	}

	while(1){

		if(isIPCam) {
					if(!vcap.read(src)) {
						std::cout << "No frame" << std::endl;
						cv::waitKey();
					}
		} else {
			img = cvQueryFrame(capture);
			src = cv::cvarrToMat(img);

		}

		Mat gray;
		cvtColor(src, gray, CV_BGR2GRAY);

		Mat bw;
		Canny(gray, bw, 0, 50, 5);
		cv::imshow("dst", bw);

		std::vector<std::vector<cv::Point> > contours;
		cv::findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);


		std::vector<cv::Point> approx;
		cv::Mat dst = src.clone();

		for (int i = 0; i < contours.size(); i++)
		{
			// Approximate contour with accuracy proportional
			// to the contour perimeter
			cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);

			// Skip small or non-convex objects 
			if (std::fabs(cv::contourArea(contours[i])) < 100 || !cv::isContourConvex(approx))
				continue;

			if (approx.size() == 3)
			{
				//setLabel(dst, "TRI", contours[i]);    // Triangles
			}
			else if (approx.size() >= 4 && approx.size() <= 6)
			{
				// Number of vertices of polygonal curve
				int vtc = approx.size();

				// Get the cosines of all corners
				std::vector<double> cos;
				for (int j = 2; j < vtc+1; j++)
					cos.push_back(angle(approx[j%vtc], approx[j-2], approx[j-1]));

				// Sort ascending the cosine values
				std::sort(cos.begin(), cos.end());

				// Get the lowest and the highest cosine
				double mincos = cos.front();
				double maxcos = cos.back();

				// Use the degrees obtained above and the number of vertices
				// to determine the shape of the contour
				if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
					setLabel(dst, "RECT", contours[i]);
				else if (vtc == 5 && mincos >= -0.34 && maxcos <= -0.27)
					setLabel(dst, "PENTA", contours[i]);
				else if (vtc == 6 && mincos >= -0.55 && maxcos <= -0.45)
					setLabel(dst, "HEXA", contours[i]);
			}
			else
			{
				// Detect and label circles
				
				double area = cv::contourArea(contours[i]);
				cv::Rect r = cv::boundingRect(contours[i]);
				int radius = r.width / 2;

				if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
				    std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
					setLabel(dst, "CIR", contours[i]);
			}

		}

		//cv::imshow("src", src);
		cv::imshow("dst", dst);
		cvWaitKey(100);

	}

	return 0;
}*/


#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cameraparams.h"
#include "patterndetector.h"

using namespace std;
using namespace cv;
using namespace ARma;

#define PAT_SIZE 64//equal to pattern_size variable (see below)
#define SAVE_VIDEO 0 //if true, it saves the video in "output.avi"
#define NUM_OF_PATTERNS 4// define the number of patterns you want to use
#define isIPCam false
#define RES_WIDTH 640
#define RES_HEIGHT 480
#define ROWS 5
#define COLS 9

//1920*1080

string filename1="../patterns/pattern1.png";//id=1
string filename2="../patterns/pattern2.png";//id=2
string filename3="../patterns/pattern3.png";//id=3
string filename4="../patterns/pattern10.png";//id=3

struct MapCell{
    int ix;
    int iy;
    float x;
    float y;
    float width;
    float height;
};

unsigned int pixelSizeW = RES_WIDTH/COLS;
unsigned int pixelSizeH = RES_HEIGHT/ROWS;

long double prevTime = time(0) * 1000;


static int loadPattern(const string , std::vector<cv::Mat>& , int& );

int main(int argc, char** argv){

    std::ofstream file;

	std::vector<cv::Mat> patternLibrary;
	std::vector<Pattern> detectedPattern;

	//Inicializar celdas del mapa

	std::vector<MapCell> mapCells;
	CvScalar color = cvScalar(255,255,255);

	for(int i=0; i < ROWS;i++){
                for (int j=0; j < COLS; j++) {
                    MapCell mCell;
                    mCell.ix = j;
                    mCell.iy = i;
                    mCell.x = j*pixelSizeW;
                    mCell.y = i*pixelSizeH;
                    mCell.width = pixelSizeW;
                    mCell.height = pixelSizeH;
                    mapCells.push_back(mCell);
        }
    }

    long double prevTime = time(0) * 1000;

	int patternCount=0;

	
	//create patterns' library using rotated versions of patterns 


	loadPattern(filename1, patternLibrary, patternCount);
	loadPattern(filename2, patternLibrary, patternCount);
	loadPattern(filename3, patternLibrary, patternCount);
	loadPattern(filename4, patternLibrary, patternCount);

	cout << patternCount << " patterns are loaded." << endl;
	

	int norm_pattern_size = PAT_SIZE;
	double fixed_thresh = 40;
	double adapt_thresh = 5;//non-used with FIXED_THRESHOLD mode
	int adapt_block_size = 45;//non-used with FIXED_THRESHOLD mode
	double confidenceThreshold = 0.35;
	int mode = 2;//1:FIXED_THRESHOLD, 2: ADAPTIVE_THRESHOLD

	PatternDetector myDetector( fixed_thresh, adapt_thresh, adapt_block_size, confidenceThreshold, norm_pattern_size, mode);

	CvCapture* capture; //= cvCaptureFromCAM(0);

	cv::VideoCapture vcap;

//Se define la IP de la camara remota
	if(isIPCam) {

		const std::string videoStreamAddress = "http://192.168.15.90:8080/videofeed?dummy=param.mjpg";

		//open the video stream and make sure it's opened
		std::cout << "Conectando..." << std::endl;
		if(!vcap.open(videoStreamAddress)) {
			std::cout << "Error opening video stream or file" << std::endl;
			return -1;
		}
		std::cout << "¡Conexion exitosa!" << std::endl;
	} else {

		capture = cvCaptureFromCAM(0);
		
		
		//cvSetCaptureProperty(capture,CV_CAP_PROP_FOURCC,CV_FOURCC('M','J','P','G'));
		//cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH, 1280 );
		//cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT, 768 );
		
	}

#if (SAVE_VIDEO)
	CvVideoWriter *video_writer = cvCreateVideoWriter( "output.avi", -1, 25, cvSize(640,480) );
#endif

	Mat imgMat;

	while(1){ //modify it for longer/shorter videos
		
				IplImage* img; // = cvQueryFrame(capture);
				Mat imgMat; // = cv::cvarrToMat(img);

				if(isIPCam) {
					if(!vcap.read(imgMat)) {
						std::cout << "No frame" << std::endl;
						cv::waitKey();
					}
				} else {
					img = cvQueryFrame(capture);
					imgMat = cv::cvarrToMat(img);

				}


				//run the detector
 				myDetector.detect(imgMat, cameraMatrix, distortions, patternLibrary, detectedPattern); 


				//Establecer las posiciones de los agentes

				for (unsigned int i =0; i<detectedPattern.size(); i++){
		            Pattern *patternDetected = &detectedPattern.at(i);
		            
		            for (int i=0; i < mapCells.size(); i++) {
		                MapCell mc = mapCells.at(i);
		                
		                if ((patternDetected->center.x > mc.x && patternDetected->center.x < (mc.x + mc.width)) && (patternDetected->center.y > mc.y && patternDetected->center.y < (mc.y + mc.height))) {
		                    patternDetected->ix = mc.ix;
		                    patternDetected->iy = mc.iy;
		                    continue;
		                }
		            }
		            
		        }


				//Dibujar patrones
				for (unsigned int i =0; i<detectedPattern.size(); i++){
					detectedPattern.at(i).draw( imgMat, cameraMatrix, distortions);
				}


				//Dibujar grid e informacion

				for (int i=0; i < mapCells.size(); i++) {
		            MapCell mc = mapCells.at(i);
		            cv::rectangle(imgMat,cvPoint(mc.x,mc.y),cvPoint(mc.x+mc.width,mc.y+mc.height),color);
		        }

				putText(imgMat, "N", Point2f(RES_WIDTH/2,50), FONT_HERSHEY_PLAIN, 1,  Scalar(0,255,0),1);
		        putText(imgMat, "S", Point2f(RES_WIDTH/2,RES_HEIGHT-50), FONT_HERSHEY_PLAIN, 1,  Scalar(0,255,0),1);
		        putText(imgMat, "E", Point2f(RES_WIDTH-50,RES_HEIGHT/2), FONT_HERSHEY_PLAIN, 1,  Scalar(0,255,0),1);
		        putText(imgMat, "O", Point2f(50,RES_HEIGHT/2), FONT_HERSHEY_PLAIN, 1,  Scalar(0,255,0),1);
		        

		        long double currentTime = time(0)*1000;
        
			        if((currentTime - prevTime) >= 5000){
			            prevTime = currentTime;
			            
			            file.open ("../log/agents.txt");

			            for (unsigned int i =0; i<detectedPattern.size(); i++){
			                Pattern *patternDetected = &detectedPattern.at(i);
			                
			                file<<""<< patternDetected->id<<","<<patternDetected->ix<<","<<patternDetected->iy<<","<<patternDetected->center.x<<","<<patternDetected->center.y<<endl;
			                
			                std::cout<<"{"<< patternDetected->id<<","<<patternDetected->ix<<","<<patternDetected->iy<<"}"<<endl;
			                file.flush();
		                
		            	}
		        	}
        
           			file.close();


		#if (SAVE_VIDEO)
				cvWriteFrame(video_writer, &((IplImage) imgMat));
		#endif
				imshow("Camera Agent", imgMat);
				
				cvWaitKey(100);

				detectedPattern.clear();

		//}
	
	} // while

#if (SAVE_VIDEO)
	cvReleaseVideoWriter(&video_writer);
#endif
	cvReleaseCapture(&capture);

	return 0;

}

int loadPattern(string filename, std::vector<cv::Mat>& library, int& patternCount){
	Mat img = imread(filename,0);
	
	if(img.cols!=img.rows){
		return -1;
		printf("Not a square pattern");
	}

	int msize = PAT_SIZE; 

	Mat src(msize, msize, CV_8UC1);
	Point2f center((msize-1)/2.0f,(msize-1)/2.0f);
	Mat rot_mat(2,3,CV_32F);
	
	resize(img, src, Size(msize,msize));
	Mat subImg = src(Range(msize/4,3*msize/4), Range(msize/4,3*msize/4));
	library.push_back(subImg);

	rot_mat = getRotationMatrix2D( center, 90, 1.0);

	for (int i=1; i<4; i++){
		Mat dst= Mat(msize, msize, CV_8UC1);
		rot_mat = getRotationMatrix2D( center, -i*90, 1.0);
		warpAffine( src, dst , rot_mat, Size(msize,msize));
		Mat subImg = dst(Range(msize/4,3*msize/4), Range(msize/4,3*msize/4));
		library.push_back(subImg);	
	}

	patternCount++;
	return 1;
}
