#include "pattern.h"
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

namespace ARma {

	Pattern::Pattern(double param1){
		id =-1;
		size = param1;
		orientation = -1;
		confidence = -1;

		rotVec = (Mat_<float>(3,1) << 0, 0, 0);
		transVec = (Mat_<float>(3,1) << 0, 0, 0);
		rotMat = Mat::eye(3, 3, CV_32F);

	}

	//convert rotation vector to rotation matrix (if you want to proceed with other libraries)
	void Pattern::rotationMatrix(const Mat& rotation_vector, Mat& rotation_matrix)
	{
		Rodrigues(rotation_vector, rotation_matrix);		
	}

	void Pattern::showPattern()
	{
		cout << "Pattern ID: " << id << endl;
		cout << "Pattern Size: " << size << endl;
		cout << "Pattern Confedince Value: " << confidence << endl;
		cout << "Pattern Orientation: " << orientation << endl;
		
		rotationMatrix(rotVec, rotMat);
		
		cout << "Exterior Matrix (from pattern to camera): " << endl;
		for (int i = 0; i<3; i++){
		cout << rotMat.at<float>(i,0) << "\t" << rotMat.at<float>(i,1) << "\t" << rotMat.at<float>(i,2) << " |\t"<< transVec.at<float>(i,0) << endl;
		}
		
	}

	void Pattern::getExtrinsics(int patternSize, const Mat& cameraMatrix, const Mat& distortions)
	{

		CvMat objectPts;//header for 3D points of pat3Dpts
		CvMat imagePts;//header for 2D image points of pat2Dpts 
		CvMat intrinsics = cameraMatrix;
		CvMat distCoeff = distortions;
		CvMat rot = rotVec;
		CvMat tra = transVec;
//		CvMat rotationMatrix = rotMat; // projectionMatrix = [rotMat tra];

		CvPoint2D32f pat2DPts[4];
		for (int i = 0; i<4; i++){
			pat2DPts[i].x = this->vertices.at(i).x;
			pat2DPts[i].y = this->vertices.at(i).y;
		}

		//3D points in pattern coordinate system
		CvPoint3D32f pat3DPts[4];
		pat3DPts[0].x = 0.0;
		pat3DPts[0].y = 0.0;
		pat3DPts[0].z = 0.0;
		pat3DPts[1].x = patternSize;
		pat3DPts[1].y = 0.0;
		pat3DPts[1].z = 0.0;
		pat3DPts[2].x = patternSize;
		pat3DPts[2].y = patternSize;
		pat3DPts[2].z = 0.0;
		pat3DPts[3].x = 0.0;
		pat3DPts[3].y = patternSize;
		pat3DPts[3].z = 0.0;

		cvInitMatHeader(&objectPts, 4, 3, CV_32FC1, pat3DPts);
		cvInitMatHeader(&imagePts, 4, 2, CV_32FC1, pat2DPts);
		
		//find extrinsic parameters
		cvFindExtrinsicCameraParams2(&objectPts, &imagePts, &intrinsics, &distCoeff, &rot, &tra);
	}

	void Pattern::draw(Mat& frame, const Mat& camMatrix, const Mat& distMatrix)
	{

		CvScalar color = cvScalar(255,255,255);
		
		switch (id){
			case 1:
				 color = cvScalar(255,0,255);
				break;
			case 2:
				 color = cvScalar(255,255,0);
				break;
			case 3:
				 color = cvScalar(0,255,255);
				break;
		}

		//model 3D points: they must be projected to the image plane
		Mat modelPts = (Mat_<float>(8,3) << 0, 0, 0, size, 0, 0, size, size, 0, 0, size, 0,
			0, 0, -size, size, 0, -size, size, size, -size, 0, size, -size );


		std::vector<cv::Point2f> model2ImagePts;

		projectPoints(modelPts, rotVec, transVec, camMatrix, distMatrix, model2ImagePts); 

		int i;

		for(i = 0; i < 4; i++){
			cv::line(frame, model2ImagePts.at(i%4), model2ImagePts.at((i+1)%4), cvScalar(0,255,0), 3);
		}
		
		 cv::circle(frame, this->center, 10, Scalar(0,255,0),1 );
		 
		cv:putText(frame, "["+to_string(this->ix)+","+to_string(this->iy)+","+this->orientationStr+"]",this->center,FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,0),2);
        
		cv::line(frame, model2ImagePts.at(2), model2ImagePts.at(3), cvScalar(0,0,255), 3);

		model2ImagePts.clear();

	}

}