/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char* argv[])
{

	/* INIT VARIABLES AND DATA STRUCTURES */

	// data location
	string dataPath = "../";

	// camera
	string imgBasePath = dataPath + "images/";
	string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
	string imgFileType = ".png";
	int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
	int imgEndIndex = 9;   // last file index to load
	int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

	// misc
	int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
	bool bVis = false;            // visualize results
	bool eval_mode = true;        //if true an evaluation loop is added in the end
	bool verbose = false;		  //prints messages to the console	
	/* MAIN LOOP OVER ALL IMAGES */

	if (!eval_mode) {
		vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
		for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
		{
			/* LOAD IMAGE INTO BUFFER */
			// assemble filenames for current index
			ostringstream imgNumber;
			imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
			string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;
			if(verbose)
				cout << imgFullFilename << endl;

			// load image from file and convert to grayscale
			cv::Mat img, imgGray;
			img = cv::imread(imgFullFilename);
			cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

			//// STUDENT ASSIGNMENT
			//// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

			// push image into data frame buffer
			DataFrame frame;
			frame.cameraImg = imgGray;
			dataBuffer.push_back(frame);
			if (dataBuffer.size() > 2) dataBuffer.erase(dataBuffer.begin());
			if(verbose)
				cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
			/* DETECT IMAGE KEYPOINTS */
			// extract 2D keypoints from current image
			vector<cv::KeyPoint> keypoints; // create empty feature list for current image
			string detectorType = "SHITOMASI";
			//"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"

			//// STUDENT ASSIGNMENT
			//// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
			//// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
			float duration;
			if (detectorType.compare("SHITOMASI") == 0) {
				detKeypointsShiTomasi(keypoints, imgGray, duration, false,verbose);
			}
			else if (detectorType.compare("HARRIS") == 0) {
				detKeypointsHARRIS(keypoints, imgGray,duration, false, verbose);
			}
			else {
				detKeypointsModern(keypoints, imgGray, detectorType,duration, false, verbose);
			}
			//// TASK MP.3 -> only keep keypoints on the preceding vehicle
			bool bFocusOnVehicle = true;
			cv::Rect vehicleRect(535, 180, 180, 150);
			if (bFocusOnVehicle)
			{
				vector<cv::KeyPoint> newKeypoints;
				float w = vehicleRect.width;
				float h = vehicleRect.height;
				float x = vehicleRect.x;
				float y = vehicleRect.y;
				for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
					float x0 = (*it).pt.x;
					float y0 = (*it).pt.y;
					if (x0 >= x && x0 < (x + w) && y0 >= y && y0 < (y + h)) {
						newKeypoints.push_back(*it);
					}
				}
				keypoints = newKeypoints;
			}
			// optional : limit number of keypoints (helpful for debugging and learning)
			bool bLimitKpts = false;
			if (bLimitKpts)
			{
				int maxKeypoints = 20;
				if (detectorType.compare("SHITOMASI") == 0)
				{ // there is no response info, so keep the first 50 as they are sorted in descending quality order
					keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
				}
				cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
				if(verbose)
					cout << " NOTE: Keypoints have been limited!" << endl;
			}
			// push keypoints and descriptor for current frame to end of data buffer
			(dataBuffer.end() - 1)->keypoints = keypoints;
			if(verbose)
				cout << "#2 : DETECT KEYPOINTS done" << endl;

			/* EXTRACT KEYPOINT DESCRIPTORS */
			//// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
			//// -> BRIEF, ORB, FREAK, AKAZE, SIFT
			cv::Mat descriptors;
			//"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"
			string descriptorName = "BRIEF"; // BRIEF, ORB, FREAK, AKAZE, SIFT

			descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorName,duration,verbose);
			// push descriptors for current frame to end of data buffer
			(dataBuffer.end() - 1)->descriptors = descriptors;
			if(verbose)
				cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

			if (dataBuffer.size() > 1) // wait until at least two images have been processed
			{
				/* MATCH KEYPOINT DESCRIPTORS */
				vector<cv::DMatch> matches;
				string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
				string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
				if (descriptorName.compare("SIFT") == 0) descriptorType == "DES_HOG";
				string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

				//// STUDENT ASSIGNMENT
				//// TASK MP.5 -> add FLANN matching in file matching2D.cpp
				//// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
				matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
					(dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
					matches, descriptorType, matcherType, selectorType,verbose);
				//// EOF STUDENT ASSIGNMENT
				// store matches in current data frame
				(dataBuffer.end() - 1)->kptMatches = matches;
				cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

				// visualize matches between current and previous image
				bVis = true;
				if (bVis)
				{
					cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
					cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
						(dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
						matches, matchImg,
						cv::Scalar::all(-1), cv::Scalar::all(-1),
						vector<char>(), cv::DrawMatchesFlags::DEFAULT);
					string windowName = "Matching keypoints between two camera images";
					cv::namedWindow(windowName, cv::WINDOW_FREERATIO);
					cv::imshow(windowName, matchImg);
					cout << "Press key to continue to next image" << endl;
					cv::waitKey(0); // wait for key to be pressed
				}
				bVis = false;
			}

		} // eof loop over all images
	} //eof non eval mode

	else { //evaluation mode loop
		cout << "eval mode" << endl;

		vector<string> detectorTypes = {"SHITOMASI", "HARRIS","FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
		vector<string> descriptorNames = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE","SIFT"};
		float kptsPerFramePerDetector[7][10][2]; //task 7 array: 7 detectors x 10 images x 2 kpis (number of keypts+avg neigh. size)
		float matchedKptsPerFramePerDescr[7][6][9]; //task 8 array: 7 detectors x 6 descriptors x 9 images transitions
		float computeDurationDetector[7][10]; //task 9.1 array: 7 detectors x 10 images
		float computeDurationDescriptor[6][10]; //task 9.2 array: 6 descriptors x 10 images

		for (int i = 0; i < descriptorNames.size(); i++) {
			for (int j = 0; j < detectorTypes.size(); j++) {
				string detectorType = detectorTypes[j];
				string descriptorName = descriptorNames[i];
				vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
				cout << "i= " << i <<" "<< descriptorName << " j= " << j <<" "<< detectorType << endl;
				if (detectorType.compare("AKAZE") != 0 && descriptorName.compare("AKAZE") == 0) {
					//AKAZE descriptor only works with AKAZE/KAZE detector
					continue;
				}
				//if (i == 1 && j == 0)
				//	verbose = true;
				for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++) {

					//0. LOAD IMAGE INTO BUFFER
					ostringstream imgNumber;
					imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
					string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;
					if(verbose)
						cout << imgFullFilename << endl;

					// load image from file and convert to grayscale
					cv::Mat img, imgGray;
					img = cv::imread(imgFullFilename);
					cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
					DataFrame frame;
					frame.cameraImg = imgGray;
					dataBuffer.push_back(frame);
					if (dataBuffer.size() > 2) dataBuffer.erase(dataBuffer.begin());
					if(verbose)
						cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

					//1. Detect keypoints
					vector<cv::KeyPoint> keypoints;
					float duration;
					if (detectorType.compare("SHITOMASI") == 0) {
						detKeypointsShiTomasi(keypoints, imgGray, duration, false,verbose);
					}
					else if (detectorType.compare("HARRIS") == 0) {
						detKeypointsHARRIS(keypoints, imgGray, duration, false, verbose);
					}
					else {
						detKeypointsModern(keypoints, imgGray, detectorType, duration, false, verbose);
					}
					//2. Crop keypoints outside ROI
					cv::Rect vehicleRect(535, 180, 180, 150);
					vector<cv::KeyPoint> newKeypoints;
					float w = vehicleRect.width;
					float h = vehicleRect.height;
					float x = vehicleRect.x;
					float y = vehicleRect.y;
					for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
						float x0 = (*it).pt.x;
						float y0 = (*it).pt.y;
						if (x0 >= x && x0 < (x + w) && y0 >= y && y0 < (y + h)) {
							newKeypoints.push_back(*it);
						}
					}
					keypoints = newKeypoints;
					(dataBuffer.end() - 1)->keypoints = keypoints;
					
					//Task 7 & task 9.1: record number of keypoints + average neighboring size + detection duration
					if (i == 0) { 
						kptsPerFramePerDetector[j][imgIndex][0] = keypoints.size();
						float avgNeighSize = 0;
						for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
						{
							avgNeighSize += (*it).response;
						}
						avgNeighSize /= keypoints.size();
						kptsPerFramePerDetector[j][imgIndex][1] = avgNeighSize;
						computeDurationDetector[j][imgIndex] = duration;
					}
					if(verbose)
						cout << "#2 : DETECT KEYPOINTS done" << endl;

					//3. Describe keypoints
					cv::Mat descriptors;
					//float duration;
					descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorName,duration,verbose);
					//task 9.2 record description time
					if (j == 0)
						computeDurationDescriptor[i][imgIndex] = duration;
					if(descriptorName.compare("AKAZE")==0 && detectorType.compare("AKAZE") == 0)
						computeDurationDescriptor[i][imgIndex] = duration;
					// push descriptors for current frame to end of data buffer
					(dataBuffer.end() - 1)->descriptors = descriptors;
					if(verbose)
						cout << "#3 : EXTRACT DESCRIPTORS done" << endl;
					if (dataBuffer.size() > 1) // wait until at least two images have been processed
					{
						// 4. Match keypoints
						vector<cv::DMatch> matches;
						string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
						string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
						if (descriptorName.compare("SIFT") == 0) descriptorType == "DES_HOG";
						string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
						matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
							(dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
							matches, descriptorType, matcherType, selectorType,verbose);
						(dataBuffer.end() - 1)->kptMatches = matches;
						if(verbose)
							cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;
						//task 8 number of matched keypoints
						matchedKptsPerFramePerDescr[j][i][imgIndex - 1] = matches.size();
					}
				} // eof loop over all images
			} //eof loop over descriptor
		} //eof loop over detector

		//task 9.1: duration of detection per image & detector
		ofstream myfile("task91.txt");
		cout << "task 91" << endl;
		for (int k = 0; k < 7; k++) {
			for (int ii = 0; ii < 10; ii++) {
				cout<< computeDurationDetector[k][ii] << " ";
				myfile << computeDurationDetector[k][ii] << " ";
			}
			cout << endl;
			myfile << endl;
		}
		myfile.close();

		//task 9.2: duration of description per image & descriptor
		myfile.open("task92.txt");
		cout << "task 92" << endl;
		for (int k = 0; k < 6; k++) {
			for (int ii = 0; ii < 10; ii++) {
				cout<< computeDurationDescriptor[k][ii] << " ";
				myfile << computeDurationDescriptor[k][ii] << " ";
			}
			cout << endl;
			myfile << endl;
		}		
		myfile.close();

		//task 7.1: number of keypts per image & detector
		myfile.open("task71.txt");
		cout << "task 71" << endl;
		for (int k = 0; k < 7; k++) {
			for (int ii = 0; ii < 10; ii++) {
				cout<< kptsPerFramePerDetector[k][ii][0] << " ";
				myfile << kptsPerFramePerDetector[k][ii][0] << " ";
			}
			cout << endl;
			myfile << endl;
		}
		myfile.close();

		//task 7.2: average neighboring size per image & detector
		myfile.open("task72.txt");
		cout << "task 72" << endl;
		for (int k = 0; k < 7; k++) {
			for (int ii = 0; ii < 10; ii++) {
				cout << kptsPerFramePerDetector[k][ii][1] << " ";
				myfile << kptsPerFramePerDetector[k][ii][1] << " ";
			}
			myfile << endl;
			cout<<endl;
		}
		myfile.close();

		//task 8: number of macthed keypoints per descriptor & detector & image transition
		for (int ii = 0; ii < 9; ii++) {
			string filename = "task8_transition" + to_string(ii) + ".txt";
			myfile.open(filename);
			for (int k = 0; k < 7; k++) {
				for (int jj = 0; jj < 6; jj++) {
					myfile << kptsPerFramePerDetector[k][jj][ii] << " ";
				}
				myfile << endl;
			}
			myfile.close();
		}

	}//eof eval mode
	return 0;
}
