#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint>& kPtsSource, std::vector<cv::KeyPoint>& kPtsRef, cv::Mat& descSource, cv::Mat& descRef,
	std::vector<cv::DMatch>& matches, std::string descriptorType, std::string matcherType, std::string selectorType, bool verbose)
{
	// configure matcher
	bool crossCheck = false;
	cv::Ptr<cv::DescriptorMatcher> matcher;

	if (matcherType.compare("MAT_BF") == 0)
	{
		//int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
		int normType = cv::NORM_HAMMING;
		if(descriptorType.compare("DES_HOG"))
			normType = cv::NORM_L2;

		matcher = cv::BFMatcher::create(normType, crossCheck);
	}

	else if (matcherType.compare("MAT_FLANN") == 0)
	{
		if (descSource.type() != CV_32F)
		{ // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
			descSource.convertTo(descSource, CV_32F);
			descRef.convertTo(descRef, CV_32F);
		}
		matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	}

	// perform matching task
	if (selectorType.compare("SEL_NN") == 0)
	{ // nearest neighbor (best match)
		matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
	}

	else if (selectorType.compare("SEL_KNN") == 0)
	{ // k nearest neighbors (k=2)
		vector<vector<cv::DMatch>> knn_matches;
		double t = (double)cv::getTickCount();
		matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		if (verbose)
			cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
		double minDescDistRatio = 0.8;
		for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
		{
			if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
			{
				matches.push_back((*it)[0]);
			}
		}
	}
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint>& keypoints, cv::Mat& img, cv::Mat& descriptors, string descriptorType, float& duration,bool verbose)
{
	// select appropriate descriptor
	cv::Ptr<cv::DescriptorExtractor> extractor;
	if (descriptorType.compare("BRISK") == 0)
	{
		int threshold = 30;        // FAST/AGAST detection threshold score.
		int octaves = 3;           // detection octaves (use 0 to do single scale)
		float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
		extractor = cv::BRISK::create(threshold, octaves, patternScale);
	}
	else if (descriptorType.compare("BRIEF") == 0) {
		bool    orientation = true; // use orientation or not
		int     bytes = 32; //length of description in bytes: 16,23,64
		extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, orientation);
	}
	else if (descriptorType.compare("ORB") == 0) {
		int 	nfeatures = 100;
		float 	scaleFactor = 1.2f;
		int 	nlevels = 8;
		int 	edgeThreshold = 31;
		int 	firstLevel = 0;
		int 	WTA_K = 2;
		cv::ORB::ScoreType 	scoreType = cv::ORB::HARRIS_SCORE;
		int 	patchSize = 31;
		int 	fastThreshold = 20;
		extractor = cv::ORB::create(nfeatures, scaleFactor);
	}
	else if (descriptorType.compare("FREAK") == 0) {
		bool 	orientationNormalized = true;
		bool 	scaleNormalized = true;
		float 	patternScale = 22.0f;
		int 	nOctaves = 4;
		extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
	}
	else if (descriptorType.compare("AKAZE") == 0) {
		cv::AKAZE::DescriptorType 	descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
		int 	descriptor_size = 0;
		int 	descriptor_channels = 3;
		float 	threshold = 0.001f;
		int 	nOctaves = 4;
		int 	nOctaveLayers = 4;
		//extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves);
		extractor = cv::AKAZE::create();

	}
	else if (descriptorType.compare("SIFT") == 0) {
		int 	nfeatures = 0;
		int 	nOctaveLayers = 3;
		double 	contrastThreshold = 0.04;
		double 	edgeThreshold = 10;
		double 	sigma = 1.6;
		//extractor = cv::xfeatures2d::SiftDescriptorExtractor::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
		extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
		//extractor = cv::xfeatures2d::SIFT::create();

	}

	// perform feature description
	double t = (double)cv::getTickCount();
	extractor->compute(img, keypoints, descriptors);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	duration = 1000 * t / 1.0;
	if (verbose)
		cout << descriptorType << " descriptor extraction in " << duration << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint>& keypoints, cv::Mat& img, float& duration, bool bVis, bool verbose)
{
	// compute detector parameters based on image size
	int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
	double maxOverlap = 0.0; // max. permissible overlap between two features in %
	double minDistance = (1.0 - maxOverlap) * blockSize;
	int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

	double qualityLevel = 0.01; // minimal accepted quality of image corners
	double k = 0.04;

	// Apply corner detection
	double t = (double)cv::getTickCount();
	vector<cv::Point2f> corners;
	cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

	// add corners to result vector
	for (auto it = corners.begin(); it != corners.end(); ++it)
	{

		cv::KeyPoint newKeyPoint;
		newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
		newKeyPoint.size = blockSize;
		keypoints.push_back(newKeyPoint);
	}
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	duration = 1000 * t / 1.0;
	if(verbose)
		cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << duration << " ms" << endl;

	// visualize results
	if (bVis)
	{
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = "Shi-Tomasi Corner Detector Results";
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
	}
}

void detKeypointsModern(vector<cv::KeyPoint>& keypoints, cv::Mat& img, std::string detectorType, float& duration, bool bVis, bool verbose)
{
	// compute detector parameters based on image size  
	cv::Ptr<cv::FeatureDetector> detector;
	if (detectorType.compare("BRISK") == 0) {
		detector = cv::BRISK::create();
	}
	else if (detectorType.compare("AKAZE") == 0) {
		detector = cv::AKAZE::create();
	}
	else if (detectorType.compare("FAST")) {
		detector = cv::FastFeatureDetector::create();
	}
	else if (detectorType.compare("ORB")) {
		detector = cv::ORB::create();
	}
	else if (detectorType.compare("SIFT")) {
		detector = cv::xfeatures2d::SIFT::create();
	}


	double t = (double)cv::getTickCount();
	detector->detect(img, keypoints);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	duration = 1000 * t / 1.0;
	if(verbose)
		cout << detectorType << " detector with n= " << keypoints.size() << " keypoints in " << duration << " ms" << endl;
	// visualize results
	if (bVis)
	{
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = detectorType + " Detector Results";
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
	}
}

void detKeypointsHARRIS(vector<cv::KeyPoint>& keypoints, cv::Mat& img, float& duration, bool bVis, bool verbose)
{
	// Detector parameters
	int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
	int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
	int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
	double k = 0.04;       // Harris parameter (see equation for details)

	// Detect Harris corners and normalize output
	cv::Mat dst, dst_norm, dst_norm_scaled;
	dst = cv::Mat::zeros(img.size(), CV_32FC1);
	double t = (double)cv::getTickCount();
	cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
	cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

	double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
	for (size_t j = 0; j < dst_norm.rows; j++)
	{
		for (size_t i = 0; i < dst_norm.cols; i++)
		{
			int response = (int)dst_norm.at<float>(j, i);
			if (response > minResponse)
			{ // only store points above a threshold
				cv::KeyPoint newKeyPoint;
				newKeyPoint.pt = cv::Point2f(i, j);
				newKeyPoint.size = 2 * apertureSize;
				newKeyPoint.response = response;
				// perform non-maximum suppression (NMS) in local neighbourhood around new key point
				bool bOverlap = false;
				for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
				{
					double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
					if (kptOverlap > maxOverlap)
					{
						bOverlap = true;
						if (newKeyPoint.response > (*it).response)
						{                      // if overlap is >t AND response is higher for new kpt
							*it = newKeyPoint; // replace old key point with new one
							break;             // quit loop over keypoints
						}
					}
				}
				if (!bOverlap)
				{                                     // only add new key point if no overlap has been found in previous NMS
					keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
				}
			}
		} // eof loop over cols
	}     // eof loop over rows
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	duration = 1000 * t / 1.0;
	if(verbose)
		cout << "HARRIS detector with n= " << keypoints.size() << " keypoints in " << duration << " ms" << endl;

	if (bVis)
	{
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = "HARRIS Detector Results";
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
	}

}