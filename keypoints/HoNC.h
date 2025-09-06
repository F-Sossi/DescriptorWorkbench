//-------------------------------------------------------------------------
// Name: HoNC.h
// Author: Clark Olson, Siqi Zhang
// Description: HoNC descriptor class, inherited from HoWH
//	HoNC uses similar keypoint detection as SIFT but taking color
//  images into Dog. For keypoint descriptor, HoNC uses a 2x2x2 
//  RGB histogram.
// Methods:
//			create()
//			HoNC()
//			operator()
//			calcSIFTDescriptor()
//-------------------------------------------------------------------------


#ifndef __OPENCV_COLOR_HIST_SIFT_H__
#define __OPENCV_COLOR_HIST_SIFT_H__

#include "HoWH.h"

using namespace std;
using namespace cv;

#ifdef __cplusplus

/*!
	HoNC implementation.
*/
class CV_EXPORTS_W HoNC : public HoWH
{
public:

//------------------------------------create()-----------------------------------------
// create a pointer to the HoNC object
//Precondition: None
//Postcondition: the pointer is created
//-------------------------------------------------------------------------------------
	CV_WRAP static Ptr<HoNC> create() 
	{
		return makePtr<HoNC>(HoNC());
	};

	CV_WRAP explicit HoNC();

//------------------------------------operator()---------------------------------------
// Overloading operator() to run the algorithm using color image:
// 1. compute keypoints using local extrema of Dog space
// 2. compute descriptors with keypoints and color pyramid
//Precondition: the following parameters must be correctly defined.
//parameters:
	//_image: color image base
	//_mask: image mask
	//keypoints: keypoints of the image
	//_descriptors: descriptors
	//useProvidedKeypoints: bool indicating whether using provided keypoints
//Postcondition: 1. keypoints are assigned
//				 2. descriptors are calculated and assigned
//-------------------------------------------------------------------------------------
	virtual void operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints, OutputArray _descriptors, bool useProvidedKeypoints= false) const;

	virtual void operator()(InputArray img, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors,
		int numScales, double linePoint1, double linePoint2, bool useProvidedKeypoints = false) const;

protected:

//------------------------------------calcSIFTDescriptor-------------------------------
//calculate colorhistsift descriptor with given information and assign descriptor to dst
//Precondition: the following parameters must be correctly defined.
//parameters:
	//img: color image
	//ptf: keypoint
	//ori: angle(degree) of the keypoint relative to the coordinates, clockwise
	//scl: radius of meaningful neighborhood around the keypoint 
	//d: newsift descr_width, 4 in this case
	//n: newsift_descr_hist_bins, 8 in this case
	//dst: descriptor array to pass in
//Postcondition: dst array is assigned with decriptors
//-------------------------------------------------------------------------------------
	virtual void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const;

//------------------------------------findScaleSpaceExtrema()--------------------------
// Detects features at extrema in color DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
//Precondition: the following parameters must be correctly defined.
//parameters:
//gauss_pyr: gaussian pyramid
//dog_pyr: difference of Gaussian pyramid
//keypoints: empty keypoints vector
//Postcondition: keypoints are assigned
//-------------------------------------------------------------------------------------
	virtual void findScaleSpaceExtrema(const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
		std::vector<KeyPoint>& keypoints) const;

//------------------------------------adjustLocalExtrema()-----------------------------
// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
//Precondition: the following parameters must be correctly defined.
//parameters:
	//dog_pyr: difference of Gaussian
	//kpt: pixel location
	//octv: 
	//layer: 
	//r: row number
	//c: column number
	//nOctaveLayers: number of octave
	//contrastThreshold:
	//edgeThreshold:
	//sigma:
//Postcondition: 
//-------------------------------------------------------------------------------------
	static bool adjustLocalExtrema(const std::vector<Mat>& dog_pyr, KeyPoint& kpt, int octv, int& layer, int& r, int& c, int nOctaveLayers, float contrastThreshold, float edgeThreshold, float sigma);

//------------------------------------calcOrientationHist()----------------------------
// Computes a gradient orientation histogram at a specified pixel
//Precondition: the following parameters must be correctly defined.
//parameters:
	//img: color image
	//pt: pixel location
	//radius: histogram range
	//sigma: 
	//hist: orientation histogram
	//n: newsift_descr_hist_bins, 8 in this case
//Postcondition: orientation is voted to histogram
//-------------------------------------------------------------------------------------
	static float calcOrientationHist(const Mat& img, Point pt, int radius, float sigma, float* hist, int n);

//------------------------------------descriptorSize()---------------------------------
// ! returns the descriptor size in floats (128)
//Precondition: None
//Postcondition: the descriptor size is returned in floats (128)
//-------------------------------------------------------------------------------------
	CV_WRAP virtual int descriptorSize() const;
};


#endif /* __cplusplus */

#endif