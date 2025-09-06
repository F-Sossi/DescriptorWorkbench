//-------------------------------------------------------------------------
// Name: HoWH.h
// Author: Clark Olson, Siqi Zhang
// Description: HoWH descriptor class, inherited from VanillaSIFT
//  HoWH uses hue value weighted by saturation, similar to SIFT
// Methods:
//			create()
//			HoWH()
//			operator()
//			createInitialColorImage()
//			calcSIFTDescriptor()
//-------------------------------------------------------------------------

#ifndef __OPENCV_HoWH_H__
#define __OPENCV_HoWH_H__

#include "DSPSIFT.h"
using namespace std;
using namespace cv;

#ifdef __cplusplus

/*!
HoWH implementation.
*/
class CV_EXPORTS_W HoWH : public DSPSIFT
{
public:
//------------------------------------create()-----------------------------------------
// create a pointer to the HoWH object
//Precondition: None
//Postcondition: the pointer is created
//-------------------------------------------------------------------------------------
	CV_WRAP static Ptr<HoWH> create() 
	{
		return makePtr<HoWH>(HoWH());
	};

	CV_WRAP explicit HoWH();

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
	virtual void operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints, OutputArray _descriptors, bool useProvidedKeypoints) const;

	virtual void operator()(InputArray img, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors,
		int numScales, double linePoint1, double linePoint2, bool useProvidedKeypoints = false) const;

	CV_WRAP virtual int descriptorSize() const;

protected:
//------------------------------------createInitialColorImage()------------------------
//create initial base image for later process
//Precondition: the following parameters must be correctly defined.
//parameters:
	//img: color image
	//doubleImageSize: bool indicating whether doubling the image 
	//sigma: gaussian blur coefficient
//Postcondition: 1. keypoints are assigned
//				 2. descriptors are calculated and assigned
//-------------------------------------------------------------------------------------
	virtual Mat createInitialColorImage(const Mat& img, bool doubleImageSize, float sigma) const;

//------------------------------------calcSIFTDescriptor-------------------------------
//calculate HoWH descriptor with given information and assign descriptor to dst
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
};

#endif /* __cplusplus */
#endif