/* DSPSIFT follows vanillia sift, but keypoints are studied over multiple domains, and results are averaged later. */
#ifndef __OPENCV_DSPSIFT_H__
#define __OPENCV_DSPSIFT_H__

#include "VanillaSIFT.h"

using namespace std;
using namespace cv;
//using namespace xfeatures2d;
using namespace hal;

#ifdef __cplusplus

/*!
DSPSIFT implementation.
*/

class CV_EXPORTS_W DSPSIFT : public VanillaSIFT {
public:

	CV_WRAP static Ptr<DSPSIFT> create() {
		return makePtr<DSPSIFT>(DSPSIFT());
	};

	CV_WRAP explicit DSPSIFT();

	using VanillaSIFT::operator();
	virtual void operator()(InputArray img, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors,
		int numScales, double linePoint1, double linePoint2, bool useProvidedKeypoints = false) const;

	//virtual void operator()(InputArray img, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints = false) const;

	using VanillaSIFT::compute;
	virtual void compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
		int numScales, double linePoint1, double linePoint2);

	//virtual void compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors);

protected:

	//virtual void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const;
	virtual void calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints, Mat& descriptors, int nOctaveLayers, int firstOctave) const;

	virtual void calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints, Mat& descriptors, int nOctaveLayers, int firstOctave, 
		int numScales, double linePoint1, double linePoint2) const;

	//virtual void operator()(InputArray img, InputArray mask, vector<KeyPoint>& keypoints,
		//int numScales, int linePoint1, int linePoint2) const;

	//virtual void compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors);


};

#endif /* __cplusplus */

#endif