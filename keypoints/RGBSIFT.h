/* RGB SIFT generates descriptors by stacking SIFT descriptors for the R, G, and B channels. */
#ifndef __OPENCV_RGB_SIFT_H__
#define __OPENCV_RGB_SIFT_H__

#include "DSPSIFT.h"
using namespace std;
using namespace cv;
#ifdef __cplusplus

namespace cv
{

	class CV_EXPORTS_W RGBSIFT : public DSPSIFT
	{
	public:
		CV_WRAP static Ptr<RGBSIFT> create()
		{
			return makePtr<RGBSIFT>(RGBSIFT());
		};
		CV_WRAP explicit RGBSIFT();
		
		//! returns the descriptor size in floats
		CV_WRAP int descriptorSize() const;
		
		//! finds the keypoints and computes descriptors for them using SIFT algorithm.
		//! Optionally it can compute descriptors for the user-provided keypoints
		virtual void operator()(InputArray img, InputArray mask,
			vector<KeyPoint>& keypoints,
			OutputArray descriptors,
			bool useProvidedKeypoints = false) const;

		virtual void operator()(InputArray img, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors,
			int numScales, double linePoint1, double linePoint2, bool useProvidedKeypoints = false) const;

	protected:
		virtual Mat createInitialColorImage(const Mat& img, bool doubleImageSize, float sigma) const;
		virtual void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const;
		virtual void normalizeHistogram(float *dst, int d, int n) const;
	};

} /* namespace cv */

#endif /* __cplusplus */

#endif

/* End of file. */
