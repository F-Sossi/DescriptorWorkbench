//-------------------------------------------------------------------------
// Name: HoWH.cpp
// Author: Clark Olson, Siqi Zhang
// Description: HoWH descriptor class, inherited from VanillaSIFT
//  HoWH uses hue value weighted by saturation, similar to SIFT
// Methods:
//			HoWH()
//			operator()
//			createInitialColorImage()
//			calcSIFTDescriptor()
//-------------------------------------------------------------------------

/**********************************************************************************************\
Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/SIFT/
Below is the original copyright.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

\**********************************************************************************************/

#include "HoWH.h"

// constructor
HoWH::HoWH()
{
	// do nothing
}

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
Mat HoWH::createInitialColorImage(const Mat& img, bool doubleImageSize, float sigma) const
{
	Mat colorImg = img, color_fpt;
	img.convertTo(color_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
	float sig_diff;
	Mat output;

	if (doubleImageSize)
	{
		sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f));
		Mat dbl;
		resize(color_fpt, dbl, Size(colorImg.cols * 2, colorImg.rows * 2), 0, 0, INTER_LINEAR);
		GaussianBlur(dbl, output, Size(), sig_diff, sig_diff);
		return output;
	}
	else
	{
		sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f));
		GaussianBlur(color_fpt, output, Size(), sig_diff, sig_diff);
		return output;
	}
}

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
void HoWH::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints, OutputArray _descriptors, bool useProvidedKeypoints) const
{
		int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
		Mat image = _image.getMat(), mask = _mask.getMat();
		if(image.empty() || image.depth() != CV_8U)
			CV_Error(CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

		if (!mask.empty() && mask.type() != CV_8UC1)
			CV_Error(CV_StsBadArg, "mask has incorrect type (!=CV_8UC1)");

		if (useProvidedKeypoints)
		{
			firstOctave = 0;
			int maxOctave = INT_MIN;
			for (size_t i = 0; i < keypoints.size(); i++)
			{
				int octave, layer;
				float scale;
				unpackOctave(keypoints[i], octave, layer, scale);
				firstOctave = std::min(firstOctave, octave);
				maxOctave = std::max(maxOctave, octave);
				actualNLayers = std::max(actualNLayers, layer - 2);
			}

			firstOctave = std::min(firstOctave, 0);
			CV_Assert(firstOctave >= -1 && actualNLayers <= nOctaveLayers);
			actualNOctaves = maxOctave - firstOctave + 1;
		}
		// base is a grey image
		Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
		//initialize color image
		Mat colorBase = createInitialColorImage(image, firstOctave < 0, (float)sigma);
		vector<Mat> gpyr, dogpyr, colorGpyr; // colorGpyr is a gaussian pyramid for color image
		int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log((double)std::min(base.cols, base.rows)) / log(2.) - 2) - firstOctave;

		//double t, tf = getTickFrequency();
		//t = (double)getTickCount();
		buildGaussianPyramid(base, gpyr, nOctaves);
		buildDoGPyramid(gpyr, dogpyr);
		// build color gaussian pyramid
		buildGaussianPyramid(colorBase, colorGpyr, nOctaves);

		// convert each blurred BGR image to HSV in the pyramid
		for (int o = 0; o < nOctaves; o++)
		{
			for (int i = 0; i < nOctaveLayers + 3; i++)
			{
				cvtColor(colorGpyr[o*(nOctaveLayers + 3) + i], colorGpyr[o*(nOctaveLayers + 3) + i], COLOR_BGR2HSV);
			}
		}
		//t = (double)getTickCount() - t;
		//printf("pyramid construction time: %g\n", t*1000./tf);

		if (!useProvidedKeypoints)
		{
			//t = (double)getTickCount();
			findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
			KeyPointsFilter::removeDuplicated(keypoints);

			if (nfeatures > 0)
				KeyPointsFilter::retainBest(keypoints, nfeatures);
			//t = (double)getTickCount() - t;
			//printf("keypoint detection time: %g\n", t*1000./tf);

			if (firstOctave < 0) {

				for (size_t i = 0; i < keypoints.size(); i++)
				{
					KeyPoint& kpt = keypoints[i];
					float scale = 1.f / (float)(1 << -firstOctave);
					kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
					kpt.pt *= scale;
					kpt.size *= scale;
				}

			}

			if (!mask.empty())
				KeyPointsFilter::runByPixelsMask(keypoints, mask);
		} else {
			// filter keypoints by mask
			KeyPointsFilter::runByPixelsMask( keypoints, mask );
		}

		if (_descriptors.needed())
		{
			//t = (double)getTickCount();
			int dsize = descriptorSize();
			_descriptors.create((int)keypoints.size(), dsize, CV_32F);
			Mat descriptors = _descriptors.getMat();

			calcDescriptors(colorGpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
			//t = (double)getTickCount() - t;
			//printf("descriptor extraction time: %g\n", t*1000./tf);
		}
}

//------------------------------------operator()---------------------------------------
// Same as operator(), but takes the parameters needed for domain size pooling
//-------------------------------------------------------------------------------------
void HoWH::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints, OutputArray _descriptors,
	int numScales, double linePoint1, double linePoint2, bool useProvidedKeypoints) const
{
	int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
	Mat image = _image.getMat(), mask = _mask.getMat();
	if (image.empty() || image.depth() != CV_8U)
		CV_Error(CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

	if (!mask.empty() && mask.type() != CV_8UC1)
		CV_Error(CV_StsBadArg, "mask has incorrect type (!=CV_8UC1)");

	if (useProvidedKeypoints)
	{
		firstOctave = 0;
		int maxOctave = INT_MIN;
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			int octave, layer;
			float scale;
			unpackOctave(keypoints[i], octave, layer, scale);
			firstOctave = std::min(firstOctave, octave);
			maxOctave = std::max(maxOctave, octave);
			actualNLayers = std::max(actualNLayers, layer - 2);
		}

		firstOctave = std::min(firstOctave, 0);
		CV_Assert(firstOctave >= -1 && actualNLayers <= nOctaveLayers);
		actualNOctaves = maxOctave - firstOctave + 1;
	}
	// base is a grey image
	Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
	//initialize color image
	Mat colorBase = createInitialColorImage(image, firstOctave < 0, (float)sigma);
	vector<Mat> gpyr, dogpyr, colorGpyr; // colorGpyr is a gaussian pyramid for color image
	int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log((double)std::min(base.cols, base.rows)) / log(2.) - 2) - firstOctave;

	//for domain-size pooling, in case it needs more octaves [due to affecting keypoint size].
	if (numScales > 1 && linePoint1 < 1.0) nOctaves += 1;

	//double t, tf = getTickFrequency();
	//t = (double)getTickCount();
	buildGaussianPyramid(base, gpyr, nOctaves);
	buildDoGPyramid(gpyr, dogpyr);
	// build color gaussian pyramid
	buildGaussianPyramid(colorBase, colorGpyr, nOctaves);

	// convert each blurred BGR image to HSV in the pyramid
	for (int o = 0; o < nOctaves; o++)
	{
		for (int i = 0; i < nOctaveLayers + 3; i++)
		{
			cvtColor(colorGpyr[o*(nOctaveLayers + 3) + i], colorGpyr[o*(nOctaveLayers + 3) + i], COLOR_BGR2HSV);
		}
	}
	//t = (double)getTickCount() - t;
	//printf("pyramid construction time: %g\n", t*1000./tf);

	if (!useProvidedKeypoints)
	{
		//t = (double)getTickCount();
		findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
		KeyPointsFilter::removeDuplicated(keypoints);

		if (nfeatures > 0)
			KeyPointsFilter::retainBest(keypoints, nfeatures);
		//t = (double)getTickCount() - t;
		//printf("keypoint detection time: %g\n", t*1000./tf);

		if (firstOctave < 0) {

			for (size_t i = 0; i < keypoints.size(); i++)
			{
				KeyPoint& kpt = keypoints[i];
				float scale = 1.f / (float)(1 << -firstOctave);
				kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
				kpt.pt *= scale;
				kpt.size *= scale;
			}

		}

		if (!mask.empty())
			KeyPointsFilter::runByPixelsMask(keypoints, mask);
	}
	else {
		// filter keypoints by mask
		KeyPointsFilter::runByPixelsMask(keypoints, mask);
	}

	if (_descriptors.needed())
	{
		//t = (double)getTickCount();
		int dsize = descriptorSize();
		_descriptors.create((int)keypoints.size(), dsize, CV_32F);
		Mat descriptors = _descriptors.getMat();

		calcDescriptors(colorGpyr, keypoints, descriptors, nOctaveLayers, firstOctave,
			numScales, linePoint1, linePoint2);
		//t = (double)getTickCount() - t;
		//printf("descriptor extraction time: %g\n", t*1000./tf);
	}
}

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
void HoWH::calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const
{
	Point pt(cvRound(ptf.x), cvRound(ptf.y));	//point object
	float cos_t = cosf(ori*(float)(CV_PI / 180));
	float sin_t = sinf(ori*(float)(CV_PI / 180));
	float bins_per_degree = n / 360.f;
	float exp_scale = -1.f / (d * d * 0.5f);
	float hist_width = SIFT_DESCR_SCL_FCTR * scl;
	int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
	// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
	radius = std::min(radius, (int)sqrt((double)img.cols*img.cols + img.rows*img.rows));
	cos_t /= hist_width;
	sin_t /= hist_width;

	int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1), histlen = (d + 2)*(d + 2)*(n + 2); 
	int rows = img.rows, cols = img.cols;

	//reserve memory for storage
	AutoBuffer<float> buf(len * 7 + histlen);
	float *X = buf, *Y = X + len, *Sat = Y, *Hue = Sat + len, *W = Hue + len;
	float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

	//initialize the histogram
	for (i = 0; i < d + 2; i++)
	{
		for (j = 0; j < d + 2; j++)
			for (k = 0; k < n + 2; k++)
				hist[(i*(d + 2) + j)*(n + 2) + k] = 0.;
	}

	for (i = -radius, k = 0; i <= radius; i++)
		for (j = -radius; j <= radius; j++)
		{
		// Calculate sample's histogram array coords rotated relative to ori.
		// Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
		// r_rot = 1.5) have full weight placed in row 1 after interpolation.
		float c_rot = j * cos_t - i * sin_t;
		float r_rot = j * sin_t + i * cos_t;
		float rbin = r_rot + d / 2 - 0.5f;
		float cbin = c_rot + d / 2 - 0.5f;
		int r = pt.y + i, c = pt.x + j;

		if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
			r > 0 && r < rows - 1 && c > 0 && c < cols - 1)
		{
			float dx = (float)(img.at<sift_wt>(r, c + 1) - img.at<sift_wt>(r, c - 1));
			float dy = (float)(img.at<sift_wt>(r - 1, c) - img.at<sift_wt>(r + 1, c));
			X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
			//assign hue and saturation value to storage
			Hue[k] = img.at<Vec3f>(r, c)[0];
			Sat[k] = img.at<Vec3f>(r, c)[1];
			W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
			k++;
		}
		}

	len = k;
	hal::exp(W, W, len);

	for (k = 0; k < len; k++)
	{
		float rbin = RBin[k], cbin = CBin[k];
		//hue value
		float hue = (Hue[k])*bins_per_degree;
		//sat value
		float sat = Sat[k] * W[k];

		int r0 = cvFloor(rbin);
		int c0 = cvFloor(cbin);
		int h0 = cvFloor(hue);
		rbin -= r0;
		cbin -= c0;
		hue -= h0;

		if (h0 < 0)
			h0 += n;
		if (h0 >= n)
			h0 -= n;

		// histogram update using tri-linear interpolation		
		float v_r1 = sat*rbin, v_r0 = sat - v_r1;
		float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
		float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
		float v_rch111 = v_rc11*hue, v_rch110 = v_rc11 - v_rch111;
		float v_rch101 = v_rc10*hue, v_rch100 = v_rc10 - v_rch101;
		float v_rch011 = v_rc01*hue, v_rch010 = v_rc01 - v_rch011;
		float v_rch001 = v_rc00*hue, v_rch000 = v_rc00 - v_rch001;

		int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + h0;
		hist[idx] += v_rch000;
		hist[idx + 1] += v_rch001;
		hist[idx + (n + 2)] += v_rch010;
		hist[idx + (n + 3)] += v_rch011;
		hist[idx + (d + 2)*(n + 2)] += v_rch100;
		hist[idx + (d + 2)*(n + 2) + 1] += v_rch101;
		hist[idx + (d + 3)*(n + 2)] += v_rch110;
		hist[idx + (d + 3)*(n + 2) + 1] += v_rch111;
	}

	//-------------------------------------------------------------------------------------
	// finalize histogram, since the orientation histograms are circular fixes things 
	for (i = 0; i < d; i++)
		for (j = 0; j < d; j++)
		{
		int idx = ((i + 1)*(d + 2) + (j + 1))*(n + 2);
		hist[idx] += hist[idx + n];
		hist[idx + 1] += hist[idx + n + 1];
		for (k = 0; k < n; k++)
			dst[(i*d + j)*n + k] = hist[idx + k];
		}
	// copy histogram to the descriptor,
	// apply hysteresis thresholding
	// and scale the result, so that it can be easily converted
	// to byte array
	float nrm2 = 0;
	len = d*d*n;
	for (k = 0; k < len; k++)
		nrm2 += dst[k] * dst[k];
	float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR; 
	for (i = 0, nrm2 = 0; i < k; i++)
	{
		float val = std::min(dst[i], thr);
		dst[i] = val;
		nrm2 += val*val;
	}
	nrm2 = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

	for (k = 0; k < len; k++)
	{
		dst[k] = (0.6f * dst[k] * nrm2);	// cfolson: 0.6 weight helps  when stacked with SIFT + others
	}

}


//------------------------------------descriptorSize()---------------------------------
// ! returns the descriptor size in floats
//Precondition: None
//Postcondition: the descriptor size is returned in floats
//-------------------------------------------------------------------------------------
int HoWH::descriptorSize() const
{
	return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}