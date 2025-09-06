//-------------------------------------------------------------------------
// Name: HoNC.cpp
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


/**********************************************************************************************\
Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/SIFT/
Below is the original copyright.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.
\**********************************************************************************************/

#include "HoNC.h"

// number of buckets in each dimension for R G B
static const int SIZE = 2;

// range of color value of each bucket
static const int BUCKET_SIZE = 256 / SIZE;

// Average standard deviation of three color bands for each keypoint
static const float AVG_STD_DEV = 48.f;

// constructor
HoNC::HoNC()
{
	// do nothing
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
void HoNC::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints, OutputArray _descriptors, bool useProvidedKeypoints) const
{
	int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
	Mat image = _image.getMat(), mask = _mask.getMat();

	if (image.empty() || image.depth() != CV_8U)
		CV_Error(CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

	if (!mask.empty() && mask.type() != CV_8UC1)
		CV_Error(CV_StsBadArg, "mask has incorrect type (!=CV_8UC1)");
	// use pre-defined keypoints
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
	// build color gaussian pyramid
	buildGaussianPyramid(colorBase, colorGpyr, nOctaves);
	// build color Dog
	buildDoGPyramid(colorGpyr, dogpyr);
	//t = (double)getTickCount() - t;
	//printf("pyramid construction time: %g\n", t*1000./tf);
	// calculate keypoints
	if (!useProvidedKeypoints)
	{
		//t = (double)getTickCount();
		findScaleSpaceExtrema(colorGpyr, dogpyr, keypoints);
		KeyPointsFilter::removeDuplicated(keypoints);

		if (nfeatures > 0)
			KeyPointsFilter::retainBest(keypoints, nfeatures);
		//t = (double)getTickCount() - t;
		//printf("keypoint detection time: %g\n", t*1000./tf);

		if (firstOctave < 0)
			for (size_t i = 0; i < keypoints.size(); i++)
			{
			KeyPoint& kpt = keypoints[i];
			float scale = 1.f / (float)(1 << -firstOctave);
			kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
			kpt.pt *= scale;
			kpt.size *= scale;
			}

		if (!mask.empty())
			KeyPointsFilter::runByPixelsMask(keypoints, mask);
	}
	else
	{
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


void HoNC::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints, OutputArray _descriptors,
	int numScales, double linePoint1, double linePoint2, bool useProvidedKeypoints) const
{
	int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
	Mat image = _image.getMat(), mask = _mask.getMat();

	if (image.empty() || image.depth() != CV_8U)
		CV_Error(CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

	if (!mask.empty() && mask.type() != CV_8UC1)
		CV_Error(CV_StsBadArg, "mask has incorrect type (!=CV_8UC1)");
	// use pre-defined keypoints
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
	// build color gaussian pyramid
	buildGaussianPyramid(colorBase, colorGpyr, nOctaves);
	// build color Dog
	buildDoGPyramid(colorGpyr, dogpyr);
	//t = (double)getTickCount() - t;
	//printf("pyramid construction time: %g\n", t*1000./tf);
	// calculate keypoints
	if (!useProvidedKeypoints)
	{
		//t = (double)getTickCount();
		findScaleSpaceExtrema(colorGpyr, dogpyr, keypoints);
		KeyPointsFilter::removeDuplicated(keypoints);

		if (nfeatures > 0)
			KeyPointsFilter::retainBest(keypoints, nfeatures);
		//t = (double)getTickCount() - t;
		//printf("keypoint detection time: %g\n", t*1000./tf);

		if (firstOctave < 0)
			for (size_t i = 0; i < keypoints.size(); i++)
			{
				KeyPoint& kpt = keypoints[i];
				float scale = 1.f / (float)(1 << -firstOctave);
				kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
				kpt.pt *= scale;
				kpt.size *= scale;
			}

		if (!mask.empty())
			KeyPointsFilter::runByPixelsMask(keypoints, mask);
	}
	else
	{
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
void HoNC::calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const
{
	Point pt(cvRound(ptf.x), cvRound(ptf.y));	//point object
	float cos_t = cosf(ori*(float)(CV_PI / 180));	
	float sin_t = sinf(ori*(float)(CV_PI / 180));
	float exp_scale = -1.f / (d * d * 0.5f);
	float hist_width = SIFT_DESCR_SCL_FCTR * scl;
	int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
	// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
	radius = std::min(radius, (int)sqrt((double)img.cols*img.cols + img.rows*img.rows));
	cos_t /= hist_width;
	sin_t /= hist_width;

	int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1), histlen = (d + 2)*(d + 2)*(n + 2);
	int rows = img.rows, cols = img.cols;

	AutoBuffer<float> buf(len * 6 + histlen);
	float *RBin = buf, *CBin = RBin + len;
	//reserve memory for RGB value of all inclosed pixels
	float *RedBin = CBin + len, *GreenBin = RedBin + len, *BlueBin = GreenBin + len;
	// rotational weight
	float *W = BlueBin + len;
	float *hist = W + len;
	//Vote for 8 color buckets
	for (i = 0; i < d + 2; i++)
	{
		for (j = 0; j < d + 2; j++)
			for (k = 0; k < n + 2; k++)
				hist[(i*(d + 2) + j)*(n + 2) + k] = 0.;
	}
	// variables to compute average and squared average for each color
	float rbar = 0, gbar = 0, bbar = 0, r2bar = 0, g2bar = 0, b2bar = 0;
	for (i = -radius, k = 0; i <= radius; i++)
		for (j = -radius; j <= radius; j++)
		{
		// Calculate sample's histogram array coords rotated relative to ori.
		// Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
		// r_rot = 1.5) have full weight placed in row 1 after interpolation.
		float c_rot = j * cos_t - i * sin_t;  // column after "rotation"
		float r_rot = j * sin_t + i * cos_t;  // row after "rotation"
		float rbin = r_rot + d / 2 - 0.5f;   // row index of 4x4 bin
		float cbin = c_rot + d / 2 - 0.5f;   // col index of 4x4 bin
		int r = pt.y + i, c = pt.x + j;      // row and column index in actual image
			
		//need a r array, g array , b array instead of X and Y.
		if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
			r > 0 && r < rows - 1 && c > 0 && c < cols - 1)
			{
				W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale; 
				RBin[k] = rbin; CBin[k] = cbin;
				//changes: color histogram
				float red = img.at<Vec3f>(r, c)[2];
				float green = img.at<Vec3f>(r, c)[1];
				float blue = img.at<Vec3f>(r, c)[0];
				RedBin[k] = red;
				GreenBin[k] = green;
				BlueBin[k] = blue;

				rbar += red;
				gbar += green;
				bbar += blue;
				r2bar += red * red;
				g2bar += green * green;
				b2bar += blue * blue;

				k++;
			}
		}

	len = k;

	// Compute averages and standard deviations
	rbar = rbar / (float) len;
	gbar = gbar / (float) len;
	bbar = bbar / (float) len;

	r2bar = r2bar / (float) len;
	g2bar = g2bar / (float) len;
	b2bar = b2bar / (float) len;

	float rsig = sqrt(r2bar - rbar * rbar);
	float gsig = sqrt(g2bar - gbar * gbar);
	float bsig = sqrt(b2bar - bbar * bbar);

	float bias = (127.5f) - (rbar + gbar + bbar) / 3;
	float gain = AVG_STD_DEV / ((rsig + gsig + bsig) / 3);

	bool separateColorNormalization = false;		// cfolson: Generally, I think this should be false
	float rbias = bias;
	float gbias = bias;
	float bbias = bias;
	float rgain = gain;
	float ggain = gain;
	float bgain = gain;
	if (separateColorNormalization) {
		rbias = 127.5f - rbar;
		gbias = 127.5f - gbar;
		bbias = 127.5f - bbar;
		rgain = AVG_STD_DEV / rsig;
		ggain = AVG_STD_DEV / gsig;
		bgain = AVG_STD_DEV / bsig;
	}
	
	// for testing with no normalization
	/*
	rbias = gbias = bbias = 0;
	rgain = ggain = bgain = 1;
	*/

	for (k = 0; k < len; k++) {
		RedBin[k] = (RedBin[k] - rbar) * rgain + rbar + rbias;
		GreenBin[k] = (GreenBin[k] - gbar) * ggain + gbar + gbias;
		BlueBin[k] = (BlueBin[k] - bbar) * bgain + bbar + bbias;

	}
	
	hal::exp(W, W, len);

	// going through all enclosed pixels and vote for bucket
	for (k = 0; k < len; k++)
	{
		float rbin = RBin[k], cbin = CBin[k];
		// RGV color value for keypoint pixel k
		float red = RedBin[k];
		float blue = BlueBin[k];
		float green = GreenBin[k];
		// weight
		float redWeight = 1.0f - (float) std::min(std::max((red - 63.5) / 128, 0.0), 1.0);
		float blueWeight = 1.0f - (float) std::min(std::max((blue - 63.5) / 128, 0.0), 1.0);
		float greenWeight = 1.0f - (float) std::min(std::max((green - 63.5) / 128, 0.0), 1.0);

		// weight container
		float rWeight[2] = { redWeight, (1.0f - redWeight) };
		float bWeight[2] = { blueWeight, (1.0f - blueWeight) };
		float gWeight[2] = { greenWeight, (1.0f - greenWeight) };

		int r0 = cvFloor(rbin);
		int c0 = cvFloor(cbin);
		rbin -= r0;
		cbin -= c0;

		//// histogram update using tri-linear interpolation
		// vote is weighted by 1 in colo histogram
		// (while vote is weighted by gradient magnitude in grediant orientation histogram)
		float v_r1 = W[k] * rbin, v_r0 = W[k] - v_r1;
		float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
		float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01; 
		float v_rco110 = v_rc11;
		float v_rco100 = v_rc10;
		float v_rco010 = v_rc01;
		float v_rco000 = v_rc00;
		for (int red = 0; red < 2; red++) 
		{
			for (int green = 0; green < 2; green++) 
			{
				for (int blue = 0; blue < 2; blue++) 
				{
					//Voting
					int binIndex = 4 * red + 2 * green + blue;
					int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + binIndex;
					hist[idx] += v_rco000 * rWeight[red] * gWeight[green] * bWeight[blue];
					hist[idx + (n + 2)] += v_rco010 * rWeight[red] * gWeight[green] * bWeight[blue];
					hist[idx + (d + 2)*(n + 2)] += v_rco100 * rWeight[red] * gWeight[green] * bWeight[blue];
					hist[idx + (d + 3)*(n + 2)] += v_rco110 * rWeight[red] * gWeight[green] * bWeight[blue];
				}
			}
		}
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
	float thr = std::sqrt(nrm2) * SIFT_DESCR_MAG_THR; 
	for (i = 0, nrm2 = 0; i < k; i++)
	{
		float val = std::min(dst[i], thr);
		dst[i] = val;
		nrm2 += val*val;
	}
	nrm2 = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);
	
	for (k = 0; k < len; k++)
	{
		dst[k] = dst[k] * nrm2;
	}

}

//------------------------------------findScaleSpaceExtrema()--------------------------
// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
//Precondition: the following parameters must be correctly defined.
//parameters:
//gauss_pyr: gaussian pyramid
//dog_pyr: difference of Gaussian pyramid
//keypoints: empty keypoints vector
//Postcondition: keypoints are assigned
//-------------------------------------------------------------------------------------

// NOTES: We never were able to get improved results with color SIFT extractors. This code is not currently in use.
void HoNC::findScaleSpaceExtrema(const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
	std::vector<KeyPoint>& keypoints) const
{
	int nOctaves = (int)gauss_pyr.size() / (nOctaveLayers + 3);
	int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
	const int n = SIFT_ORI_HIST_BINS;
	float hist[n];
	KeyPoint kpt;

	keypoints.clear();

	for (int o = 0; o < nOctaves; o++)
		for (int i = 1; i <= nOctaveLayers; i++)
		{
		int idx = o*(nOctaveLayers + 2) + i;
		const Mat& img = dog_pyr[idx];
		const Mat& prev = dog_pyr[idx - 1];
		const Mat& next = dog_pyr[idx + 1];
        // TODO: This variable is not used. Determine if needed
		//int step = (int)img.step1();
		int rows = img.rows, cols = img.cols;

		for (int r = SIFT_IMG_BORDER; r < rows - SIFT_IMG_BORDER; r++)
		{
			const Vec3f* currptr = img.ptr<Vec3f>(r);
			const Vec3f* currptrminus = img.ptr<Vec3f>(r - 1);
			const Vec3f* currptrplus = img.ptr<Vec3f>(r + 1);
			const Vec3f* prevptr = prev.ptr<Vec3f>(r);
			const Vec3f* prevptrminus = prev.ptr<Vec3f>(r - 1);
			const Vec3f* prevptrplus = prev.ptr<Vec3f>(r + 1);
			const Vec3f* nextptr = next.ptr<Vec3f>(r);
			const Vec3f* nextptrminus = next.ptr<Vec3f>(r - 1);
			const Vec3f* nextptrplus = next.ptr<Vec3f>(r + 1);
			for (int c = SIFT_IMG_BORDER; c < cols - SIFT_IMG_BORDER; c++)
			{
				//get the average of RGB
				sift_wt val = (abs(currptr[c].val[0]) + abs(currptr[c].val[1]) + abs(currptr[c].val[2])) / 3;
				//list of local values 
				//current level
				sift_wt cur_val1 = (abs(currptr[c - 1].val[0]) + abs(currptr[c - 1].val[1]) + abs(currptr[c - 1].val[2])) / 3;
				sift_wt cur_val2 = (abs(currptr[c + 1].val[0]) + abs(currptr[c + 1].val[1]) + abs(currptr[c + 1].val[2])) / 3;
				sift_wt cur_val3 = (abs(currptrminus[c - 1].val[0]) + abs(currptrminus[c - 1].val[1]) + abs(currptrminus[c - 1].val[2])) / 3;
				sift_wt cur_val4 = (abs(currptrminus[c].val[0]) + abs(currptrminus[c].val[1]) + abs(currptrminus[c].val[2])) / 3;
				sift_wt cur_val5 = (abs(currptrminus[c + 1].val[0]) + abs(currptrminus[c + 1].val[1]) + abs(currptrminus[c + 1].val[2])) / 3;
				sift_wt cur_val6 = (abs(currptrplus[c - 1].val[0]) + abs(currptrplus[c - 1].val[1]) + abs(currptrplus[c - 1].val[2])) / 3;
				sift_wt cur_val7 = (abs(currptrplus[c].val[0]) + abs(currptrplus[c].val[1]) + abs(currptrplus[c].val[2])) / 3;
				sift_wt cur_val8 = (abs(currptrplus[c + 1].val[0]) + abs(currptrplus[c + 1].val[1]) + abs(currptrplus[c + 1].val[2])) / 3;
				//next level
				sift_wt next_val1 = (abs(nextptr[c - 1].val[0]) + abs(nextptr[c - 1].val[1]) + abs(nextptr[c - 1].val[2])) / 3;
				sift_wt next_val2 = (abs(nextptr[c].val[0]) + abs(nextptr[c].val[1]) + abs(nextptr[c].val[2])) / 3;
				sift_wt next_val3 = (abs(nextptr[c + 1].val[0]) + abs(nextptr[c + 1].val[1]) + abs(nextptr[c + 1].val[2])) / 3;
				sift_wt next_val4 = (abs(nextptrminus[c - 1].val[0]) + abs(nextptrminus[c - 1].val[1]) + abs(nextptrminus[c - 1].val[2])) / 3;
				sift_wt next_val5 = (abs(nextptrminus[c].val[0]) + abs(nextptrminus[c].val[1]) + abs(nextptrminus[c].val[2])) / 3;
				sift_wt next_val6 = (abs(nextptrminus[c + 1].val[0]) + abs(nextptrminus[c + 1].val[1]) + abs(nextptrminus[c + 1].val[2])) / 3;
				sift_wt next_val7 = (abs(nextptrplus[c - 1].val[0]) + abs(nextptrplus[c - 1].val[1]) + abs(nextptrplus[c - 1].val[2])) / 3;
				sift_wt next_val8 = (abs(nextptrplus[c].val[0]) + abs(nextptrplus[c].val[1]) + abs(nextptrplus[c].val[2])) / 3;
				sift_wt next_val9 = (abs(nextptrplus[c + 1].val[0]) + abs(nextptrplus[c + 1].val[1]) + abs(nextptrplus[c + 1].val[2])) / 3;
				//previous level
				sift_wt prev_val1 = (abs(prevptr[c - 1].val[0]) + abs(prevptr[c - 1].val[1]) + abs(prevptr[c - 1].val[2])) / 3;
				sift_wt prev_val2 = (abs(prevptr[c].val[0]) + abs(prevptr[c].val[1]) + abs(prevptr[c].val[2])) / 3;
				sift_wt prev_val3 = (abs(prevptr[c + 1].val[0]) + abs(prevptr[c + 1].val[1]) + abs(prevptr[c + 1].val[2])) / 3;
				sift_wt prev_val4 = (abs(prevptrminus[c - 1].val[0]) + abs(prevptrminus[c - 1].val[1]) + abs(prevptrminus[c - 1].val[2])) / 3;
				sift_wt prev_val5 = (abs(prevptrminus[c].val[0]) + abs(prevptrminus[c].val[1]) + abs(prevptrminus[c].val[2])) / 3;
				sift_wt prev_val6 = (abs(prevptrminus[c + 1].val[0]) + abs(prevptrminus[c + 1].val[1]) + abs(prevptrminus[c + 1].val[2])) / 3;
				sift_wt prev_val7 = (abs(prevptrplus[c - 1].val[0]) + abs(prevptrplus[c - 1].val[1]) + abs(prevptrplus[c - 1].val[2])) / 3;
				sift_wt prev_val8 = (abs(prevptrplus[c].val[0]) + abs(prevptrplus[c].val[1]) + abs(prevptrplus[c].val[2])) / 3;
				sift_wt prev_val9 = (abs(prevptrplus[c + 1].val[0]) + abs(prevptrplus[c + 1].val[1]) + abs(prevptrplus[c + 1].val[2])) / 3;
				// find local extrema with pixel accuracy using green value
				if ( std::abs(val) > threshold &&
					((val > 0 && val >= cur_val1 && val >= cur_val2 &&            //local maxima
					val >= cur_val3 && val >= cur_val4 && val >= cur_val5 &&
					val >= cur_val6 && val >= cur_val7 && val >= cur_val8 &&
					val >= next_val2 && val >= next_val1 && val >= next_val3 &&
					val >= next_val4 && val >= next_val5 && val >= next_val6 &&
					val >= next_val7 && val >= next_val8 && val >= next_val9 &&
					val >= prev_val2 && val >= prev_val1 && val >= prev_val3 &&
					val >= prev_val4 && val >= prev_val5 && val >= prev_val6 &&
					val >= prev_val7 && val >= prev_val8 && val >= prev_val9)
					||
					(val < 0 && val <= cur_val1 && val <= cur_val2 &&            //local minima
					val <= cur_val3 && val <= cur_val4 && val <= cur_val5 &&
					val <= cur_val6 && val <= cur_val7 && val <= cur_val8 &&
					val <= next_val2 && val <= next_val1 && val <= next_val3 &&
					val <= next_val4 && val <= next_val5 && val <= next_val6 &&
					val <= next_val7 && val <= next_val8 && val <= next_val9 &&
					val <= prev_val2 && val <= prev_val1 && val <= prev_val3 &&
					val <= prev_val4 && val <= prev_val5 && val <= prev_val6 &&
					val <= prev_val7 && val <= prev_val8 && val <= prev_val9)))
				{
					int r1 = r, c1 = c, layer = i;
					if (!adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
						nOctaveLayers, (float)contrastThreshold,
						(float)edgeThreshold, (float)sigma))
						continue;
					float scl_octv = kpt.size*0.5f / (1 << o);
					//change
					float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers + 3) + layer],
						Point(c1, r1),
						cvRound(SIFT_ORI_RADIUS * scl_octv),
						SIFT_ORI_SIG_FCTR * scl_octv,
						hist, n);
					float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
					for (int j = 0; j < n; j++)
					{
						int l = j > 0 ? j - 1 : n - 1;
						int r2 = j < n - 1 ? j + 1 : 0;

						if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr)
						{
							float bin = j + 0.5f * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[j] + hist[r2]);
							bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
							kpt.angle = 360.f - (float)((360.f / n) * bin);
							if (std::abs(kpt.angle - 360.f) < FLT_EPSILON)
								kpt.angle = 0.f;
							keypoints.push_back(kpt);
						}
					}
				}
			}
		}
	}
}

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
bool HoNC::adjustLocalExtrema(const std::vector<Mat>& dog_pyr, KeyPoint& kpt, int octv, int& layer, int& r, int& c, int nOctaveLayers, float contrastThreshold, float edgeThreshold, float sigma)
{
	const float img_scale = 1.f / (255 * SIFT_FIXPT_SCALE);
	const float deriv_scale = img_scale*0.5f;
	const float second_deriv_scale = img_scale;
	const float cross_deriv_scale = img_scale*0.25f;

	float xi = 0, xr = 0, xc = 0, contr = 0;
	int i = 0;

	for (; i < SIFT_MAX_INTERP_STEPS; i++)
	{
		int idx = octv*(nOctaveLayers + 2) + layer;
		const Mat& img = dog_pyr[idx];
		const Mat& prev = dog_pyr[idx - 1];
		const Mat& next = dog_pyr[idx + 1];
		//
		sift_wt dD_val1 = (abs(img.at<Vec3f>(r, c + 1).val[0]) - abs(img.at<Vec3f>(r, c - 1).val[0])
						+ abs(img.at<Vec3f>(r, c + 1).val[1]) - abs(img.at<Vec3f>(r, c - 1).val[1])
						+ abs(img.at<Vec3f>(r, c + 1).val[2]) - abs(img.at<Vec3f>(r, c - 1).val[2])) / 3;
		sift_wt dD_val2 = (abs(img.at<Vec3f>(r + 1, c).val[0]) - abs(img.at<Vec3f>(r - 1, c).val[0])
						+ abs(img.at<Vec3f>(r + 1, c).val[1]) - abs(img.at<Vec3f>(r - 1, c).val[1])
						+ abs(img.at<Vec3f>(r + 1, c).val[2]) - abs(img.at<Vec3f>(r - 1, c).val[2])) / 3;
		sift_wt dD_val3 = (abs(next.at<Vec3f>(r, c).val[0]) - abs(prev.at<Vec3f>(r, c).val[0])
						+ abs(next.at<Vec3f>(r, c).val[1]) - abs(prev.at<Vec3f>(r, c).val[1])
						+ abs(next.at<Vec3f>(r, c).val[2]) - abs(prev.at<Vec3f>(r, c).val[2])) / 3;
		//
		Vec3f dD(dD_val1 * deriv_scale, dD_val2 * deriv_scale, dD_val3 * deriv_scale);
		// value 2 
		float v2 = (float)(abs(img.at<Vec3f>(r, c).val[0]) + abs(img.at<Vec3f>(r, c).val[1]) + abs(img.at<Vec3f>(r, c).val[2])) / 3 * 2;
		// 
		float dxx = ((abs(img.at<Vec3f>(r, c + 1).val[0]) + abs(img.at<Vec3f>(r, c + 1).val[1]) + abs(img.at<Vec3f>(r, c + 1).val[2])
				  + abs(img.at<Vec3f>(r, c - 1).val[0]) + abs(img.at<Vec3f>(r, c - 1).val[1]) + abs(img.at<Vec3f>(r, c - 1).val[2])) / 3
				   - v2) * second_deriv_scale;
		//
		float dyy = ((abs(img.at<Vec3f>(r + 1, c).val[0]) + abs(img.at<Vec3f>(r + 1, c).val[1]) + abs(img.at<Vec3f>(r + 1, c).val[2])
				  + abs(img.at<Vec3f>(r - 1, c).val[0]) + abs(img.at<Vec3f>(r - 1, c).val[1]) + abs(img.at<Vec3f>(r - 1, c).val[2])) / 3
				  - v2) * second_deriv_scale;
		//
		float dss = ((abs(next.at<Vec3f>(r, c).val[0]) + abs(next.at<Vec3f>(r, c).val[1]) + abs(next.at<Vec3f>(r, c).val[2])
				  + abs(prev.at<Vec3f>(r, c).val[0]) + abs(prev.at<Vec3f>(r, c).val[1]) + abs(prev.at<Vec3f>(r, c).val[2])) / 3
				  - v2) * second_deriv_scale;
		//
		float dxy = (abs(img.at<Vec3f>(r + 1, c + 1).val[0]) + abs(img.at<Vec3f>(r + 1, c + 1).val[1]) + abs(img.at<Vec3f>(r + 1, c + 1).val[2])
				  - abs(img.at<Vec3f>(r + 1, c - 1).val[0]) - abs(img.at<Vec3f>(r + 1, c - 1).val[1]) - abs(img.at<Vec3f>(r + 1, c - 1).val[2])
				  - abs(img.at<Vec3f>(r - 1, c + 1).val[0]) - abs(img.at<Vec3f>(r - 1, c + 1).val[1]) - abs(img.at<Vec3f>(r - 1, c + 1).val[2])
				  + abs(img.at<Vec3f>(r - 1, c - 1).val[0]) + abs(img.at<Vec3f>(r - 1, c - 1).val[1]) + abs(img.at<Vec3f>(r - 1, c - 1).val[2])) / 3 * cross_deriv_scale;
		//
		float dxs = (abs(next.at<Vec3f>(r, c + 1).val[0]) + abs(next.at<Vec3f>(r, c + 1).val[1]) + abs(next.at<Vec3f>(r, c + 1).val[2])
				  - abs(next.at<Vec3f>(r, c - 1).val[0]) - abs(next.at<Vec3f>(r, c - 1).val[1]) - abs(next.at<Vec3f>(r, c - 1).val[2])
				  - abs(prev.at<Vec3f>(r, c + 1).val[0]) - abs(prev.at<Vec3f>(r, c + 1).val[1]) - abs(prev.at<Vec3f>(r, c + 1).val[2])
				  + abs(prev.at<Vec3f>(r, c - 1).val[0]) + abs(prev.at<Vec3f>(r, c - 1).val[1]) + abs(prev.at<Vec3f>(r, c - 1).val[2])) / 3 * cross_deriv_scale;
		//
		float dys = (abs(next.at<Vec3f>(r + 1, c).val[0]) + abs(next.at<Vec3f>(r + 1, c).val[1]) + abs(next.at<Vec3f>(r + 1, c).val[2])
				  - abs(next.at<Vec3f>(r - 1, c).val[0]) - abs(next.at<Vec3f>(r - 1, c).val[1]) - abs(next.at<Vec3f>(r - 1, c).val[2])
				  - abs(prev.at<Vec3f>(r + 1, c).val[0]) - abs(prev.at<Vec3f>(r + 1, c).val[1]) - abs(prev.at<Vec3f>(r + 1, c).val[2])
				  + abs(prev.at<Vec3f>(r - 1, c).val[0]) + abs(prev.at<Vec3f>(r - 1, c).val[1]) + abs(prev.at<Vec3f>(r - 1, c).val[2])) / 3 * cross_deriv_scale;

		Matx33f H(dxx, dxy, dxs,
			dxy, dyy, dys,
			dxs, dys, dss);

		Vec3f X = H.solve(dD, DECOMP_LU);

		xi = -X[2];
		xr = -X[1];
		xc = -X[0];

		if (std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f)
			break;

		if (std::abs(xi) > (float)(INT_MAX / 3) ||
			std::abs(xr) > (float)(INT_MAX / 3) ||
			std::abs(xc) > (float)(INT_MAX / 3))
			return false;

		c += cvRound(xc);
		r += cvRound(xr);
		layer += cvRound(xi);

		if (layer < 1 || layer > nOctaveLayers ||
			c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER ||
			r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER)
			return false;
	}

	// ensure convergence of interpolation
	if (i >= SIFT_MAX_INTERP_STEPS)
		return false;

	{
		int idx = octv * (nOctaveLayers + 2) + layer;
		const Mat& img = dog_pyr[idx];
		const Mat& prev = dog_pyr[idx - 1];
		const Mat& next = dog_pyr[idx + 1];
		sift_wt dD_val1 = (abs(img.at<Vec3f>(r, c + 1).val[0]) - abs(img.at<Vec3f>(r, c - 1).val[0])
						+ abs(img.at<Vec3f>(r, c + 1).val[1]) - abs(img.at<Vec3f>(r, c - 1).val[1])
						+ abs(img.at<Vec3f>(r, c + 1).val[2]) - abs(img.at<Vec3f>(r, c - 1).val[2])) / 3;
		sift_wt dD_val2 = (abs(img.at<Vec3f>(r + 1, c).val[0]) - abs(img.at<Vec3f>(r - 1, c).val[0])
						+ abs(img.at<Vec3f>(r + 1, c).val[1]) - abs(img.at<Vec3f>(r - 1, c).val[1])
						+ abs(img.at<Vec3f>(r + 1, c).val[2]) - abs(img.at<Vec3f>(r - 1, c).val[2])) / 3;
		sift_wt dD_val3 = (abs(next.at<Vec3f>(r, c).val[0]) - abs(prev.at<Vec3f>(r, c).val[0])
						+ abs(next.at<Vec3f>(r, c).val[1]) - abs(prev.at<Vec3f>(r, c).val[1])
						+ abs(next.at<Vec3f>(r, c).val[2]) - abs(prev.at<Vec3f>(r, c).val[2])) / 3;

		Matx31f dD(dD_val1 * deriv_scale, dD_val2*deriv_scale, dD_val3*deriv_scale);
		float t = dD.dot(Matx31f(xc, xr, xi));
		//
		contr = (abs(img.at<Vec3f>(r, c).val[0]) + abs(img.at<Vec3f>(r, c).val[1]) + abs(img.at<Vec3f>(r, c).val[2])) / 3 * img_scale + t * 0.5f;
		if (std::abs(contr) * nOctaveLayers < contrastThreshold)
			return false;

		// principal curvatures are computed using the trace and det of Hessian
		float v2 = (abs(img.at<Vec3f>(r, c).val[0]) + abs(img.at<Vec3f>(r, c).val[1]) + abs(img.at<Vec3f>(r, c).val[2])) / 3 * 2.f;
		//
		float dxx = ((abs(img.at<Vec3f>(r, c + 1).val[0]) + abs(img.at<Vec3f>(r, c + 1).val[1]) + abs(img.at<Vec3f>(r, c + 1).val[2])
				  + abs(img.at<Vec3f>(r, c - 1).val[0]) + abs(img.at<Vec3f>(r, c - 1).val[1]) + abs(img.at<Vec3f>(r, c - 1).val[2])) / 3
				  - v2) * second_deriv_scale;
		//
		float dyy = ((abs(img.at<Vec3f>(r + 1, c).val[0]) + abs(img.at<Vec3f>(r + 1, c).val[1]) + abs(img.at<Vec3f>(r + 1, c).val[2])
				  + abs(img.at<Vec3f>(r - 1, c).val[0]) + abs(img.at<Vec3f>(r - 1, c).val[1]) + abs(img.at<Vec3f>(r - 1, c).val[2])) / 3
				  - v2) * second_deriv_scale;
		//
		float dxy = (abs(img.at<Vec3f>(r + 1, c + 1).val[0]) + abs(img.at<Vec3f>(r + 1, c + 1).val[1]) + abs(img.at<Vec3f>(r + 1, c + 1).val[2])
				  - abs(img.at<Vec3f>(r + 1, c - 1).val[0]) - abs(img.at<Vec3f>(r + 1, c - 1).val[1]) - abs(img.at<Vec3f>(r + 1, c - 1).val[2])
				  - abs(img.at<Vec3f>(r - 1, c + 1).val[0]) - abs(img.at<Vec3f>(r - 1, c + 1).val[1]) - abs(img.at<Vec3f>(r - 1, c + 1).val[2])
				  + abs(img.at<Vec3f>(r - 1, c - 1).val[0]) + abs(img.at<Vec3f>(r - 1, c - 1).val[1]) + abs(img.at<Vec3f>(r - 1, c - 1).val[2])) / 3 * cross_deriv_scale;

		float tr = dxx + dyy;
		float det = dxx * dyy - dxy * dxy;

		if (det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det)
			return false;
	}

	kpt.pt.x = (c + xc) * (1 << octv);
	kpt.pt.y = (r + xr) * (1 << octv);
	kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5) * 255) << 16);
	kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv) * 2;
	kpt.response = std::abs(contr);

	return true;
}

//------------------------------------calcOrientationHist()----------------------------
// Computes a gradient orientation histogram at a specified pixel using color info
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
float HoNC::calcOrientationHist(const Mat& img, Point pt, int radius,
	float sigma, float* hist, int n)
{
	int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1);

	float expf_scale = -1.f / (2.f * sigma * sigma);
	AutoBuffer<float> buf(len * 12 + n + 4);
	float *X = buf, *Y = X + 3 * len, *Mag = X, *Ori = Y + 3 * len, *W = Ori + 3 * len;
	float* temphist = W + 3 * len + 2;

	for (i = 0; i < n; i++)
		temphist[i] = 0.f;

	for (i = -radius, k = 0; i <= radius; i++)
	{
		int y = pt.y + i;
		if (y <= 0 || y >= img.rows - 1)
			continue;
		for (j = -radius; j <= radius; j++)
		{
			int x = pt.x + j;
			if (x <= 0 || x >= img.cols - 1)
				continue;
			//compute derivative using average of RGB
			/*
			float dx = (float)(abs(img.at<Vec3f>(y, x + 1)[0]) + abs(img.at<Vec3f>(y, x + 1)[1]) + abs(img.at<Vec3f>(y, x + 1)[2])
			- abs(img.at<Vec3f>(y, x - 1)[0]) - abs(img.at<Vec3f>(y, x - 1)[1]) - abs(img.at<Vec3f>(y, x - 1)[2])) / 3;
			float dy = (float)(abs(img.at<Vec3f>(y - 1, x)[0]) + abs(img.at<Vec3f>(y - 1, x)[1]) + abs(img.at<Vec3f>(y - 1, x)[2])
			- abs(img.at<Vec3f>(y + 1, x)[0]) - abs(img.at<Vec3f>(y + 1, x)[1]) - abs(img.at<Vec3f>(y + 1, x)[2])) / 3;

			X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
			k++;
			*/
			float dx0 = (float)(img.at<Vec3f>(y, x + 1)[0] - img.at<Vec3f>(y, x - 1)[0]);
			float dx1 = (float)(img.at<Vec3f>(y, x + 1)[1] - img.at<Vec3f>(y, x - 1)[1]);
			float dx2 = (float)(img.at<Vec3f>(y, x + 1)[2] - img.at<Vec3f>(y, x - 1)[2]);
			float dy0 = (float)(img.at<Vec3f>(y - 1, x)[0] - img.at<Vec3f>(y + 1, x)[0]);
			float dy1 = (float)(img.at<Vec3f>(y - 1, x)[1] - img.at<Vec3f>(y + 1, x)[1]);
			float dy2 = (float)(img.at<Vec3f>(y - 1, x)[2] - img.at<Vec3f>(y + 1, x)[2]);
			W[k] = (i*i + j*j)*expf_scale;
			X[k] = dx0; Y[k] = dy0;
			k++;
			W[k] = (i*i + j*j)*expf_scale;
			X[k] = dx1; Y[k] = dy1;
			k++;
			W[k] = (i*i + j*j)*expf_scale;
			X[k] = dx2; Y[k] = dy2;
			k++;
		}
	}

	len = k;

	// compute gradient values, orientations and the weights over the pixel neighborhood
	hal::exp(W, W, len);
	hal::fastAtan2(Y, X, Ori, len, true);
	hal::magnitude(X, Y, Mag, len);

	for (k = 0; k < len; k++)
	{
		int bin = cvRound((n / 360.f)*Ori[k]);
		if (bin >= n)
			bin -= n;
		if (bin < 0)
			bin += n;
		temphist[bin] += W[k] * Mag[k];
	}

	// smooth the histogram
	temphist[-1] = temphist[n - 1];
	temphist[-2] = temphist[n - 2];
	temphist[n] = temphist[0];
	temphist[n + 1] = temphist[1];
	for (i = 0; i < n; i++)
	{
		hist[i] = (temphist[i - 2] + temphist[i + 2])*(1.f / 16.f) +
			(temphist[i - 1] + temphist[i + 1])*(4.f / 16.f) +
			temphist[i] * (6.f / 16.f);
	}

	float maxval = hist[0];
	for (i = 1; i < n; i++)
		maxval = std::max(maxval, hist[i]);

	return maxval;
}


//------------------------------------descriptorSize()---------------------------------
// ! returns the descriptor size in floats
//Precondition: None
//Postcondition: the descriptor size is returned in floats
//-------------------------------------------------------------------------------------
int HoNC::descriptorSize() const
{
	return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}