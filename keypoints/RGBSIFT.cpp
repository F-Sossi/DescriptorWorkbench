
/**********************************************************************************************\
Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/SIFT/
Below is the original copyright.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

//    The following patent has been issued for methods embodied in this
//    software: "Method and apparatus for identifying scale invariant features
//    in an image and use of same for locating an object in an image," David
//    G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
//    filed March 8, 1999. Asignee: The University of British Columbia. For
//    further details, contact David Lowe (lowe@cs.ubc.ca) or the
//    University-Industry Liaison Office of the University of British
//    Columbia.

//    Note that restrictions imposed by this patent (and possibly others)
//    exist independently of and may be in conflict with the freedoms granted
//    in this license, which refers to copyright of the program, not patents
//    for any methods that it implements.  Both copyright and patent law must
//    be obeyed to legally use and redistribute this program and it is not the
//    purpose of this license to induce you to infringe any patents or other
//    property right claims or to contest validity of any such claims.  If you
//    redistribute or use the program, then this license merely protects you
//    from committing copyright infringement.  It does not protect you from
//    committing patent infringement.  So, before you do anything with this
//    program, make sure that you have permission to do so not merely in terms
//    of copyright, but also in terms of patent law.

//    Please note that this license is not to be understood as a guarantee
//    either.  If you use the program according to this license, but in
//    conflict with patent law, it does not mean that the licensor will refund
//    you for any losses that you incur if you are sued for your patent
//    infringement.

//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright and
//          patent notices, this list of conditions and the following
//          disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in
//          the documentation and/or other materials provided with the
//          distribution.
//        * Neither the name of Oregon State University nor the names of its
//          contributors may be used to endorse or promote products derived
//          from this software without specific prior written permission.

//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************************/


#include "RGBSIFT.h"

namespace cv
{

	// initialize color base image for calculating color gaussian pyramid
	Mat RGBSIFT::createInitialColorImage(const Mat& img, bool doubleImageSize, float sigma) const
	{
		Mat colorImg = img, color_fpt;
		img.convertTo(color_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
		Mat output;

		float sig_diff;
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




	//-------------------------------------------------------------------------------------
	//img: color image
	//ptf: keypoint
	//ori: angle(degree) of the keypoint relative to the coordinates, clockwise
	//scl: radius of meaningful neighborhood around the keypoint 
	//d: newsift descr_width, 4 in this case
	//n: SIFT_descr_hist_bins, 8 in this case
	//dst: descriptor array to pass in
	//changes: 1. img now is a color image
	void RGBSIFT::calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl,
		int d, int n, float* dst) const
	{

		Point pt(cvRound(ptf.x), cvRound(ptf.y));
		float cos_t = cosf(ori * (float)(CV_PI / 180));
		float sin_t = sinf(ori * (float)(CV_PI / 180));
		float bins_per_rad = n / 360.f;
		float exp_scale = -1.f / (d * d * 0.5f);
		float hist_width = SIFT_DESCR_SCL_FCTR * scl;
		int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
		// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
		radius = std::min(radius, (int)sqrt((double)img.cols * img.cols + img.rows * img.rows));
		cos_t /= hist_width;
		sin_t /= hist_width;

		//int i, j, k, len = (radius * 2 + 1) * (radius * 2 + 1), histlen = (d + 2) * (d + 2) * (n + 2);
        // TODO: This line above replaced by the following lines to avoid integer overflow
        int i, j, k;
        int64_t len = static_cast<int64_t>(radius * 2 + 1) * static_cast<int64_t>(radius * 2 + 1);
        int64_t histlen = static_cast<int64_t>(d + 2) * static_cast<int64_t>(d + 2) * static_cast<int64_t>(n + 2);

        // Check for integer overflow
        if (len > std::numeric_limits<int>::max() || histlen > std::numeric_limits<int>::max()) {
            // Handle the overflow case, e.g., throw an exception or return an error code
            // ...
            return;
        }

		int rows = img.rows, cols = img.cols;


		//AutoBuffer<float> buf(len * 12 + histlen * 3);
        // TODO: This line above replaced by the following lines to avoid integer overflow
        AutoBuffer<float> buf(static_cast<size_t>(len) * 12 + static_cast<size_t>(histlen) * 3);

		float *X1 = buf, *Y1 = X1 + len, *X2 = Y1 + len, *Y2 = X2 + len, *X3 = Y2 + len, *Y3 = X3 + len;
		float *Mag1 = Y1, *Mag2 = Y2, *Mag3 = Y3, *Ori1 = Mag3 + len, *Ori2 = Ori1 + len, *Ori3 = Ori2 + len, *W = Ori3 + len;
		float *RBin = W + len, *CBin = RBin + len, *hist1 = CBin + len, *hist2 = hist1 + histlen, *hist3 = hist2 + histlen;

		for (i = 0; i < d + 2; i++)
		{
			for (j = 0; j < d + 2; j++) {
				for (k = 0; k < n + 2; k++) {
					hist1[(i*(d + 2) + j)*(n + 2) + k] = 0.;
					hist2[(i*(d + 2) + j)*(n + 2) + k] = 0.;
					hist3[(i*(d + 2) + j)*(n + 2) + k] = 0.;
				}
			}
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
					X1[k] = (img.at<Vec3f>(r, c + 1)[0] - img.at<Vec3f>(r, c - 1)[0]);
					Y1[k] = (img.at<Vec3f>(r - 1, c)[0] - img.at<Vec3f>(r + 1, c)[0]);
					X2[k] = (img.at<Vec3f>(r, c + 1)[1] - img.at<Vec3f>(r, c - 1)[1]);
					Y2[k] = (img.at<Vec3f>(r - 1, c)[1] - img.at<Vec3f>(r + 1, c)[1]);
					X3[k] = (img.at<Vec3f>(r, c + 1)[2] - img.at<Vec3f>(r, c - 1)[2]);
					Y3[k] = (img.at<Vec3f>(r - 1, c)[2] - img.at<Vec3f>(r + 1, c)[2]);
					RBin[k] = rbin; CBin[k] = cbin;
					W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
					k++;
				}
			}

		len = k;
		hal::fastAtan2(Y1, X1, Ori1, len, true);
		hal::magnitude(X1, Y1, Mag1, len);
		hal::fastAtan2(Y2, X2, Ori2, len, true);
		hal::magnitude(X2, Y2, Mag2, len);
		hal::fastAtan2(Y3, X3, Ori3, len, true);
		hal::magnitude(X3, Y3, Mag3, len);
		hal::exp(W, W, len);

		for (k = 0; k < len; k++)
		{
			float rbin = RBin[k], cbin = CBin[k];
			float obin1 = (Ori1[k] - ori)*bins_per_rad;
			float mag1 = Mag1[k] * W[k];
			float obin2 = (Ori2[k] - ori)*bins_per_rad;
			float mag2 = Mag2[k] * W[k];
			float obin3 = (Ori3[k] - ori)*bins_per_rad;
			float mag3 = Mag3[k] * W[k];

			int r0 = cvFloor(rbin);
			int c0 = cvFloor(cbin);
			int o01 = cvFloor(obin1);
			int o02 = cvFloor(obin2);
			int o03 = cvFloor(obin3);
			rbin -= r0;
			cbin -= c0;
			obin1 -= o01;
			obin2 -= o02;
			obin3 -= o03;

			if (o01 < 0)
				o01 += n;
			if (o01 >= n)
				o01 -= n;
			if (o02 < 0)
				o02 += n;
			if (o02 >= n)
				o02 -= n;
			if (o03 < 0)
				o03 += n;
			if (o03 >= n)
				o03 -= n;

			// histogram update using tri-linear interpolation
			float v1_r1 = mag1*rbin, v1_r0 = mag1 - v1_r1;
			float v1_rc11 = v1_r1*cbin, v1_rc10 = v1_r1 - v1_rc11;
			float v1_rc01 = v1_r0*cbin, v1_rc00 = v1_r0 - v1_rc01;
			float v1_rco111 = v1_rc11*obin1, v1_rco110 = v1_rc11 - v1_rco111;
			float v1_rco101 = v1_rc10*obin1, v1_rco100 = v1_rc10 - v1_rco101;
			float v1_rco011 = v1_rc01*obin1, v1_rco010 = v1_rc01 - v1_rco011;
			float v1_rco001 = v1_rc00*obin1, v1_rco000 = v1_rc00 - v1_rco001;

			float v2_r1 = mag2*rbin, v2_r0 = mag2 - v2_r1;
			float v2_rc11 = v2_r1*cbin, v2_rc10 = v2_r1 - v2_rc11;
			float v2_rc01 = v2_r0*cbin, v2_rc00 = v2_r0 - v2_rc01;
			float v2_rco111 = v2_rc11*obin2, v2_rco110 = v2_rc11 - v2_rco111;
			float v2_rco101 = v2_rc10*obin2, v2_rco100 = v2_rc10 - v2_rco101;
			float v2_rco011 = v2_rc01*obin2, v2_rco010 = v2_rc01 - v2_rco011;
			float v2_rco001 = v2_rc00*obin2, v2_rco000 = v2_rc00 - v2_rco001;

			float v3_r1 = mag3*rbin, v3_r0 = mag3 - v3_r1;
			float v3_rc11 = v3_r1*cbin, v3_rc10 = v3_r1 - v3_rc11;
			float v3_rc01 = v3_r0*cbin, v3_rc00 = v3_r0 - v3_rc01;
			float v3_rco111 = v3_rc11*obin3, v3_rco110 = v3_rc11 - v3_rco111;
			float v3_rco101 = v3_rc10*obin3, v3_rco100 = v3_rc10 - v3_rco101;
			float v3_rco011 = v3_rc01*obin3, v3_rco010 = v3_rc01 - v3_rco011;
			float v3_rco001 = v3_rc00*obin3, v3_rco000 = v3_rc00 - v3_rco001;

			int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o01;
			hist1[idx] += v1_rco000;
			hist1[idx + 1] += v1_rco001;
			hist1[idx + (n + 2)] += v1_rco010;
			hist1[idx + (n + 3)] += v1_rco011;
			hist1[idx + (d + 2)*(n + 2)] += v1_rco100;
			hist1[idx + (d + 2)*(n + 2) + 1] += v1_rco101;
			hist1[idx + (d + 3)*(n + 2)] += v1_rco110;
			hist1[idx + (d + 3)*(n + 2) + 1] += v1_rco111;
			idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o02;
			hist2[idx] += v2_rco000;
			hist2[idx + 1] += v2_rco001;
			hist2[idx + (n + 2)] += v2_rco010;
			hist2[idx + (n + 3)] += v2_rco011;
			hist2[idx + (d + 2)*(n + 2)] += v2_rco100;
			hist2[idx + (d + 2)*(n + 2) + 1] += v2_rco101;
			hist2[idx + (d + 3)*(n + 2)] += v2_rco110;
			hist2[idx + (d + 3)*(n + 2) + 1] += v2_rco111;
			idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o03;
			hist3[idx] += v3_rco000;
			hist3[idx + 1] += v3_rco001;
			hist3[idx + (n + 2)] += v3_rco010;
			hist3[idx + (n + 3)] += v3_rco011;
			hist3[idx + (d + 2)*(n + 2)] += v3_rco100;
			hist3[idx + (d + 2)*(n + 2) + 1] += v3_rco101;
			hist3[idx + (d + 3)*(n + 2)] += v3_rco110;
			hist3[idx + (d + 3)*(n + 2) + 1] += v3_rco111;
		}

		// finalize histogram, since the orientation histograms are circular
		for (i = 0; i < d; i++)
			for (j = 0; j < d; j++)
			{
				int idx = ((i + 1)*(d + 2) + (j + 1))*(n + 2);
				hist1[idx] += hist1[idx + n];
				hist1[idx + 1] += hist1[idx + n + 1];
				hist2[idx] += hist2[idx + n];
				hist2[idx + 1] += hist2[idx + n + 1];
				hist3[idx] += hist3[idx + n];
				hist3[idx + 1] += hist3[idx + n + 1];
				for (k = 0; k < n; k++) {
					dst[(i*d + j)*n + k] = hist1[idx + k];
					dst[(i*d + j)*n + k + d * d * n] = hist2[idx + k];
					dst[(i*d + j)*n + k + d * d * n * 2] = hist3[idx + k];
				}
			}
		// copy histogram to the descriptor,
		// apply hysteresis thresholding
		// and scale the result, so that it can be easily converted
		// to byte array
		normalizeHistogram(dst, d, n);
	}


	void RGBSIFT::normalizeHistogram(float *dst, int d, int n) const {
		float nrm1sqr = 0, nrm2sqr = 0, nrm3sqr = 0;
		int len = d*d*n;
		for (int k = 0; k < len; k++) {
			nrm1sqr += dst[k] * dst[k];
			nrm2sqr += dst[k + len] * dst[k + len];
			nrm3sqr += dst[k + 2*len] * dst[k + 2*len];
		}
		float thr1 = std::sqrt(nrm1sqr)*SIFT_DESCR_MAG_THR;
		float thr2 = std::sqrt(nrm2sqr)*SIFT_DESCR_MAG_THR;
		float thr3 = std::sqrt(nrm3sqr)*SIFT_DESCR_MAG_THR;
		nrm1sqr = nrm2sqr = nrm3sqr = 0;
		for (int k = 0; k < len; k++)
		{
			float val = std::min(dst[k], thr1);
			dst[k] = val;
			nrm1sqr += val*val;
			val = std::min(dst[k + len], thr2);
			dst[k + len] = val;
			nrm2sqr += val*val;
			val = std::min(dst[k + 2*len], thr3);
			dst[k + 2*len] = val;
			nrm3sqr += val*val;
		}

		// Factor of three added below to make vector lengths comparable with SIFT
		nrm1sqr = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(3*nrm1sqr), FLT_EPSILON);
		nrm2sqr = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(3*nrm2sqr), FLT_EPSILON);
		nrm3sqr = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(3*nrm3sqr), FLT_EPSILON);
		for (int k = 0; k < len; k++)
		{
			dst[k] = (dst[k] * nrm1sqr);
			dst[k + len] = (dst[k + len] * nrm2sqr);
			dst[k + 2*len] = (dst[k + 2*len] * nrm3sqr);
		}

	}


	//////////////////////////////////////////////////////////////////////////////////////////

	RGBSIFT::RGBSIFT()
	{
	}

	int RGBSIFT::descriptorSize() const
	{
		return 3 * SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS;
	}

	void RGBSIFT::operator()(InputArray _image, InputArray _mask,
		vector<KeyPoint>& keypoints,
		OutputArray _descriptors,
		bool useProvidedKeypoints) const
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

		//double t, tf = getTickFrequency();
		//t = (double)getTickCount();
		buildGaussianPyramid(base, gpyr, nOctaves);
		buildDoGPyramid(gpyr, dogpyr);
		// build color gaussian pyramid
		buildGaussianPyramid(colorBase, colorGpyr, nOctaves);
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
			calcDescriptors(colorGpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
			//t = (double)getTickCount() - t;
			//printf("descriptor extraction time: %g\n", t*1000./tf);
		}
	}

	//same as the operator() above, but it also takes the parameters needed
	//for domain-size pooling.
	void RGBSIFT::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints, OutputArray _descriptors,
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

}