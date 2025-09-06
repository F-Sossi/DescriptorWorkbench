//-------------------------------------------------------------------------
// Name: VanillaSIFT.h
// Author: Clark Olson, Jordan Soltman, Sam Hoover Siqi Zhang
//		This is mostly previous code from Rob Hess / OpenCV
// Description: vanillaSIFT descriptor class, inherited from Feature2D
//  This class is used as a basic class for other feature representation.
// Methods:
//			create()
//			VanillaSIFT()
//			operator()
//			descriptorSize()
//			descriptorType()
//			compute()
//			buildGaussianPyramid()
//			buildDoGPyramid()
//			findScaleSpaceExtrema()
//			calcDescriptors()
//			calcSIFTDescriptor()
//			createInitialImage()
//			detectImpl()
//			compteImpl()
//			calcOrientationHist()
//			adjustLocalExtrema()
//			unpackOctave()
//-------------------------------------------------------------------------

/**********************************************************************************************\
 Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/sift/
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

#ifndef VANILLASIFT_H
#define VANILLASIFT_H

#include "opencv4/opencv2/features2d.hpp"
#include "opencv4/opencv2/opencv_modules.hpp"
#include "opencv4/opencv2/imgproc/imgproc.hpp"
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/features2d/features2d.hpp"  //3.0 version
// #include "opencv2/features2d/features2d.hpp"
// #include "opencv2/opencv.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/hal/hal.hpp"
#include <algorithm>
#include <cstdarg>
#include <iostream>
#include <opencv2/core/core_c.h>  // CV_STSBadArg

using namespace std;
using namespace cv;
#ifdef __cplusplus

namespace cv {

	class CV_EXPORTS_W VanillaSIFT : public Feature2D {

	public:
		static const int SIFT_DESCR_WIDTH = 4;			// default width of descriptor histogram array
		static const int SIFT_DESCR_HIST_BINS = 8;		// default number of bins per histogram in descriptor array
		static const float SIFT_INIT_SIGMA;				// assumed gaussian blur for input image
		static const int SIFT_IMG_BORDER = 5;			// width of border in which to ignore keypoints
		static const int SIFT_MAX_INTERP_STEPS = 5;		// maximum steps of keypoint interpolation before failure
		static const int SIFT_ORI_HIST_BINS = 36;		// default number of bins in histogram for orientation assignment
		static const float SIFT_ORI_SIG_FCTR;			// determines gaussian sigma for orientation assignment
		static const float SIFT_ORI_RADIUS;				// determines the radius of the region used in orientation assignment	
		static const float SIFT_ORI_PEAK_RATIO;			// orientation magnitude relative to max that results in new feature
		static const float SIFT_DESCR_SCL_FCTR;			// determines the size of a single descriptor orientation histogram		
		static const float SIFT_DESCR_MAG_THR;			// threshold on magnitude of elements of descriptor vector	
		static const float SIFT_INT_DESCR_FCTR;			// factor used to convert floating-point descriptor to unsigned char
	
		// intermediate type used for DoG pyramids
		typedef float sift_wt;
		static const int SIFT_FIXPT_SCALE = 1;

//------------------------------------create()-----------------------------------------
// create a pointer to the VanillaSIFT object
//Precondition: None
//Postcondition: the pointer is created
//-------------------------------------------------------------------------------------
		CV_WRAP static Ptr<VanillaSIFT> create(int _nfeatures = 0, int _nOctaveLayers = 3, double _contrastThreshold = 0.04, double _edgeThreshold = 10, double _sigma = 1.6) {
			return makePtr<VanillaSIFT>(_nfeatures, _nOctaveLayers, _contrastThreshold, _edgeThreshold, _sigma);
		}

//------------------------------------VanillaSIFT()------------------------------------
// VanillaSIFT constructor, initialize variables
//Precondition: the following parameters must be correctly defined.
//parameters:
	//nfeatures: 0 by default
	//nOctaveLayers: 3 by default
	//contrastThreshold: 0.04 by default
	//edgeThreshold: 10 by default
	//sigma: 1.6 by default
//Postcondition: variables are assigned
//-------------------------------------------------------------------------------------
		CV_WRAP explicit VanillaSIFT(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04, double edgeThreshold = 10, double sigma = 1.6);

//------------------------------------descriptorSize()---------------------------------
// ! returns the descriptor size in floats
//Precondition: None
//Postcondition: the descriptor size is returned in floats
//-------------------------------------------------------------------------------------
		CV_WRAP virtual int descriptorSize() const;

//------------------------------------descriptorType()---------------------------------
//! returns the descriptor type
//Precondition: None
//Postcondition: the descriptor type is returned
//-------------------------------------------------------------------------------------
		CV_WRAP virtual int descriptorType() const;

//------------------------------------operator()---------------------------------------
// Overloading operator() to run the SIFT algorithm :
// 1. compute keypoints using local extrema of Dog space
// 2. compute descriptors with keypoints
//Precondition: the following parameters must be correctly defined.
//parameters:
	//img: color image base
	//mask: image mask
	//keypoints: keypoints of the image
//Postcondition: keypoints are assigned
//-------------------------------------------------------------------------------------
		virtual void operator()(InputArray img, InputArray mask, vector<KeyPoint>& keypoints) const;

//virtual void detectAndCompute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors, bool useProvidedKeypoints=false) const;

        virtual void detectAndCompute( InputArray image, InputArray mask,
                                       CV_OUT std::vector<KeyPoint>& keypoints,
                                       OutputArray descriptors,
                                       bool useProvidedKeypoints);
//------------------------------------operator()---------------------------------------
// Overloading operator() to run the SIFT algorithm :
// 1. compute keypoints using local extrema of Dog space
// 2. compute descriptors with keypoints
//Precondition: the following parameters must be correctly defined.
//parameters:
	//img: image base
	//mask: image mask
	//keypoints: keypoints of the image
	//descriptors: descriptors
	//useProvidedKeypoints: bool indicating whether using provided keypoints, false by default
//Postcondition: 1. keypoints are assigned
//				 2. descriptors are calculated and assigned
//-------------------------------------------------------------------------------------
		virtual void operator()(InputArray img, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints = false) const;

//------------------------------------compute()----------------------------------------
// compute the descriptors with keypoints
//Precondition: the following parameters must be correctly defined.
//parameters:
	//image: image base
	//keypoints: calculated keypoints
	//descriptors: descriptors to be computed
//Postcondition: descriptors are calculated and assigned
//-------------------------------------------------------------------------------------
		virtual void compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors);

//------------------------------------buildGaussianPyramid()---------------------------
// compute Gaussian pyramid using base image
//Precondition: the following parameters must be correctly defined.
//parameters:
	//base: image base
	//pyr: Mat vector to be assigned with gaussian blurred image
	//nOctaves: number of octaves
//Postcondition: images are blurred and assigned to pyr
//-------------------------------------------------------------------------------------
		void buildGaussianPyramid(const Mat& base, std::vector<Mat>& pyr, int nOctaves) const;
		
//------------------------------------buildDoGPyramid()--------------------------------
// compute diffierence of Gaussian pyramid using Gaussian pyramid
//Precondition: the following parameters must be correctly defined.
//parameters:
	//pyr: gaussian pyramid
	//dogpyr: Mat array to be assigned with difference of Gaussian
//Postcondition: difference of Gaussian images are assigned to dogpyr
//-------------------------------------------------------------------------------------
		void buildDoGPyramid(const std::vector<Mat>& pyr, std::vector<Mat>& dogpyr) const;

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
		virtual void findScaleSpaceExtrema(const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr, std::vector<KeyPoint>& keypoints) const;

	protected:

//------------------------------------calcDescriptors()--------------------------------
// set up variables and call calcSIFTDescriptor() to compute descriptors
//Precondition: the following parameters must be correctly defined.
//parameters:
	//gpyr: gaussian pyramid
	//keypoints: computed keypoints
	//descriptors: empty descriptors vector
	//nOctaveLayers: number of octave layers
	//firstOctave: index of first octave
//Postcondition: descriptors are assigned
//-------------------------------------------------------------------------------------
		virtual void calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints, Mat& descriptors, int nOctaveLayers, int firstOctave) const;
		
//------------------------------------calcSIFTDescriptor()-----------------------------
// compute SIFT descriptor and perform normalization for one keypoint
//Precondition: the following parameters must be correctly defined.
//parameters:
	//img: color image
	//ptf: keypoint
	//ori: angle(degree) of the keypoint relative to the coordinates, clockwise
	//scl: radius of meaningful neighborhood around the keypoint 
	//d: newsift descr_width, 4 in this case
	//n: newsift_descr_hist_bins, 8 in this case
	//dst: descriptor array to pass in
//Postcondition: descriptors are assigned to dst
//-------------------------------------------------------------------------------------
		virtual void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const;

//------------------------------------createInitialImage()-----------------------------
//create initial grey-scale base image for later process
//Precondition: the following parameters must be correctly defined.
//parameters:
	//img: color image
	//doubleImageSize: bool indicating whether doubling the image 
	//sigma: gaussian blur coefficient
//Postcondition: image is assigned
//-------------------------------------------------------------------------------------
		virtual Mat createInitialImage(const Mat& img, bool doubleImageSize, float sigma) const;

//------------------------------------detectImpl()-------------------------------------
//only detect keypoints without computing descriptors
//Precondition: the following parameters must be correctly defined.
//parameters:
	//image: color image
	//keypoints: empty vector to be filled
	//mask: image mask
//Postcondition: keypoints are detected and assigned
//-------------------------------------------------------------------------------------
		void detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask = Mat()) const;
		
//------------------------------------computeImpl()------------------------------------
// call vanillaSIFT constructor and compute descriptors
//Precondition: the following parameters must be correctly defined.
//parameters:
	//image: color image
	//keypoints: empty vector to be filled
	//descriptors: empty vector to be filled
//Postcondition: keypoints are detected and descriptors are assigned
//-------------------------------------------------------------------------------------
		void computeImpl(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const;


		
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
	
//------------------------------------unpackOctave()-----------------------------------
// calculate octave related data
//Precondition: the following parameters must be correctly defined.
//parameters:
	//kpt: keypoint in image
	//octave: octave numbers
	//layer: location of the keypoint among layers
	//scale: 
//Postcondition: octave, layer and scale are computed
//-------------------------------------------------------------------------------------
		static inline void unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale) {
			octave = kpt.octave & 255;
			layer = (kpt.octave >> 8) & 255;
			octave = octave < 128 ? octave : (-128 | octave);
			scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
		}

		// variables for building Dog
		CV_PROP_RW int nfeatures;
		CV_PROP_RW int nOctaveLayers;
		CV_PROP_RW double contrastThreshold;
		CV_PROP_RW double edgeThreshold;
		CV_PROP_RW double sigma;
	};

} // namespace cv

#endif /* __cplusplus */

#endif 