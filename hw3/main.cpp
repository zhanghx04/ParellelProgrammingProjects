#include <iostream>

#include <stdio.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <chrono>
#include <vector>
#include <cfloat>
#include <errno.h>
#include <math.h>
#include <ios>

// #include <sys/types.h>
// #include <sys/stat.h>
// #include <sys/wait.h>
// #include <sys/ipc.h>
// #include <sys/sem.h>
// #include <sys/shm.h>
// #include <unistd.h>
// #include <cstdlib>

#include <boost/thread/mutex.hpp>
#include <boost/utility.hpp> // for noncopyable
#include <boost/thread/thread.hpp>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>

#include "DistanceThreader.hpp"

// #include <SLIC.h>

using namespace zhang;

// ######################################################
// Templates
// ######################################################
template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n){
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;
 
    std::vector<T> vec(first, last);
    return vec;
}

template<typename T>
void print(std::vector<T> const &v){
    for (auto i : v)
        std::cout << i << ' ';
    std::cout << std::endl;
}

// ######################################################
// Prototypes
// ######################################################

void SLIC_PerformSuperpixel(std::vector<double>& kseedsl, 
                            std::vector<double>& kseedsa, 
                            std::vector<double>& kseedsb,
                            std::vector<double>& kseedsx,
                            std::vector<double>& kseedsy,
                                            int* klabels,
                                      const int& STEP,
                                   const double& M,
                                             int hight,
                                             int width,
                                         double* m_lvec,
                                         double* m_avec,
                                         double* m_bvec,
                                             int p);

void GetLABXYSeeds(
    std::vector<double>&		kseedsl,
	std::vector<double>&		kseedsa,
	std::vector<double>&		kseedsb,
	std::vector<double>&		kseedsx,
	std::vector<double>&		kseedsy,
    const int&					STEP,
    int                         height,        
    int                         width,
    double*                     m_lvec,
    double*                     m_avec,
    double*                     m_bvec);

void DrawContoursAroundSegments(
	cv::Mat&	    		image,
	int *	            	labels,
	const int&				width,
	const int&				height);


// ==========================
//  Main
// ==========================
int main (int argc, char** argv) 
{

    
    // ------------------------------------------
    // Check four inputs
    // ------------------------------------------
    if (argc != 5) 
    {
        std::cout << "[Error] Input parameters missing, need four inputs:" << std::endl
                    << "    1. The input file (String)" << std::endl
                    << "    2. The level of parallelism (int)" << std::endl
                    << "    3. Count of target superpixels (int)" << std::endl
                    << "    4. The output file name (String)" << std::endl;
        return -1;
    }

    // Store inputs
    std::string input_filename, output_filename;
    int lvl_para, k_superpixel;

    input_filename  = argv[1];              // input image path
    lvl_para        = std::stoi(argv[2]);   // number of processes to process the dataset
    k_superpixel    = std::stoi(argv[3]);   // the count of target superpixels. eg. 200
    output_filename = argv[4];

    std::cout << "[Info] Four inputs:" << std::endl
                << "    1.           The input file (String) --> " << input_filename << std::endl
                << "    2.    The level of parallelism (int) --> " << lvl_para << std::endl
                << "    3. Count of target superpixels (int) --> " << k_superpixel << std::endl
                << "    4.     The output file name (String) --> " << output_filename << std::endl << std::endl;



    // ------------------------------------------
    // Read the image
    // ------------------------------------------
    cv::Mat img = cv::imread(input_filename, cv::IMREAD_COLOR);

    std::cout << "[Info] Image Size: " << img.rows << " x " << img.cols << " x " << img.channels() << std::endl;

    // convert to lab format
    cv::Mat img_lab(img.size(), CV_8UC3);
    cv::cvtColor(img, img_lab, cv::COLOR_BGR2Lab);



    // ------------------------------------------
    // Prepare Data
    // ------------------------------------------
    // image size
    int height = img.rows;
    int width = img.cols;

    // get each channel
    double lvec [height*width];
    double avec [height*width];
    double bvec [height*width];

    // save the l a b to arrays
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    int i = 0;
    for (int r = 0; r < height; r++) 
    {
        for (int c = 0; c < width; c++) 
        {
            lvec[i] = img_lab.at<cv::Vec3b>(r, c)[0];
            avec[i] = img_lab.at<cv::Vec3b>(r, c)[1];
            bvec[i] = img_lab.at<cv::Vec3b>(r, c)[2];
            i++;
        }
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[Info] Process image using: " << duration.count() << " ms" << std::endl;

    // kseeds
    std::vector<double> kseedsl(0);
    std::vector<double> kseedsa(0);
    std::vector<double> kseedsb(0);
    std::vector<double> kseedsx(0);
    std::vector<double> kseedsy(0);
    
    // parameters for SLIC algorithm
    int*             klabels = NULL; //1, 512x512
    int                    k = k_superpixel;    // superpixel size
    double                 M = 30;     // Compactness factor
    const int superpixelsize = 0.5 + double(width*height)/double(k);
    const int           STEP = sqrt(double(superpixelsize))+0.5;
    
    int p = lvl_para;


    
    
    
    // ------------------------------------------
    // SLIC get kseeds
    // ------------------------------------------
    // GetLABXYSeeds
    GetLABXYSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, STEP, height, width, lvec, avec, bvec);

    
    
//     SLIC_PerformSuperpixel(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, M, height, width, lvec, avec, bvec, lvl_para);
    
    

    
    
    
    // ------------------------------------------
    // SLIC superpixel
    // ------------------------------------------
    const int sz = height * width;
    const int numk = kseedsl.size();
//     numk = kseedsl.size();
    //----------------
    int offset = STEP;
    //if(STEP < 8) offset = STEP*1.5;//to prevent a crash due to a very small step size
    //----------------
    
   
    
    std::vector<double> clustersize(numk, 0);
    std::vector<double> inv(numk, 0);//to store 1/clustersize[k] values
    std::vector<double> sigmal(numk, 0);
    std::vector<double> sigmaa(numk, 0);
    std::vector<double> sigmab(numk, 0);
    std::vector<double> sigmax(numk, 0);
    std::vector<double> sigmay(numk, 0);
    std::vector<double> distvec(sz, DBL_MAX);
    double invwt = 1.0/((STEP/M)*(STEP/M));
//     int x1, y1, x2, y2;
//     double l, a, b;
//     double dist;
//     double distxy;
    klabels = new int[sz];

   
    // create thread
    DistanceThreader mythread(&kseedsl, &kseedsa, &kseedsb, &kseedsx, &kseedsy, &distvec, klabels, lvec, avec, bvec, offset, invwt, width, height, p, numk);
    
    // declare thread group
    boost::thread_group tg;
    
    // tic
    start = std::chrono::high_resolution_clock::now();
    
    
    
    for( int itr = 0; itr < 10; itr++ ) {
        
        std::cout << "[Info] ::Iteration:: " << itr << std::endl;

        
        for (int i =0; i < 100; i++) {
            std::cout << kseedsa[i] << " "; 
        }
        std::cout << std::endl;

        distvec.assign(sz, DBL_MAX);
        

        for (int t = 0; t < p; ++t) {
            tg.add_thread(new boost::thread(boost::ref(mythread), t));
        }
        
        // Block until all threads are completed
        // The print answer
        
        tg.join_all();
        std::cout << "About to block and wait on all my threads." << std::endl;
        

        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------
        //instead of reassigning memory on each iteration, just reset.

        sigmal.assign(numk, 0);
        sigmaa.assign(numk, 0);
        sigmab.assign(numk, 0);
        sigmax.assign(numk, 0);
        sigmay.assign(numk, 0);
        clustersize.assign(numk, 0);
        //------------------------------------
        //edgesum.assign(numk, 0);
        //------------------------------------
        {int ind(0);
        for( int r = 0; r < height; r++ )
        {
            for( int c = 0; c < width; c++ )
            {
                sigmal[klabels[ind]] += lvec[ind];
                sigmaa[klabels[ind]] += avec[ind];
                sigmab[klabels[ind]] += bvec[ind];
                sigmax[klabels[ind]] += c;
                sigmay[klabels[ind]] += r;
                //------------------------------------
                //edgesum[klabels[ind]] += edgemag[ind];
                //------------------------------------
                clustersize[klabels[ind]] += 1.0;
                ind++;
            }
        }}
        {for( int k = 0; k < numk; k++ )
        {
            if( clustersize[k] <= 0 ) clustersize[k] = 1;
            inv[k] = 1.0/clustersize[k];//computing inverse now to multiply, than divide later
        }}

        {for( int k = 0; k < numk; k++ )
        {
            kseedsl[k] = sigmal[k]*inv[k];
            kseedsa[k] = sigmaa[k]*inv[k];
            kseedsb[k] = sigmab[k]*inv[k];
            kseedsx[k] = sigmax[k]*inv[k];
            kseedsy[k] = sigmay[k]*inv[k];
            //------------------------------------
            //edgesum[k] *= inv[k];
            //------------------------------------
        }}
        std::cout << "Calculated Centroid:::" << itr << std::endl;
    }
    

    // toc
    end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "[Info] SLIC using: " << duration.count() << " ms" << std::endl;

    
    

    DrawContoursAroundSegments(img, klabels, width, height);

    cv::imwrite("./" + output_filename + ".png", img);

    std::cout << "[Info] Output image saved in build/" << output_filename << ".png" << std::endl;
    sleep(0.01);
    

    return 0;
}
























// ----------------------------------------------------------------------------------------

//===========================================================================
///	GetLABXYSeeds_ForGivenStepSize
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void GetLABXYSeeds(
	std::vector<double>&		kseedsl,
	std::vector<double>&		kseedsa,
	std::vector<double>&		kseedsb,
	std::vector<double>&		kseedsx,
	std::vector<double>&		kseedsy,
    const int&					STEP,
    int                         m_height,        
    int                         m_width,
    double*                     m_lvec,
    double*                     m_avec,
    double*                     m_bvec)
{
    const bool hexgrid = false;
	int numseeds(0);
	int n(0);

	//int xstrips = m_width/STEP;
	//int ystrips = m_height/STEP;
	int xstrips = (0.5+double(m_width)/double(STEP));
	int ystrips = (0.5+double(m_height)/double(STEP));

    int xerr = m_width  - STEP*xstrips;if(xerr < 0){xstrips--;xerr = m_width - STEP*xstrips;}
    int yerr = m_height - STEP*ystrips;if(yerr < 0){ystrips--;yerr = m_height- STEP*ystrips;}

	double xerrperstrip = double(xerr)/double(xstrips);
	double yerrperstrip = double(yerr)/double(ystrips);

	int xoff = STEP/2;
	int yoff = STEP/2;
	//-------------------------
	numseeds = xstrips*ystrips;
	//-------------------------
	kseedsl.resize(numseeds);
	kseedsa.resize(numseeds);
	kseedsb.resize(numseeds);
	kseedsx.resize(numseeds);
	kseedsy.resize(numseeds);

	for( int y = 0; y < ystrips; y++ )
	{
		int ye = y*yerrperstrip;
		for( int x = 0; x < xstrips; x++ )
		{
			int xe = x*xerrperstrip;
            int seedx = (x*STEP+xoff+xe);
            if(hexgrid){ seedx = x*STEP+(xoff<<(y&0x1))+xe; seedx = std::min(m_width-1,seedx); }//for hex grid sampling
            int seedy = (y*STEP+yoff+ye);
            int i = seedy*m_width + seedx;
			
			kseedsl[n] = m_lvec[i];
			kseedsa[n] = m_avec[i];
			kseedsb[n] = m_bvec[i];
            kseedsx[n] = seedx;
            kseedsy[n] = seedy;
//             std::cout << kseedsl[n] << ' ';
			n++;
		}
	}
//     std::cout << std::endl;
}

void DrawContoursAroundSegments(
	cv::Mat&	    		image,
	int *		            labels,
	const int&				width,
	const int&				height)
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};


	int sz = width*height;
	std::vector<bool> istaken(sz, false);
	std::vector<int> contourx(sz);
    std::vector<int> contoury(sz);
	int mainindex(0);int cind(0);
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			int np(0);
			for( int i = 0; i < 8; i++ )
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if( (x >= 0 && x < width) && (y >= 0 && y < height) )
				{
					int index = y*width + x;

					//if( false == istaken[index] )//comment this to obtain internal contours
					{
						if( labels[mainindex] != labels[index] ) np++;
					}
				}
			}
			if( np > 1 )
			{
				contourx[cind] = k;
				contoury[cind] = j;
				istaken[mainindex] = true;
				//img[mainindex] = color;
				cind++;
			}
			mainindex++;
		}
	}

	int numboundpix = cind;//int(contourx.size());
	for( int j = 0; j < numboundpix; j++ )
	{
		// int ii = contoury[j]*width + contourx[j];
		// ubuff[ii] = 0xffffff;
        image.at<cv::Vec3b>(contoury[j], contourx[j])[0] = 255;
        image.at<cv::Vec3b>(contoury[j], contourx[j])[1] = 255;
        image.at<cv::Vec3b>(contoury[j], contourx[j])[2] = 255;

		for( int n = 0; n < 8; n++ )
		{
			int x = contourx[j] + dx8[n];
			int y = contoury[j] + dy8[n];
			if( (x >= 0 && x < width) && (y >= 0 && y < height) )
			{
				int ind = y*width + x;
				if(!istaken[ind]){
                    image.at<cv::Vec3b>(y, x)[0] = 0;
                    image.at<cv::Vec3b>(y, x)[1] = 0;
                    image.at<cv::Vec3b>(y, x)[2] = 0;
                } 
			}
		}
	}
    std::cout << "done drawing" << std::endl;
}
