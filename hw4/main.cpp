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
#include <mpi.h>



// #include "DistanceThreader.hpp"

// #include <SLIC.h>

#define ITERATIONS 10
#define MASTER 0

// MPI Tags
#define LBL_TAG 7
#define LBL_IDX_TAG 8
#define DIST_TAG 9


// using namespace zhang;

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

void SLIC_PerformSuperpixel(
                            // std::vector<double>& kseedsl, 
                            // std::vector<double>& kseedsa, 
                            // std::vector<double>& kseedsb,
                            // std::vector<double>& kseedsx,
                            // std::vector<double>& kseedsy,
                            double*              kseedsl,
                            double*              kseedsa,
                            double*              kseedsb,
                            double*              kseedsx,
                            double*              kseedsy,
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
    // std::vector<double>&		kseedsl,
	// std::vector<double>&		kseedsa,
	// std::vector<double>&		kseedsb,
	// std::vector<double>&		kseedsx,
	// std::vector<double>&		kseedsy,
    double*                     kseedsl,
    double*                     kseedsa,
    double*                     kseedsb,
    double*                     kseedsx,
    double*                     kseedsy,
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

void parallelWork(
            // std::vector<double> & ksdl,
            // std::vector<double> & ksda,
            // std::vector<double> & ksdb,
            // std::vector<double> & ksdx,
            // std::vector<double> & ksdy,
            double*  kseedsl,
            double*  kseedsa,
            double*  kseedsb,
            double*  kseedsx,
            double*  kseedsy,
            // std::vector<double> & kdist,
            //    int * klbl,
            //    int * lbl,
             double* lvec,
             double* avec,
             double* bvec,
                 int offset,
              double invwt,
          const int& width,
          const int& height,
                 int numk,
                 int rank,
                 int size);



// ==========================
//  Main
// ==========================
int main (int argc, char** argv) 
{
    // ---------------------
    // INIT MPI
    // ---------------------
    int rank;   // level of the node, 0-> master, other -> workers
    int size;   // number of nodes
    int ierr;

    ierr = MPI_Init(&argc, &argv);
    if ( ierr != 0) {
        std::cout << "MPI - Fatal error!" << std::endl;
        std::cout << "MPI_Init returned ierr = " << ierr << std::endl;
        exit(1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("MPI task %d of %d has started...\n", rank, size);

    // ---------------------
    // END INIT MPI
    // ---------------------

    
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

    if (rank == MASTER) {
        std::cout << "[Info] Four inputs:" << std::endl
                << "    1.           The input file (String) --> " << input_filename << std::endl
                << "    2.    The level of parallelism (int) --> " << lvl_para << std::endl
                << "    3. Count of target superpixels (int) --> " << k_superpixel << std::endl
                << "    4.     The output file name (String) --> " << output_filename << std::endl << std::endl;
    }
    



    // ------------------------------------------
    // Read the image
    // ------------------------------------------
    cv::Mat img = cv::imread(input_filename, cv::IMREAD_COLOR);

    if (rank == MASTER)
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

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    if (rank == MASTER) {
        // save the l a b to arrays
        start = std::chrono::high_resolution_clock::now();
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
        end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "[Info] Process image using: " << duration.count() << " ms" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&lvec, height*width, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&avec, height*width, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&bvec, height*width, MPI_DOUBLE, MASTER, MPI_COMM_WORLD); 
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    // parameters for SLIC algorithm
    // int*             klabels = NULL; //1, 512x512
    int                    k = k_superpixel;    // superpixel size
    double                 M = 30;     // Compactness factor
    const int superpixelsize = 0.5 + double(width*height)/double(k);
    const int           STEP = sqrt(double(superpixelsize))+0.5;

    int p = lvl_para;

    // kseeds
    // std::vector<double> kseedsl(0);
    // std::vector<double> kseedsa(0);
    // std::vector<double> kseedsb(0);
    // std::vector<double> kseedsx(0);
    // std::vector<double> kseedsy(0);
    double kseedsl[k];
    double kseedsa[k];
    double kseedsb[k];
    double kseedsx[k];
    double kseedsy[k];
    
    
    if (rank == MASTER) {
        // ------------------------------------------
        // SLIC get kseeds
        // ------------------------------------------
        // GetLABXYSeeds
        std::cout << "[INFO] start to get LABXY seeds..." << std::endl;
        GetLABXYSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, STEP, height, width, lvec, avec, bvec);
    }
    MPI_Barrier(MPI_COMM_WORLD); 
    MPI_Bcast(&kseedsl, k, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&kseedsa, k, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&kseedsb, k, MPI_DOUBLE, MASTER, MPI_COMM_WORLD); 
    MPI_Bcast(&kseedsx, k, MPI_DOUBLE, MASTER, MPI_COMM_WORLD); 
    MPI_Bcast(&kseedsy, k, MPI_DOUBLE, MASTER, MPI_COMM_WORLD); 
    MPI_Barrier(MPI_COMM_WORLD);
    
//     SLIC_PerformSuperpixel(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, M, height, width, lvec, avec, bvec, lvl_para);
    
    
    // ------------------------------------------------------------------------------------
    // SLIC superpixel
    // ------------------------------------------------------------------------------------
    const int sz = height * width;
    const int numk = sizeof(kseedsl)/sizeof(kseedsl[0]);
    std::cout << "main :: numk " << numk << std::endl;
//     numk = kseedsl.size();
    //----------------
    int offset = STEP;
    //if(STEP < 8) offset = STEP*1.5;//to prevent a crash due to a very small step size
    //----------------
    
   
    
    // std::vector<double> clustersize(numk, 0);
    // std::vector<double> inv(numk, 0);//to store 1/clustersize[k] values
    
    // std::vector<double> sigmal(numk, 0);
    // std::vector<double> sigmaa(numk, 0);
    // std::vector<double> sigmab(numk, 0);
    // std::vector<double> sigmax(numk, 0);
    // std::vector<double> sigmay(numk, 0);

    // std::vector<double> distvec(sz, DBL_MAX);
    double invwt = 1.0/((STEP/M)*(STEP/M));
    // int * klabels = NULL;
    // klabels = new int[sz];
    int klabels[sz]; //1, 512x512
    double distvec[sz];
    for (auto& d : distvec) d = DBL_MAX;




    // tic
    start = std::chrono::high_resolution_clock::now();
    
    
    for( int itr = 0; itr < ITERATIONS; itr++ ) {
        MPI_Barrier(MPI_COMM_WORLD); 
        

        if ( rank != MASTER) {
            // std::cout << "IN worker:: " << rank << std::endl;
            // std::cout << "iter::: " << itr << " , kseedsy::    ";
            // for (int i = 0; i < 100; ++i) std::cout << kseedsy[i] << " ";
            // std::cout << std::endl;

            parallelWork(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, lvec, avec, bvec, offset, invwt, width, height, numk, rank-1, size-1); 
        } else {
            std::cout << std::endl << "[Info] ::Iteration:: " << itr << std::endl;

            for (auto& d : distvec) d = DBL_MAX; 

            // std::cout << std::endl << "$$$$$$$$ .   kdist::   ";
            // for (int i = 0; i < 100; ++i ) std::cout << distvec[i] << " ";
            // std::cout << std::endl;

            

            for (int r = 1; r < size; r++) {
                int worker_res[sz];
                double distance[sz];
                MPI_Recv(&worker_res, sz, MPI_INT, r, LBL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&distance, sz, MPI_DOUBLE, r, DIST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // std::cout << " results from rank:: " << rank << " .    ";
                // for (int i = 0; i < 100; i++) std::cout << "[ " << worker_res[i] << ", " << distance[i] << " ]  ";
                // std::cout << std::endl;


                for (int i = 0; i < sz; i++) {
                    if (worker_res[i] >= 0 && distance[i] <= distvec[i]) {
                        distvec[i] = distance[i];
                        klabels[i] = worker_res[i];
                    }
                } 
            }

            // std::cout << "ITER:: " << itr << ", label::      ";
            // for (int i = 0; i < 200; i++) std::cout << klabels[i] << " ";
            // std::cout << std::endl;


            //-----------------------------------------------------------------
            // Recalculate the centroid and store in the seed values
            //-----------------------------------------------------------------
            //instead of reassigning memory on each iteration, just reset.
            // sigmal.assign(numk, 0);
            // sigmaa.assign(numk, 0);
            // sigmab.assign(numk, 0);
            // sigmax.assign(numk, 0);
            // sigmay.assign(numk, 0);
            // clustersize.assign(numk, 0);
            std::vector<double> clustersize(numk, 0);
            std::vector<double> inv(numk, 0);//to store 1/clustersize[k] values
            
            std::vector<double> sigmal(numk, 0);
            std::vector<double> sigmaa(numk, 0);
            std::vector<double> sigmab(numk, 0);
            std::vector<double> sigmax(numk, 0);
            std::vector<double> sigmay(numk, 0);
            //------------------------------------
            //edgesum.assign(numk, 0);
            //------------------------------------

            {int ind(0);
            for( int r = 0; r < height; r++ )
            {
                for( int c = 0; c < width; c++ )
                {
                    sigmal.at(klabels[ind]) += lvec[ind];
                    sigmaa.at(klabels[ind]) += avec[ind];
                    sigmab.at(klabels[ind]) += bvec[ind];
                    sigmax.at(klabels[ind]) += c;
                    sigmay.at(klabels[ind]) += r;
                    //------------------------------------
                    //edgesum[klabels[ind]] += edgemag[ind];
                    //------------------------------------
                    clustersize.at(klabels[ind]) += 1.0;
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
            std::cout << "       Calculated Centroid:::" << itr << std::endl;
            
            // std::cout << "ITER::" << itr << " .  kseedsy::   ";
            // for (auto a : kseedsy) std::cout << a << " ";
            // std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&kseedsl, numk, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
        MPI_Bcast(&kseedsa, numk, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
        MPI_Bcast(&kseedsb, numk, MPI_DOUBLE, MASTER, MPI_COMM_WORLD); 
        MPI_Bcast(&kseedsx, numk, MPI_DOUBLE, MASTER, MPI_COMM_WORLD); 
        MPI_Bcast(&kseedsy, numk, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    }
    
    if (rank == MASTER) {
        // toc
        end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "[Info] SLIC using: " << duration.count() << " ms" << std::endl;
        
        DrawContoursAroundSegments(img, klabels, width, height);

        cv::imwrite("./" + output_filename + ".png", img);

        std::cout << "[Info] Output image saved in build/" << output_filename << ".png" << std::endl;
    }    

    MPI_Finalize();

    // klabels = NULL;
    // delete klabels;    
    return 0;
}
























// ----------------------------------------------------------------------------------------
void parallelWork(
            // std::vector<double> & ksdl,
            // std::vector<double> & ksda,
            // std::vector<double> & ksdb,
            // std::vector<double> & ksdx,
            // std::vector<double> & ksdy,
             double* ksdl,
             double* ksda,
             double* ksdb,
             double* ksdx,
             double* ksdy,
            // std::vector<double> & kdist,
            //    int * klbl,
            //    int * lbl,
             double* lvec,
             double* avec,
             double* bvec,
                 int offset,
              double invwt,
          const int& width,
          const int& height,
                 int numk,
                 int rank,
                 int size)
{
    // std::cout << "In rank :::: " << rank+1 << ", size::: " << size+1 << " numk::" << numk << std::endl;
    int sz = width*height;
    int lbl[sz];
    for (auto& a : lbl) a = -1;
    double kdist[sz];
    for (auto& d : kdist) d = DBL_MAX; 

    int x1, y1, x2, y2;
    double l, a, b;
    double dist;
    double distxy;

    // std::cout << "\nlbl in worker :: " << rank+1 << "    ";
    // for (int i = 0; i < 100; i++) std::cout << lbl[i] << " ";
    // std::cout << std::endl; 

    for (int n = 0; n < numk; n++)
    {
        if ( n % size == rank)
        {
           
            y1 = std::max(0.0,             ksdy[n]-offset);
            y2 = std::min((double)height,  ksdy[n]+offset);
            x1 = std::max(0.0,             ksdx[n]-offset);
            x2 = std::min((double)width,   ksdx[n]+offset);
            for( int y = y1; y < y2; y++ )
            {
                for( int x = x1; x < x2; x++ )
                {
                    int i = y*width + x;
                    // get the value of l a b in the pixel
                    l = lvec[i];
                    a = avec[i];
                    b = bvec[i];
                    dist =          (l - ksdl[n])*(l - ksdl[n]) +
                                    (a - ksda[n])*(a - ksda[n]) +
                                    (b - ksdb[n])*(b - ksdb[n]);
                    distxy =        (x - ksdx[n])*(x - ksdx[n]) +
                                    (y - ksdy[n])*(y - ksdy[n]);

                    //------------------------------------------------------------------------
                    dist += distxy*invwt;//dist = sqrt(dist) + sqrt(distxy*invwt);//this is more exact
                    //------------------------------------------------------------------------
                    if( dist < kdist[i] )
                    {
                        // if (i == 0) printf("         HERE::::::::::::, RANK:::::%d .    X::%d .   X1::%d .    X2::%d .     Y::%d     Y1::%d .    Y2::%d .   " , rank+1, x, ksdx[n], x2, y, ksdy[n], y2);
                        kdist[i] = dist;
                        // klbl[i]  = n;
                        lbl[i] = n;
                    }
                }
            }
        }
    }

    // std::cout << "\nklabel in worker :: " << rank+1 << "    ";
    // for (int i = 0; i < 500; i++) std::cout << lbl[i] << " ";
    // std::cout << std::endl; 

    MPI_Send(&lbl, sz, MPI_INT, MASTER, LBL_TAG, MPI_COMM_WORLD);
    MPI_Send(&kdist, sz, MPI_DOUBLE, MASTER, DIST_TAG, MPI_COMM_WORLD);
}

//===========================================================================
///	GetLABXYSeeds_ForGivenStepSize
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void GetLABXYSeeds(
	// std::vector<double>&		kseedsl,
	// std::vector<double>&		kseedsa,
	// std::vector<double>&		kseedsb,
	// std::vector<double>&		kseedsx,
	// std::vector<double>&		kseedsy,
    double*                     kseedsl,
    double*                     kseedsa,
    double*                     kseedsb,
    double*                     kseedsx,
    double*                     kseedsy,
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
    std::cout << "numseeds:: " << numseeds << std::endl;
	//-------------------------
	// kseedsl.resize(numseeds);
	// kseedsa.resize(numseeds);
	// kseedsb.resize(numseeds);
	// kseedsx.resize(numseeds);
	// kseedsy.resize(numseeds);

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
			n++;
		}
	}
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
