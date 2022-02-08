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


// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Utilities and system includes
#include <helper_string.h>    // helper for string parsing
#include <helper_image.h>     // helper for image and data comparison
#include <helper_cuda.h>      // helper for cuda error checking functions

using namespace cooperative_groups;

#define ITERATIONS 10

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

#define cudaDeviceScheduleBlockingSync 0x04    // Device flag - Use blocking synchronization

// ================================================================================================================
// ================================================================================================================

int grid_divide(int length, int block_size)
{
    return ((length % block_size) == 0) ? (length / block_size) : (length / block_size + 1);
}


void SLIC_PerformSuperpixel( double* kseedsl, 
                             double* kseedsa, 
                             double* kseedsb,
                             double* kseedsx,
                             double* kseedsy,
                               int*& klabels,       // number of superpixel
                          const int& STEP,          // S, neighbor size
                       const double& M,             // Compactness factor. ranging from 10 to 40 
                                 int m_height,        // height of the image
                                 int m_width,         // width of the image
                             double* m_lvec,
                             double* m_avec,
                             double* m_bvec,
                                 int numk
                            )
{
    int sz = m_height * m_width;
    //const int numk = kseedsl.size();
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
    int x1, y1, x2, y2;
    double l, a, b;
    double dist;
    double distxy;
    klabels = new int[sz];
    for( int itr = 0; itr < ITERATIONS; itr++ )
    {
        // std::cout << "[Info] ::Iteration:: " << itr << std::endl;
        distvec.assign(sz, DBL_MAX);
        for( int n = 0; n < numk; n++ )
        {
            y1 = std::max(0.0,               kseedsy[n]-offset);
            y2 = std::min((double)m_height,  kseedsy[n]+offset);
            x1 = std::max(0.0,               kseedsx[n]-offset);
            x2 = std::min((double)m_width,   kseedsx[n]+offset);
    
            for( int y = y1; y < y2; y++ )
            {
                for( int x = x1; x < x2; x++ )
                {
                    int i = y*m_width + x;
                    // get the value of l a b in the pixel
                    l = m_lvec[i];
                    a = m_avec[i];
                    b = m_bvec[i];
                    dist =          (l - kseedsl[n])*(l - kseedsl[n]) +
                                    (a - kseedsa[n])*(a - kseedsa[n]) +
                                    (b - kseedsb[n])*(b - kseedsb[n]);
                    distxy =        (x - kseedsx[n])*(x - kseedsx[n]) +
                                    (y - kseedsy[n])*(y - kseedsy[n]);
                    
                    //------------------------------------------------------------------------
                    dist += distxy*invwt;//dist = sqrt(dist) + sqrt(distxy*invwt);//this is more exact
                    //------------------------------------------------------------------------
                    if( dist < distvec[i] )
                    {
                        distvec[i] = dist;
                        klabels[i]  = n;
                    }
                }
            }
        }
        
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
        for( int r = 0; r < m_height; r++ )
        {
            for( int c = 0; c < m_width; c++ )
            {
                sigmal[klabels[ind]] += m_lvec[ind];
                sigmaa[klabels[ind]] += m_avec[ind];
                sigmab[klabels[ind]] += m_bvec[ind];
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
    }


}


//===========================================================================
///	GetLABXYSeeds_ForGivenStepSize
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void GetLABXYSeeds(
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
    std::cout << "    Done drawing" << std::endl;
}



// =====
// host
// =====
__device__ int domax(double x, double y)
{
  return (x > y)? x : y;
}

__device__ int domin(double x, double y)
{
  return (x < y)? x : y;
}

__global__ void parallelWork(int* lbl, double* kseedsl, double* kseedsa, double* kseedsb, double* kseedsx, double* kseedsy, double* vl, double* va, double* vb, double offset, double invwt, int width, int height, int numk)
{
    //grid_group g = this_grid();
    
    
    // --- Find X and Y
    const int xx = threadIdx.x;
    const int yy = threadIdx.y;
    
    const int x = blockDim.x * blockIdx.x + xx;
    const int y = blockDim.y * blockIdx.y + yy;
    
    
    //const int tid = yy * blockDim.x + xx; 
    //const int sz = width*height;
    //__shared__ double kdist[BLOCK_WIDTH*BLOCK_HEIGHT];
    //for (auto& d : kdist) d = DBL_MAX;
    
    
    double kdist = DBL_MAX;
    //int idx = 0;
    
    // --- Compute distances
    int x1, y1, x2, y2;
    double l, a, b;
    double dist;
    double distxy;
    
    int theidx = 0;
    for (int n = 0; n < numk; n++)
    {
        y1 = domax(0.0,             kseedsy[n]-offset);
        y2 = domin((double)height,  kseedsy[n]+offset);
        x1 = domax(0.0,             kseedsx[n]-offset);
        x2 = domin((double)width,   kseedsx[n]+offset);
        
        if (false)//(x == 0 && y == 0) 
        {
            printf("Numk: %d  ---  x1: %d, x2: %d, y1: %d, y2: %d", theidx, x1, x2, y1, y2);
            printf("  ---  kseedsl: %f, kseedsa: %f, kseedsb: %f, kseedsx: %f, kseedsy: %f\n", kseedsl[n], kseedsa[n], kseedsb[n], kseedsx[n], kseedsy[n]);
            theidx++;
        }
        
        l = vl[y * width + x];
        a = va[y * width + x];
        b = vb[y * width + x];
        
        if (x >= x1 && x < x2 && y >= y1 && y < y2) 
        {
            dist =   (l - kseedsl[n])*(l - kseedsl[n]) +
                     (a - kseedsa[n])*(a - kseedsa[n]) +
                     (b - kseedsb[n])*(b - kseedsb[n]);
            distxy = (x - kseedsx[n])*(x - kseedsx[n]) +
                     (y - kseedsy[n])*(y - kseedsy[n]);
            
            dist += distxy*invwt;          
        
            if (dist < kdist) 
            {
                kdist = dist;
                lbl[y * width + x] = n;
                //__syncthreads();
            }
        }
    }

    __syncthreads();
    //if (g.is_valid())
        //g.sync();
}



// =====
// main
// =====


int main(int argc, char **argv)
{
    // ---------------
    // Print CUDA Info
    // ---------------

    printf("\n\n[INFO] Printing CUDA Info\n\n");

    int devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProp;

    // --- get number of SMs on this GPU
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    //checkCudaErrors(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

    printf("> Device %d: \"%s\"\n", devID, deviceProp.name);




    printf("\n\n");
    printf("========================\n");
    printf(" HPC Homework6 Starting\n");
    printf("========================\n");



    // ------------------------------------------
    // Check four inputs
    // ------------------------------------------
    if (argc != 4) 
    {
        std::cout << "[Error] Input parameters missing, need four inputs:" << std::endl
                    << "    1. The input file (String)" << std::endl
                    << "    2. Count of target superpixels (int)" << std::endl
                    << "    3. The output file name (String)" << std::endl;
        return -1;
    }

    // Store inputs
    std::string input_filename, output_filename;
    int k_superpixel;

    input_filename  = argv[1];              // input image path
    //lvl_para        = std::stoi(argv[2]);   // number of processes to process the dataset
    k_superpixel    = std::stoi(argv[2]);   // the count of target superpixels. eg. 200
    output_filename = argv[3];

    
    std::cout << "\n[Info] Three inputs:" << std::endl
            << "    1.           The input file (String) --> " << input_filename << std::endl
            << "    2. Count of target superpixels (int) --> " << k_superpixel << std::endl
            << "    3.     The output file name (String) --> " << output_filename << std::endl << std::endl;
    


    // ------------------------------------------
    // Read the image
    // ------------------------------------------
    cv::Mat img = cv::imread(input_filename, cv::IMREAD_COLOR);

    std::cout << "\n[Info] Image Size: " << img.rows << " x " << img.cols << " x " << img.channels() << std::endl;

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
    clock_t time = clock();
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
    time = clock() - time;
    std::cout << "[Info] Process image using: " << (double)time/(CLOCKS_PER_SEC/1000) << " ms" << std::endl;

    
    // parameters for SLIC algorithm
    int*         klabels_cpu = NULL; //1, 512x512
    int                    k = k_superpixel;    // superpixel size
    double                 M = 30;     // Compactness factor
    const int superpixelsize = 0.5 + double(width*height)/double(k);
    const int           STEP = sqrt(double(superpixelsize))+0.5;
    
    
    //const int sz = height * width;
    int numk = k;

    int offset = STEP;
    double invwt = 1.0/((STEP/M)*(STEP/M));

    // kseeds
    double kseedsl[numk];
    double kseedsa[numk];
    double kseedsb[numk];
    double kseedsx[numk];
    double kseedsy[numk];
    
    
    
    // ------------------------------------------
    // SLIC
    // ------------------------------------------
    // GetLABXYSeeds
    printf("\n[INFO] Getting L A B X Y Seeds...\n");
    GetLABXYSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, STEP, height, width, lvec, avec, bvec);


    printf("\n[INFO] Starting C-SLIC...\n");
    clock_t cpu_time = clock();
    SLIC_PerformSuperpixel(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels_cpu, STEP, M, height, width, lvec, avec, bvec, numk);
    cpu_time = clock() - cpu_time;
    
    
    // ------------------------------------------------------------------------------------
    // SLIC superpixel
    // ------------------------------------------------------------------------------------

    // --- Start GSLIC
    printf("\n[INFO] Starting G-SLIC...\n");
    
    // --- Allocate the host and device memory spaces
    printf("    Allocating memory spaces\n");
    
    // kseeds
    double* d_kl = NULL;
    double* d_ka = NULL;
    double* d_kb = NULL;
    double* d_kx = NULL;
    double* d_ky = NULL;
    // lab vectors
    double* d_vl = NULL;
    double* d_va = NULL;
    double* d_vb = NULL;
    // klabels
    int* h_lbl = (int*)malloc(height * width * sizeof(int));
    int* d_lbl = NULL;
    
    
    checkCudaErrors(cudaMalloc((void**) &d_kl,  (numk * sizeof(double))));
    checkCudaErrors(cudaMalloc((void**) &d_ka,  (numk * sizeof(double))));
    checkCudaErrors(cudaMalloc((void**) &d_kb,  (numk * sizeof(double))));
    checkCudaErrors(cudaMalloc((void**) &d_kx,  (numk * sizeof(double))));
    checkCudaErrors(cudaMalloc((void**) &d_ky,  (numk * sizeof(double))));
    
    checkCudaErrors(cudaMalloc((void**) &d_vl,  (height * width * sizeof(double))));
    checkCudaErrors(cudaMalloc((void**) &d_va,  (height * width * sizeof(double))));
    checkCudaErrors(cudaMalloc((void**) &d_vb,  (height * width * sizeof(double))));
    checkCudaErrors(cudaMalloc((void**) &d_lbl, (height * width * sizeof(int))));
    
    // --- Copy LAB vec data from host to device
    checkCudaErrors(cudaMemcpy(d_vl, &lvec, height * width * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_va, &avec, height * width * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_vb, &bvec, height * width * sizeof(double), cudaMemcpyHostToDevice));

    // --- Grid and block sizes
    const dim3 grid  (grid_divide(width, BLOCK_WIDTH), grid_divide(height, BLOCK_HEIGHT), 1);
    const dim3 block (BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    
    
    clock_t gpu_time = clock();
    
    for( int itr = 0; itr < ITERATIONS; itr++ ) {
        printf("    Iteration: %d", itr);
        
        // --- Copy data from host to device
        printf("   -  Copy kseeds to device...\n");
        
        checkCudaErrors(cudaMemcpy(d_kl, &kseedsl, numk * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_ka, &kseedsa, numk * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_kb, &kseedsb, numk * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_kx, &kseedsx, numk * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_ky, &kseedsy, numk * sizeof(double), cudaMemcpyHostToDevice));
        
        

        // --------
        //  kernel
        // --------
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        
        cudaFuncSetCacheConfig(parallelWork, cudaFuncCachePreferShared); // cudaFuncCachePreferNone
        
        
        // Start kernel function

        parallelWork<<<grid, block>>>(d_lbl, d_kl, d_ka, d_kb, d_kx, d_ky, d_vl, d_va, d_vb, offset, invwt, width, height, numk);
        //cudaDeviceSynchronize();

        
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        printf("                   -  Kernel elapsed time: %.3f ms\n", time);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        
        // --- Copy results from Device to host
        printf("                   -  Start to copy Klabels from Device to host \n");
        checkCudaErrors(cudaMemcpy(h_lbl, d_lbl, height * width * sizeof(int), cudaMemcpyDeviceToHost));
        
        
        
        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------
        //instead of reassigning memory on each iteration, just reset.

        std::vector<double> clustersize(numk, 0);
        std::vector<double> inv(numk, 0);    //to store 1/clustersize[k] values

        std::vector<double> sigmal(numk, 0);
        std::vector<double> sigmaa(numk, 0);
        std::vector<double> sigmab(numk, 0);
        std::vector<double> sigmax(numk, 0);
        std::vector<double> sigmay(numk, 0);
        
        //printf("label last: %d \n", h_lbl[height*width-1]);
        
        printf("                   -  Computing Sigmas \n");
        
        {int ind(0);
        for( int r = 0; r < height; r++ )
        {
            for( int c = 0; c < width; c++ )
            {
                sigmal.at(h_lbl[ind]) += lvec[ind];
                sigmaa.at(h_lbl[ind]) += avec[ind];
                sigmab.at(h_lbl[ind]) += bvec[ind];
                sigmax.at(h_lbl[ind]) += c;
                sigmay.at(h_lbl[ind]) += r;
                //------------------------------------
                //edgesum[klabels[ind]] += edgemag[ind];
                //------------------------------------
                clustersize.at(h_lbl[ind]) += 1.0;
                ind++;
            }
        }}
        printf("                   -  Computed Sigmas \n");
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
        
        printf("                   -  Update new KSeeds \n");
    }
    
    gpu_time = clock() - gpu_time;
    
    
    // Print out execution time
    printf("\n[INFO] %d Iterations Execution time:\n", ITERATIONS);
    printf("       - CPU using %.3f ms\n", (double)cpu_time/(CLOCKS_PER_SEC/1000));
    printf("       - GPU using %.3f ms\n", (double)gpu_time/(CLOCKS_PER_SEC/1000));
    
    
    printf("\n[INFO] Drawing Contours...\n");
    
    DrawContoursAroundSegments(img, h_lbl, width, height);
    
    cv::imwrite("./" + output_filename + ".png", img);

    std::cout << "\n[Info] Output image saved in build/" << output_filename << ".png" << std::endl;

    
    
    
    // --- Clean up
    printf("\n[INFO] Cleaning up memory spaces...\n");
    
    // label
    if (h_lbl) { free(h_lbl); h_lbl=NULL; }
    if (d_lbl) { cudaFree(d_lbl);  d_lbl=NULL; }
    
    // distance
    //if (h_dist) { free(h_dist); h_dist=NULL; }
    //if (d_dist) { cudaFree(d_dist);  d_dist=NULL; }
    
    // kseedsval
    if (d_kl) { cudaFree(d_kl); d_kl=NULL; }
    if (d_ka) { cudaFree(d_ka); d_ka=NULL; }
    if (d_kb) { cudaFree(d_kb); d_kb=NULL; }
    if (d_kx) { cudaFree(d_kx); d_kx=NULL; }
    if (d_ky) { cudaFree(d_ky); d_ky=NULL; }
    
    // lab vec
    if (d_vl) { cudaFree(d_vl); d_vl=NULL; }
    if (d_va) { cudaFree(d_va); d_va=NULL; }
    if (d_vb) { cudaFree(d_vb); d_vb=NULL; }
    
    

    return 0;
}
