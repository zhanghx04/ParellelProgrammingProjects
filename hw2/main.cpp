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

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <unistd.h>
#include <cstdlib>

// #include <SLIC.h>

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

void parallelWork(
            double * ksdl,
            double * ksda,
            double * ksdb,
            double * ksdx,
            double * ksdy,
            double * kdist,
               int * klbl,
               int   n,
             double* lvec,
             double* avec,
             double* bvec,
                 int offset,
              double invwt,
          const int& width,
          const int& height);

// Generate random key
int prng(int n) {
    auto t =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    srand(t);
    return rand() % n;
}

union mysemun 
{
	int val;
	struct semid_ds *buf; // see comment above about what a semid_ds is 
	ushort *array;
};

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
    // SLIC
    // ------------------------------------------
    // GetLABXYSeeds
    GetLABXYSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, STEP, height, width, lvec, avec, bvec);

    
    
//     SLIC_PerformSuperpixel(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, M, height, width, lvec, avec, bvec, lvl_para);
    
    

    
    
    // =============================
    // BEGIN: Do the Semaphore Setup
    // The semaphore will be a mutex
    // =============================
    int semId; 			            // ID of semaphore set
    key_t semKey = prng(1000000); 		    // key to pass to semget(), key_t is an IPC key type defined in sys/types
    int semFlag = IPC_CREAT | 0666; // Flag to create with rw permissions

    int semCount = p; 		        // number of semaphores to pass to semget()
    int numOps = 1; 		        // number of operations to do
    
    
    
    // Create semaphores
    semId = semget(semKey, semCount, semFlag);
    if ( semId == -1)
    {
        std::cerr << "Failed to semget(" << semKey << "," << semCount << "," << semFlag << ")" << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "Successful semget resulted in (" << semId << ")" << std::endl;
    }
    
    

    // Initialize the semaphore
    union semun {
        int val;
        struct semid_ds *buf;
        ushort * array;
    } argument;

    // Count mutex semaphore
    argument.val = p;
    if( semctl(semId, 0, SETVAL, argument) < 0)
    {
        std::cerr << "Init: Failed to initialize (" << semId << ")" << std::endl;
        exit(1);
    }
    else
	{
		std::cout << "Init: Initialized (" << semId << ")" << std::endl; 
	}
    
    
    ///============
    int semId1; 			            // ID of semaphore set
    key_t semKey1 = semKey + 10; 		    // key to pass to semget(), key_t is an IPC key type defined in sys/types
    int semCount1 = 1; 		        // number of semaphores to pass to semget()
    
    
    
    // Create semaphores
    semId1 = semget(semKey1, semCount1, semFlag);
    if ( semId1 == -1)
    {
        std::cerr << "Failed to semget(" << semKey1 << "," << semCount1 << "," << semFlag << ")" << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "Successful semget resulted in (" << semId1 << ")" << std::endl;
    }
    
    
    union semun1 {
        int val;
        struct semid_ds *buf;
        ushort * array;
    } argument1;
    argument1.val = 0;
    if( semctl(semId1, 0, SETVAL, argument1) < 0)
    {
        std::cerr << "Init: Failed to initialize (" << semId1 << ")" << std::endl;
        exit(1);
    }
    else
	{
		std::cout << "Init: Initialized (" << semId1 << ")" << std::endl; 
	}


	// Define semaphore operations
	struct sembuf semWait[1];
	semWait[0].sem_num = 0; // Mutex
	semWait[0].sem_op = -1; // Wait
	semWait[0].sem_flg = 0;//SEM_UNDO; // allows calling process to block and wait

    struct sembuf semSignal[1];
    semSignal[0].sem_num = 0; // Mutex
    semSignal[0].sem_op = 1;  // Signal
    semSignal[0].sem_flg = 0; //SEM_UNDO; // allows calling process to block and wait

    struct sembuf turn1Wait[1];
    turn1Wait[0].sem_num = 0; // Mutex
    turn1Wait[0].sem_op = -p; // Wait
    turn1Wait[0].sem_flg = 0; //SEM_UNDO; // allows calling process to block and wait

    struct sembuf turn1Signal[1];
    turn1Signal[0].sem_num = 0; // Mutex
    turn1Signal[0].sem_op = p;  // Signal
    turn1Signal[0].sem_flg = 0; //SEM_UNDO; // allows calling process to block and wait

    
    // =============================
	// END: Do the Semaphore Setup
	// =============================
    std::cout << "semIds:: " << semId << std::endl;
    
    
    

    const int sz = height * width;
    const int numk = kseedsl.size();
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
//     klabels = new int[sz];

    // =========================================
	// BEGIN: Do the Shared Memory Segment Setup
	// 
	// =========================================

    int shmlId, shmaId, shmbId, shmxId, shmyId, shmDistId, shmLabelId, shmNumkId;
    // key to pass to shmget(), key_t is an IPC key type defined in sys/types
	key_t shmlKey = semKey + 1,
          shmaKey = semKey + 2,
          shmbKey = semKey + 3,
          shmxKey = semKey + 4,
          shmyKey = semKey + 5,
       shmDistKey = semKey + 6,
      shmLabelKey = semKey + 7,
       shmNumkKey = semKey + 8; 		
    
	int shmFlag = IPC_CREAT | 0666; // Flag to create with rw permissions
	
	// This will be shared:
	double * ksdl, * ksda, * ksdb, * ksdx, * ksdy;
    double * kdist;
    int * klbl;

//     double * ksdlPtr = NULL;
//     double * ksdaPtr = NULL;
//     double * ksdbPtr = NULL;
//     double * ksdxPtr = NULL;
//     double * ksdyPtr = NULL;
//     double * distPtr = NULL;
//     int * klblPtr = NULL;
    int * numkPtr = NULL;
    
    // l
    if ((shmlId = shmget(shmlKey, numk * sizeof(double), shmFlag)) < 0)
	{
		std::cerr << "Init: Failed to initialize shared memory (" << shmlId << ")" << std::endl; 
		exit(1);
	}
    if ((ksdl = (double *)shmat(shmlId, NULL, 0)) == (double *) -1)
	{
		std::cerr << "Init: Failed to attach shared memory (" << shmlId << ")" << std::endl; 
		exit(1);
	}
    // a
    if ((shmaId = shmget(shmaKey, numk * sizeof(double), shmFlag)) < 0)
	{
		std::cerr << "Init: Failed to initialize shared memory (" << shmaId << ")" << std::endl; 
		exit(1);
	}
    if ((ksda = (double *)shmat(shmaId, NULL, 0)) == (double *) -1)
	{
		std::cerr << "Init: Failed to attach shared memory (" << shmaId << ")" << std::endl; 
		exit(1);
	}
    // b
    if ((shmbId = shmget(shmbKey, numk * sizeof(double), shmFlag)) < 0)
	{
		std::cerr << "Init: Failed to initialize shared memory (" << shmbId << ")" << std::endl; 
		exit(1);
	}
    if ((ksdb = (double *)shmat(shmbId, NULL, 0)) == (double *) -1)
	{
		std::cerr << "Init: Failed to attach shared memory (" << shmbId << ")" << std::endl; 
		exit(1);
	}
    // x
    if ((shmxId = shmget(shmxKey, numk * sizeof(double), shmFlag)) < 0)
	{
		std::cerr << "Init: Failed to initialize shared memory (" << shmxId << ")" << std::endl; 
		exit(1);
	}
    if ((ksdx = (double *)shmat(shmxId, NULL, 0)) == (double *) -1)
	{
		std::cerr << "Init: Failed to attach shared memory (" << shmxId << ")" << std::endl; 
		exit(1);
	}
    // y
    if ((shmyId = shmget(shmyKey, numk * sizeof(double), shmFlag)) < 0)
	{
		std::cerr << "Init: Failed to initialize shared memory (" << shmyId << ")" << std::endl; 
		exit(1);
	}
    if ((ksdy = (double *)shmat(shmyId, NULL, 0)) == (double *) -1)
	{
		std::cerr << "Init: Failed to attach shared memory (" << shmyId << ")" << std::endl; 
		exit(1);
	}
    // dist
    if ((shmDistId = shmget(shmDistKey, sz * sizeof(double), shmFlag)) < 0)
	{
		std::cerr << "Init: Failed to initialize shared memory (" << shmDistId << ")" << std::endl; 
		exit(1);
	}
    if ((kdist = (double *)shmat(shmDistId, NULL, 0)) == (double *) -1)
	{
		std::cerr << "Init: Failed to attach shared memory (" << shmDistId << ")" << std::endl; 
		exit(1);
	}
    // label
    if ((shmLabelId = shmget(shmLabelKey, sz * sizeof(int), shmFlag)) < 0)
	{
		std::cerr << "Init: Failed to initialize shared memory (" << shmLabelId << ")" << std::endl; 
		exit(1);
	}
    if ((klbl = (int *)shmat(shmLabelId, NULL, 0)) == (int *) -1)
	{
		std::cerr << "Init: Failed to attach shared memory (" << shmLabelId << ")" << std::endl; 
		exit(1);
	}
    // numk
    // label
    if ((shmNumkId = shmget(shmNumkKey, 1 * sizeof(int), shmFlag)) < 0)
	{
		std::cerr << "Init: Failed to initialize shared memory (" << shmLabelId << ")" << std::endl; 
		exit(1);
	}
    if ((numkPtr = (int *)shmat(shmNumkId, NULL, 0)) == (int *) -1)
	{
		std::cerr << "Init: Failed to attach shared memory (" << shmLabelId << ")" << std::endl; 
		exit(1);
	}
    
    *numkPtr = 0;
    
    
    // =========================================
	// END: Do the Shared Memory Segment Setup
	// =========================================
    
   
    
    // Initialize shm
    for (size_t i = 0; i < numk; i++)
    {
        ksdl[i] = kseedsl[i];
        ksda[i] = kseedsa[i];
        ksdb[i] = kseedsb[i];
        ksdx[i] = kseedsx[i];
        ksdy[i] = kseedsy[i];
        
//         std::cout << "ksdb:: " << ksdb[i] << " kseedsl:: " << kseedsb[i] << std::endl;
//         std::cout << "checkpoint" << std::endl;
    }
    
    for (size_t i = 0; i < sz; i++)
    {
        kdist[i] = DBL_MAX;
    }

    
    std::cout << "DBL_MAX::" << DBL_MAX << std::endl;
    
    // =========================================
    // BEGIN: Forking 
    // =========================================
    pid_t pid[p];
    int myIdx = -1;     // parent process will be -1

    
    
    // tic
    start = std::chrono::high_resolution_clock::now();
    
        
    for (int i = 0; i < p; ++i)
    {
        pid[i] = fork();
        if (pid[i] == 0)
        {
            myIdx = i; // pid[myIdx] will give 0 in the child
            std::cout<< "Child " << i<<std::endl;
            break;
        }
        else if (pid[i] < 0)
        {
            std::cerr << "Could not fork!!! (" << pid << ")" << std::endl;
            exit(1);
        }
    }
    
//     struct sembuf operations[1];

//     // Set up the sembuf structure.
//     operations[0].sem_num = 0; 	// use the first(only, because of semCount above) semaphore
//     operations[0].sem_op = -1; 	// this the operation... the value is added to semaphore (a P-Op = -1)
//     operations[0].sem_flg = IPC_NOWAIT;	// set to IPC_NOWAIT to allow the calling process to fast-fail

    // std::cout << "In the child (if): about to no-wait on semaphore" << std::endl; 
    
    int retval;
    int retval2;
    
    for( int itr = 0; itr < 10; itr++ ) {
        if (myIdx == -1)
        {
            std::cout << "[Info] ::Iteration:: " << itr << std::endl;
            
            
        }
        


        


        if (myIdx != -1)
        {
             std::cout << "This is a child process - " << itr<< ' '<< myIdx << std::endl;
            
            retval = semop(semId, semWait, numOps);// wait -1
            if (retval == -1) {
                int errval = errno;
                std::cout << "Child " << myIdx << "itr " << itr << ' ' <<strerror(errval) << std::endl;
            }
            
            
            std::cout << "semWait::p" << myIdx << ":" << retval << std::endl;

            for (int k = 0; k < numk; k++)
            {
    //                     std::cout << "Child:::" << id << std::endl;
                
                if (k % p == myIdx)
                {    
                    std::cout << "This is a child process - " << myIdx << " " <<  ksdl << " " << ksda << " " << ksdb << " " << ksdx << " " << ksdy << " " << kdist << " " << klbl << " " << k << " " << lvec << " " << avec << " " << bvec << " " << offset << " " << invwt << " " << width << " " << height << std::endl;
                    parallelWork(ksdl, ksda, ksdb, ksdx, ksdy, kdist, klbl, k, lvec, avec, bvec, offset, invwt, width, height);
                }

    //                     std::cout << "Child::: +++++++++ "  << k << std::endl
                
            }
            //while(*numkPtr < numk);
//             retval2 = semop(semId1, semSignal, numOps); // signal + 1
            semop(semId1, semSignal, numOps);
//             if (0 ==  && itr >= 9) {
//                 std::cout << "Child process (" << pid[myIdx] << ") is done." << myIdx << std::endl;
// //                 exit(0);
//             }
        }
        else if (myIdx == -1)
        {
            
            //-----------------------------------------------------------------
            // Recalculate the centroid and store in the seed values
            //-----------------------------------------------------------------
            //instead of reassigning memory on each iteration, just reset.
            
            std::cout << "Parent wait::::" << std::endl;
            semop(semId1, turn1Wait, numOps);
            
            
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
                    sigmal[klbl[ind]] += lvec[ind];
                    sigmaa[klbl[ind]] += avec[ind];
                    sigmab[klbl[ind]] += bvec[ind];
                    sigmax[klbl[ind]] += c;
                    sigmay[klbl[ind]] += r;
                    //------------------------------------
                    //edgesum[klabels[ind]] += edgemag[ind];
                    //------------------------------------
                    clustersize[klbl[ind]] += 1.0;
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
                ksdl[k] = sigmal[k]*inv[k];
                ksda[k] = sigmaa[k]*inv[k];
                ksdb[k] = sigmab[k]*inv[k];
                ksdx[k] = sigmax[k]*inv[k];
                ksdy[k] = sigmay[k]*inv[k];
                //------------------------------------
                //edgesum[k] *= inv[k];
                //------------------------------------
            }}
            std::cout << "Calculated Centroid:::" << itr << std::endl;
            
            for (size_t i = 0; i < sz; i++)
            {
                kdist[i] = DBL_MAX;
            }
            
            semop(semId, turn1Signal, numOps);  // signal +p
        }
       
        
        
        

    }
    if (myIdx != -1)
    {
        exit(0);
    }
    
    // ============================== 
	// All this code is boiler-plate	
	// ============================== 

	std::cout << "In the parent: " << std::endl; 

	int status;	// catch the status of the child

	do  // in reality, mulptiple signals or exit status could come from the child
	{
        for (size_t i = 0; i < p; ++i)
        {
            pid_t w = waitpid(pid[i], &status, WUNTRACED | WCONTINUED);
            if (w == -1)
            {
                int errval = errno;
                std::cerr << "Error waiting for child process ("<< pid[i] <<")" << "\n" << strerror(errno) << std::endl;
                break;
            }

            if (WIFEXITED(status))
            {
                if (status > 0)
                {
                    std::cerr << "Child process ("<< pid[i] <<") exited with non-zero status of " << WEXITSTATUS(status) << std::endl;
                    continue;
                }
                else
                {
                    std::cout << "Child process ("<< pid[i] <<") exited with status of " << WEXITSTATUS(status) << std::endl;
                    continue;
                }
            }
            else if (WIFSIGNALED(status))
            {
                std::cout << "Child process ("<< pid[i] <<") killed by signal (" << WTERMSIG(status) << ")" << std::endl;
                continue;			
            }
            else if (WIFSTOPPED(status))
            {
                std::cout << "Child process ("<< pid[i] <<") stopped by signal (" << WSTOPSIG(status) << ")" << std::endl;
                continue;			
            }
            else if (WIFCONTINUED(status))
            {
                std::cout << "Child process ("<< pid[i] <<") continued" << std::endl;
                continue;
            }
        }
		
	}
	while (!WIFEXITED(status) && !WIFSIGNALED(status));
    
    for ( auto i : pid)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    
    
    
    // toc
    end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "[Info] SLIC using: " << duration.count() << " ms" << std::endl;

    // cv::Mat output_l(1, height*width, CV_8UC1);
    // cv::Mat output_a(1, height*width, CV_8UC1);
    // cv::Mat output_b(1, height*width, CV_8UC1);
    // std::vector<cv::Mat> output;


    // for (size_t i = 0; i < height*width; i++)
    // {
    //     int idx = klabels[i];
    //     output_l.at<unsigned char>(1, i) = (unsigned int)kseedsl[idx];
    //     output_a.at<unsigned char>(1, i) = (unsigned int)kseedsa[idx];
    //     output_b.at<unsigned char>(1, i) = (unsigned int)kseedsb[idx];
    //     // std::cout << (unsigned char)kseedsl[idx] << ' ';
    // }
    // std::cout << std::endl;
    
    // std::vector<int> mshape({height, width});

    // output_l = output_l.reshape(1, mshape);
    // output_a = output_a.reshape(1, mshape);
    // output_b = output_b.reshape(1, mshape);

    // output.push_back(output_l);
    // output.push_back(output_a);
    // output.push_back(output_b);


    // cv::Mat fin_img(img.size(), CV_8UC3);
    // cv::merge(output, fin_img);

    // std::cout << fin_img.at<cv::Vec3b>(0, 3)[0] << " " << fin_img.at<cv::Vec3b>(0, 3)[1] << " " << fin_img.at<cv::Vec3b>(0, 3)[2] << std::endl;

    // cv::Mat output_rgb(fin_img.size(), CV_8UC3);
    // cv::cvtColor(fin_img, output_rgb, cv::COLOR_Lab2BGR);
    
    
    
    
    
    for (size_t i=0; i < numk; i++)
    {
        std::cout << ksdl[i] << ' ';
    }
    std::cout << std::endl;
    
    
    
    
    

    DrawContoursAroundSegments(img, klbl, width, height);

    cv::imwrite("./" + output_filename + ".png", img);

    std::cout << "[Info] Output image saved in build/" << output_filename << ".png" << std::endl;
    sleep(0.01);
    
    

    
    // Clean shared memory
	shmctl(shmlId, IPC_RMID, NULL);
	shmctl(shmaId, IPC_RMID, NULL);
	shmctl(shmbId, IPC_RMID, NULL);
	shmctl(shmxId, IPC_RMID, NULL);
	shmctl(shmyId, IPC_RMID, NULL);
    shmctl(shmDistId, IPC_RMID, NULL);
    shmctl(shmLabelId, IPC_RMID, NULL);
    shmctl(shmNumkId, IPC_RMID, NULL);
	semctl(semId, 0, IPC_RMID);	

    return 0;
}




pid_t spawnChild()
{
    pid_t ch_pid = fork();
    if (ch_pid == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    } 
    
    std::cout << "spawn child with pid - " << ch_pid << std::endl;
    return ch_pid;
}























// ----------------------------------------------------------------------------------------
void parallelWork(
            double * ksdl,
            double * ksda,
            double * ksdb,
            double * ksdx,
            double * ksdy,
            double * kdist,
               int * klbl,
               int   n,
             double* lvec,
             double* avec,
             double* bvec,
                 int offset,
              double invwt,
          const int& width,
          const int& height)
{
    int x1, y1, x2, y2;
    double l, a, b;
    double dist;
    double distxy;
    
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
                kdist[i] = dist;
                klbl[i]  = n;
//                 std::cout << "test::::" << n << std::endl;
            }
        }
    }
    
}



/*
void SLIC_PerformSuperpixel(std::vector<double>& kseedsl, 
                            std::vector<double>& kseedsa, 
                            std::vector<double>& kseedsb,
                            std::vector<double>& kseedsx,
                            std::vector<double>& kseedsy,
                                            int* klabels,       // number of superpixel
                                      const int& STEP,          // S, neighbor size
                                   const double& M,             // Compactness factor. ranging from 10 to 40 
                                             int m_height,        // height of the image
                                             int m_width,         // width of the image
                                         double* m_lvec,
                                         double* m_avec,
                                         double* m_bvec,
                                       const int p
                            )
{
    
    pid_t pid[p];
    
    // =============================
    // BEGIN: Do the Semaphore Setup
    // The semaphore will be a mutex
    // =============================
    int semId[p]; 			            // ID of semaphore set
    key_t semKey = 123459; 		    // key to pass to semget(), key_t is an IPC key type defined in sys/types
    int semFlag = IPC_CREAT | 0666; // Flag to create with rw permissions

    int semCount = 1; 		        // number of semaphores to pass to semget()
    int numOps = 1; 		        // number of operations to do
    
    
    for (size_t i = 0; i < p; i++)
    {
        // Create semaphores
        semId[i] = semget(semKey++, semCount, semFlag);
        if ( semId[i] == -1)
        {
            std::cerr << "Failed to semget(" << semKey << "," << semCount << "," << semFlag << ")" << std::endl;
            exit(1);
        }
        else
        {
            std::cout << "Successful semget resulted in (" << semId << ")" << std::endl;
        }
        
        // Initialize the semaphore
        union semun {
            int val;
            struct semid_ds *buf;
            ushort * array;
        } argument;

        argument.val = p; // NOTE: We are setting this to one to make it a MUTEX
        if( semctl(semId[i], 0, SETVAL, argument) < 0)
        {
            std::cerr << "Init: Failed to initialize (" << semId << ")" << std::endl; 
            exit(1);
        }
        else
        {
            std::cout << "Init: Initialized (" << semId << ")" << std::endl; 
        }
    }
    // =============================
	// END: Do the Semaphore Setup
	// =============================
    std::cout << "semIds:: ";
    for (auto i : semId){
        std::cout << i << ' ';
    }
    std::cout << std::endl;
    
    
    

    const int sz = m_height * m_width;
    const int numk = kseedsl.size();
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
//     klabels = new int[sz];

    // =========================================
	// BEGIN: Do the Shared Memory Segment Setup
	// 
	// =========================================

    int shmId; 			// ID of shared memory segment
	key_t shmKey = semKey++; 		// key to pass to shmget(), key_t is an IPC key type defined in sys/types
	int shmFlag = IPC_CREAT | 0666; // Flag to create with rw permissions
	
	// This will be shared:
	int * sharedIndexPtr = NULL;
    
    if ((shmId = shmget(shmKey, (sz) * sizeof(int), shmFlag)) < 0)
	{
		std::cerr << "Init: Failed to initialize shared memory (" << shmId << ")" << std::endl; 
		exit(1);
	}
    if ((klabels = (int *)shmat(shmId, NULL, 0)) == (int *) -1)
	{
		std::cerr << "Init: Failed to attach shared memory (" << shmId << ")" << std::endl; 
		exit(1);
	}
    // =========================================
	// END: Do the Shared Memory Segment Setup
	// =========================================
    
    sharedIndexPtr = &klabels[0];
    *sharedIndexPtr = 0;
    
    
//     for (size_t i = 0; i < p; i++)
//     {
//         pid[i] = fork();
        
//         if ( pid[i] < 0 )
//         { 
//             std::cerr << "Could not fork!!! ("<< pid[i] <<")" << std::endl;
//             exit(1);
//         }
//         if (pid[i] == 0)
//         {
// //             std::cout << "This is a child process - " << pid[i] << std::endl;
//             _exit(0);
//         }
//         else
//         {
//             std::cout << "This is a parent process - " << pid[i] << std::endl;
//             std::cout << "M :: " << M << std::endl;
//         }
//     }

    for (size_t i = 0; i < p; i++)
    {
        pid[i] = fork();
        if (pid[i] == 0)
        {
            std::cout << "This is a child process - " << pid[i] << std::endl;
            exit(0);
        }
    }
    
    std::cout << "I just forked without error, I see ("<< pid <<")" << std::endl;
    
    
    
    for( int itr = 0; itr < 10; itr++ )
    {
//         std::cout << "[Info] ::Iteration:: " << itr << std::endl;
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

    // ============================== 
	// All this code is boiler-plate	
	// ============================== 

	std::cout << "In the parent: " << std::endl; 

	int status;	// catch the status of the child

	do  // in reality, mulptiple signals or exit status could come from the child
	{
        for (size_t i = 0; i < p; ++i)
        {
            pid_t w = waitpid(pid[i], &status, WUNTRACED | WCONTINUED);
            if (w == -1)
            {
                std::cerr << "Error waiting for child process ("<< pid <<")" << std::endl;
                break;
            }

            if (WIFEXITED(status))
            {
                if (status > 0)
                {
                    std::cerr << "Child process ("<< pid <<") exited with non-zero status of " << WEXITSTATUS(status) << std::endl;
                    continue;
                }
                else
                {
                    std::cout << "Child process ("<< pid <<") exited with status of " << WEXITSTATUS(status) << std::endl;
                    continue;
                }
            }
            else if (WIFSIGNALED(status))
            {
                std::cout << "Child process ("<< pid <<") killed by signal (" << WTERMSIG(status) << ")" << std::endl;
                continue;			
            }
            else if (WIFSTOPPED(status))
            {
                std::cout << "Child process ("<< pid <<") stopped by signal (" << WSTOPSIG(status) << ")" << std::endl;
                continue;			
            }
            else if (WIFCONTINUED(status))
            {
                std::cout << "Child process ("<< pid <<") continued" << std::endl;
                continue;
            }
        }
		
	}
	while (!WIFEXITED(status) && !WIFSIGNALED(status));
    
    for ( auto i : pid)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;

}
*/

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