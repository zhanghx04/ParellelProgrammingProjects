#ifndef DISTANCETHREADER_HPP
#define DISTANCETHREADER_HPP

#include <boost/thread/mutex.hpp>
#include <boost/utility.hpp> // for noncopyable
#include <stdio.h>
#include <vector>


namespace zhang {
    
class DistanceThreader : boost::noncopyable
{
    public:
        DistanceThreader(   std::vector<double> * ksdl,
                            std::vector<double> * ksda,
                            std::vector<double> * ksdb,
                            std::vector<double> * ksdx,
                            std::vector<double> * ksdy,
                            std::vector<double> * kdist,
                                           int * klbl,
                                         double* lvec,
                                         double* avec,
                                         double* bvec,
                                             int offset,
                                          double invwt,
                                       const int width,
                                       const int height,
                                             int p,
                                             int numk);
        ~DistanceThreader();
        void distvec_reset();
        void update(std::vector<double> * ksdl,
                    std::vector<double> * ksda,
                    std::vector<double> * ksdb,
                    std::vector<double> * ksdx,
                    std::vector<double> * ksdy,
                    std::vector<double> * kdist,
                                  int * klbl);
        void operator()(int myIndex);
    
    private:
        // Mutex
        boost::mutex * _mutex;
        
        // kseeds
        std::vector<double> * _kseedsl;
        std::vector<double> * _kseedsa;
        std::vector<double> * _kseedsb;
        std::vector<double> * _kseedsx;
        std::vector<double> * _kseedsy;
        
        // kdistance
        std::vector<double> * _kdist;
        
        // labels
        int *               _klbl;
        
        // lab vectors
        double * _lvec;
        double * _avec;
        double * _bvec;
        
        int _offset;
        double _invwt;
        
        int _width;
        int _height;
        int _p;
        int _numk;
        
};
    
}; // end of hzny2 namespace

#endif // _DISTANCETHREADER_HPP