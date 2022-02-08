#include <boost/thread/mutex.hpp>
#include "DistanceThreader.hpp"
#include <iostream>
#include <stdexcept>
#include <boost/thread/thread.hpp>

#include <cfloat>

zhang::DistanceThreader::DistanceThreader (std::vector<double> * ksdl, 
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
                                           int numk)
{
    std::cout << "Creating DistanceThreader... (" << p << " threads)" << std::endl;
    
    _kseedsl = ksdl;
    _kseedsa = ksda;
    _kseedsb = ksdb;
    _kseedsx = ksdx;
    _kseedsy = ksdy;
    _kdist   = kdist;
    _klbl    = klbl;
    _lvec    = lvec;
    _avec    = avec;
    _bvec    = bvec;
    
    _offset  = offset;
    _invwt   = invwt;
    _width   = width;
    _height  = height;
    _p       = p;
    _numk    = numk;
    
    _mutex = new boost::mutex();
    
}

zhang::DistanceThreader::~DistanceThreader ()
{
    std::cout << "Destructing DistanceThreader" << std::endl;
    
    delete _mutex;
}

void zhang::DistanceThreader::distvec_reset(){
    (*_kdist).assign(_width*_height, DBL_MAX);
}

void zhang::DistanceThreader::update(std::vector<double> * ksdl,
            std::vector<double> * ksda,
            std::vector<double> * ksdb,
            std::vector<double> * ksdx,
            std::vector<double> * ksdy,
            std::vector<double> * kdist,
                          int * klbl)
{
//     ksdl = _kseedsl;
//     ksda = _kseedsa;
//     ksdb = _kseedsb ;
//     ksdx = _kseedsx;
//     ksdy = _kseedsy;
//     kdist = _kdist;
//     klbl = _klbl;
    _kseedsl = ksdl;
    _kseedsa = ksda;
    _kseedsb = ksdb;
    _kseedsx = ksdx;
    _kseedsy = ksdy;
    _kdist   = kdist;
    _klbl    = klbl;
}

void zhang::DistanceThreader::operator()(int myIdx)
{
    std::cout << "in operator - (" << myIdx << ")" << std::endl;

    
    for (int n = 0; n < _numk; n++)
    {
    
        if ( n % _p == myIdx)
        {
//             boost::mutex::scoped_lock lock(*(_mutex));
            int x1, y1, x2, y2;
            double l, a, b;
            double dist;
            double distxy;
            
            y1 = std::max(0.0,              (*_kseedsy)[n]-_offset);
            y2 = std::min((double)_height,  (*_kseedsy)[n]+_offset);
            x1 = std::max(0.0,              (*_kseedsx)[n]-_offset);
            x2 = std::min((double)_width,   (*_kseedsx)[n]+_offset);
            
            std::cout << "offset::" << _offset << "  kseedsx::" << (*_kseedsx)[n] << "  kseedsy::" << (*_kseedsy)[n] << "  y1::" << y1 << "  y2::" << y2 << "  x1::" << x1 << "  x2::" << x2 << std::endl;
            for( int y = y1; y < y2; y++ )
            {
                for( int x = x1; x < x2; x++ )
                {
                    int i = y*_width + x;
                    // get the value of l a b in the pixel
                    l = _lvec[i];
                    a = _avec[i];
                    b = _bvec[i];
                    dist =          (l - (*_kseedsl)[n])*(l - (*_kseedsl)[n]) +
                                    (a - (*_kseedsa)[n])*(a - (*_kseedsa)[n]) +
                                    (b - (*_kseedsb)[n])*(b - (*_kseedsb)[n]);
                    distxy =        (x - (*_kseedsx)[n])*(x - (*_kseedsx)[n]) +
                                    (y - (*_kseedsy)[n])*(y - (*_kseedsy)[n]);

                    //--------------------------------------------------------------
                    //dist = sqrt(dist) + sqrt(distxy*invwt);//this is more exact
                    dist += distxy*_invwt;
                    //--------------------------------------------------------------
                   
                    
//                         boost::mutex::scoped_lock lock(*(_mutex));
                    if( dist < (*_kdist).at(i) )
                    {
//                         std::cout << "kdist before::" << (*_kdist).at(i) << " dist::" << dist;
                        (*_kdist).at(i) = dist;
                        _klbl[i]  = n;
                        
//                         std::cout << " kdist after::" << (*_kdist)[i] << std::endl;
                    }
                    
                    
                 
                    
                }
            }
        }
        
    }
}