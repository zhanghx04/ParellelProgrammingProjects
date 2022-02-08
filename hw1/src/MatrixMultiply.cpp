#include "MatrixMultiply.hpp"

#include <exception>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <numeric>

scottgs::MatrixMultiply::MatrixMultiply() 
{
	;
}

scottgs::MatrixMultiply::~MatrixMultiply()
{
	;
}


scottgs::FloatMatrix scottgs::MatrixMultiply::operator()(const scottgs::FloatMatrix& lhs, const scottgs::FloatMatrix& rhs) const
{
	// Verify acceptable dimensions
	if (lhs.size2() != rhs.size1())
		throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

	scottgs::FloatMatrix result(lhs.size1(),rhs.size2());


	// YOUR ALGORIHM WITH COMMENTS GOES HERE:
	
	
	/*
		##### Method 3 #####
        Based on Method 2 store the sizes into variables
    */

	int r = lhs.size1();
	int c = rhs.size2();
	int size = lhs.size2();	// get the size for multiplication
	scottgs::FloatMatrix rhs_trans = boost::numeric::ublas::trans(rhs);

	for (int i = 0; i < r*c; i++) {
		float val = 0;
		for (int s = 0; s < size; s++) {
			val += (lhs.data()[(i/c)*size + s] * rhs_trans.data()[(i%c)*size + s]);
		}
		result.data()[i] = val;
	}

	/*
		##### Method 2 #####
        Transposed right hand side matrix, and using data() to access values
    */
	// scottgs::FloatMatrix rhs_trans = boost::numeric::ublas::trans(rhs);

	// for (int i = 0; i < lhs.size1()*rhs.size2(); i++) {
	// 	float val = 0;
	// 	for (int s = 0; s < lhs.size2(); s++) {
	// 		val += (lhs.data()[(i/rhs.size2())*lhs.size2() + s] * rhs_trans.data()[(i%rhs.size2())*lhs.size2() + s]);
	// 	}
	// 	result.data()[i] = val;
	// }

	/*
		##### Method 1 #####
        Use simple three nested loop to implement
    */

	// for (int r = 0; r < result.size1(); r++) {
	// 	for (int c = 0; c < result.size2(); c ++) {
	// 		float val {0};
			
	// 		// Start to compute
	// 		for (int s = 0; s < lhs.size2(); s++) {
	// 			val += (lhs(r, s) * rhs(s, c));
	// 		}
	// 		result(r, c) = val;
	// 	}
	// }
 



	return result;
}

scottgs::FloatMatrix scottgs::MatrixMultiply::multiply(const scottgs::FloatMatrix& lhs, const scottgs::FloatMatrix& rhs) const
{
	// Verify acceptable dimensions
	if (lhs.size2() != rhs.size1())
		throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

	return boost::numeric::ublas::prod(lhs,rhs);
}

