#include <iostream>

#include <stdio.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

std::map<int, int> getHist(cv::Mat& matrix, std::vector<int>& range, bool isL); 

int main (int argc, char** argv) {
	
	// Check if there is no image input
	// if (argc != 2) {
	// 	std::cout << "Input image missing..." << std::endl;
	// 	return -1;
	// }
	std::string img_name, img_path;
	img_path = "../Astronaught.png";

	if ( argc >= 2) {
		// Read the input image path
		std::cout << "[Info] Read image from path: " << argv[1] << std::endl;
		img_path = argv[1];
	}

	img_name = img_path.substr(img_path.find_last_of("/\\") + 1);

	// Read image
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	
	cv::Mat image;
	// image = cv::imread(argv[1], 1);
	image = cv::imread(img_path, cv::IMREAD_COLOR);

	// cv::Mat image_float(image.size(), CV_32FC3);
	// image.convertTo(image_float, CV_32FC3);


	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	if ( !image.data ) {
		std::cout << "[Error] No image data." << std::endl;
		return -1;
	}

	std::cout << "[Info]              Image Name: " << img_name << std::endl;
	std::cout << "[Info]        Image Dimensions: " << image.rows << " x " << image.cols << " x " << image.channels() << std::endl;
	std::cout << "[Time]      Reading image took: " << duration.count() << " milliseconds" << std::endl;

	// cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
	// cv::imshow("Display Image", image);
	// cv::waitKey(0);
	// cv::destroyAllWindows();

	// Convert RGB to Lab
	start = std::chrono::high_resolution_clock::now();
	
	cv::Mat image_lab(image.size(), CV_32FC3);
	cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);
 
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "[Time] Convert RGB to Lab took: " << duration.count() << " milliseconds" << std::endl;

	image.release();	// Deallcate the image
	// image_float.release();

	// cv::namedWindow("Display Lab Image", cv::WINDOW_AUTOSIZE);
	// cv::imshow("Display Lab Image", image_lab);
	// cv::waitKey(0);
	// cv::destroyAllWindows();

	
	
	std::vector<cv::Mat> lab(3);
	cv::split(image_lab, lab);

	image_lab.release();

	// Display L a b channels
	// cv::namedWindow("Lab", cv::WINDOW_AUTOSIZE);
	// cv::imshow("L", lab[0]);
	// cv::imshow("a", lab[1]);
	// cv::imshow("b", lab[2]);

	// cv::waitKey(0);
	// cv::destroyAllWindows();



	// Histogram
 	start = std::chrono::high_resolution_clock::now();
	
	std::vector<int> l_range({0, 100});
	std::vector<int> ab_range({-127, 127});

	std::map<int, int> dict_l, dict_a, dict_b;

	dict_l = getHist(lab[0], l_range, true);
	dict_a = getHist(lab[1], ab_range, false);
	dict_b = getHist(lab[2], ab_range, false);

	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "[Time]      Compute histograms: " << duration.count() << " milliseconds" << std::endl;

	std::cout << "L Histogram: [";
	for(auto it = dict_l.cbegin(); it != dict_l.cend(); ++it){
		std::cout << "(" << it->first << ", " << it->second << "), ";
	}
	std::cout << "]\n" << std::endl;
	
	std::cout << "A Histogram: [";
	for(auto it = dict_a.cbegin(); it != dict_a.cend(); ++it){
		std::cout << "(" << it->first << ", " << it->second << "), ";
	}
	std::cout << "]\n" << std::endl;
	
	std::cout << "B Histogram: [";
	for(auto it = dict_b.cbegin(); it != dict_b.cend(); ++it){
		std::cout << "(" << it->first << ", " << it->second << "), ";
	}
	std::cout << "]\n" << std::endl;
	
	
	

	std::vector<cv::Mat>().swap(lab);	// memory

	std::cout << "[Info] done!" << std::endl;


	return 0;
}


std::map<int, int> getHist(cv::Mat& matrix, std::vector<int>& range, bool isL) {

	// get the size
	int row {matrix.rows};
	int col {matrix.cols};

	// std::cout << "value: " << (int) (matrix.at<unsigned char>(0,0) - 180) << std::endl;

	std::map<int, int> dict;
	int ii = 0;
	for (int i=0; i < row; i++) {
		for (int j=0; j < col; j++) {
			int val;
			if (isL){
				val = (int)(matrix.at<unsigned char>(i, j) * 100 / 255);
			} else {
				val = (int)(matrix.at<unsigned char>(i, j) - 180);
			}
			
			if ( dict.find(val) == dict.end() ){
				dict.insert(std::pair<int, int>(val, 1));
			} else {
				// std::cout << dict[val] << std::endl;
				dict[val] += 1;
			}
		}
	}

	return dict;

	// std::cout << "pixel value: " << (unsigned int)matrix.at<unsigned char>(200,180) << std::endl;
	// for(auto it = dict.cbegin(); it != dict.cend(); ++it){
	// 	std::cout << "(" << it->first << ", " << it->second << "), ";
	// }
	// std::cout << "]\n" << std::endl;
}
