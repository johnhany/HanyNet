/************************************************************************/
/*                        HanyNet  by John Hany                         */
/*																		*/
/*	  A naive implementation of Deep Convolutional Neural Networks.		*/
/*	Based on LeNet which is composed of several convolutional layers	*/
/*	and subsampling layers followed by a fully connnected layer.		*/
/*																		*/
/*	Released under MIT license.											*/
/*																		*/
/*	Welcome to my blog http://johnhany.net/, if you can read Chinese:)	*/
/*																		*/
/************************************************************************/

#ifndef _HANY_NET_HEADER
#define _HANY_NET_HEADER

#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string>

#include "rapidxml\rapidxml.hpp"       
#include "rapidxml\rapidxml_utils.hpp"

//  ifdef, print info messages on screen
#define _HANY_NET_PRINT_MSG

//  ifdef, multi-threading is on
//	no multi-threading support for now
//#define _HANY_NET_MULTI_THREADING

//  ifndef, load CNN from file named 'CNN_parameters.xml'
//  ifdef, train CNN from scratch
//#define _HANY_NET_TRAIN_FROM_SCRATCH

//  ifdef, limit maximum sample number for debugging
//#define _HANY_NET_LIMIT_SAMPLE_NUM	10

//define some error codes
#define EXIT_CODE_PARA_INPUT_LAYER		1110
#define EXIT_CODE_PARA_HIDDEN_LAYER		1111
#define EXIT_CODE_PARA_ACTIV_FUNC		1112
#define EXIT_CODE_CNN_NOT_INIT			1113
#define EXIT_CODE_PARSE_MAT_INVALID		1114
#define EXIT_CODE_CAMERA_NOT_EXIST		1115

using namespace cv;
using namespace std;

struct layer {
	string type;	//'i', 'c', 's', or 'o' (as long as the first letter can match up)
	string func;	//'relu' or 'sigmoid' for conv layer, 'softmax' for output layer
	int maps;		//for 'c' only. number of kernels
	int size;		//for 'c' only. size of kernel
	int scale;		//for 's' only. scale of subsampling(pooling)
	int classes;	//for 'o' only. number of classes
};

class CNN {
private:
	int epoch_num;
	double gradient_alpha;
	bool initial_done;

	vector<int> layer_weight_idx;

	Mat sigmoid(Mat src);					//for CV_64FC1 image only
	Mat ReLU(Mat src);						//for single-channel image only
	Mat softmax(Mat src);					//for single-channel images only

	//Parse Mat from given xml file node, using RapidXML.
	//  Only accept CV_64FC1 Mat.
	Mat parseMatFromXMLd(string mat_rows, string mat_cols, string mat_data);

public:
	int group_samples;

	int sample_width;
	int sample_height;

	int class_count;

	double general_loss;
	double test_correct_ratio;

	vector<layer> net_struct;				//defines every layer
	vector<pair<Mat, uchar>> train_set;		//training samples
	vector<pair<Mat, uchar>> test_set;		//test samples
	vector<uchar> net_labels;				//contains test labels

	vector<vector<Mat>> net_nodes;			//contains layer nodes
	vector<vector<Mat>> net_deltas;			//delta values for weights update
	vector<vector<Mat>> net_weights;		//weight values between each layer
	vector<vector<Mat>> net_biases;			//bias values of each layer
	vector<double> stage_loss;				//loss after each training iteration

	vector<double> time_ff, time_bp;

	CNN();
	void readPara(string file_name);		//called before initCNN()
	void initCNN();					//initialize CNN and load MNIST
	void trainCNN();				//train CNN with back propagation
	void downloadCNN(string file_name);				//dump trained CNN parameters to file
	void uploadCNN(string file_name);				//load pre-trained CNN parameters from file

	//CNN reference. 'input' should contain CV_8UC1 images.
	//If input.size() >= group_samples, each sample will only be referenced once.
	//If input.size() < group_samples, especially when input.size() < group_samples/2, 
	//	each sample will be duplicated and duplicated samples will be referenced independently.
	//	For all duplicated samples respect to the same original sample, the maximum votes will
	//	be calculated. If there are more than one duplicated samples having the same maximum
	//	vote, take the one with maximum sum-up predict probability as the final predict result.
	vector<Mat> predictCNN(vector<Mat>& input);
};

#endif
