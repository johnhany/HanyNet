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

#ifndef _HANY_NET_UTILS_HEADER
#define _HANY_NET_UTILS_HEADER

#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>

using namespace cv;
using namespace std;

int reverseInt(int i);
void loadMNIST(string pic_filename, string label_filename, vector<pair<Mat, uchar>> &sample_set);

#endif
