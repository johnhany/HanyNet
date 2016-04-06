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

#include "Utils.h"

int reverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;

	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void loadMNIST(string pic_filename, string label_filename, vector<pair<Mat, uchar>> &sample_set) {
	ifstream pic_file(pic_filename, ios::binary);
	ifstream label_file(label_filename, ios::binary);

	if(pic_file.is_open() && label_file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		label_file.read((char*)&magic_number, sizeof(magic_number));
		pic_file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		label_file.read((char*)&number_of_images, sizeof(number_of_images));
		pic_file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);

		pic_file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		pic_file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);

		for(int i = 0; i < number_of_images; ++i)
		{
			uchar label = 0;
			label_file.read((char*)&label, sizeof(label));

			cv::Mat sample = Mat::zeros(n_rows, n_cols, CV_8UC1);
			for(int r = 0; r < n_rows; ++r)
			{
				for(int c = 0; c < n_cols; ++c)
				{
					uchar tmp = 0;
					pic_file.read((char*)&tmp, sizeof(tmp));
					sample.at<uchar>(r, c) = (int)tmp;
				}
			}
			sample_set.push_back(make_pair(sample, label));
		}
	}
}
