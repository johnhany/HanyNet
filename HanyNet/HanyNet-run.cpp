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

#include "HanyNet.h"
#include "Utils.h"

#define _HANY_NET_LOAD_MNIST
//#define _HANY_NET_LOAD_SAMPLE_FROM_PIC
//#define _HANY_NET_CAPTURE_FACE_FROM_CAMERA

#define _HANY_NET_PREDICT_MNIST
//#define _HANY_NET_PREDICT_IMAGE_SERIES
//#define _HANY_NET_PREDICT_VEDIO_SERIES
//#define _HANY_NET_PREDICT_CAMERA

//#define _HANY_NET_WITH_LABEL_NAMES

string parameter_file = "parameters.xml";
string pretrained_cnn_file = "HanyNet_pretrained_MNIST.xml";
string trained_cnn_file = "HanyNet_trained.xml";
string output_video_file = "result.avi";

string camera_window_name = "HanyNet Testing";

string sample_file_pre = "path_to_your_dataset\\";
string label_file = "path_to_your_dataset_label_file.txt";

string haar_file = "haarcascade_frontalface_alt2.xml";

vector<pair<int, string>> label_list;

int sample_num = 20;		//Number of samples for each class

int main(int argc, char* argv[])
{
	CNN net;

	double time_cost;


	//-------- CNN Initializing --------
	//----------------------------------

	//Read parameters file
	net.readPara(parameter_file);


	//-------- Load Dataset ------------
	//----------------------------------

#ifdef _HANY_NET_WITH_LABEL_NAMES
	ifstream read_label(label_file);
	for(int c = 0; c < net.class_count; c++) {
		string new_label_name;
		read_label >> new_label_name;
		label_list.push_back(make_pair(c, new_label_name));
	}
#endif

#ifdef _HANY_NET_LOAD_MNIST
#ifdef _HANY_NET_PRINT_MSG
	cout << "Loading MNIST dataset..." << endl;
	time_cost = (double)getTickCount();
#endif

	loadMNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", net.train_set);
	loadMNIST("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", net.test_set);

#ifdef _HANY_NET_PRINT_MSG
	time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
	cout << "Load samples done." << endl << "Time cost: " << time_cost << "s." << endl << endl;
#endif
#endif

#ifdef _HANY_NET_TRAIN_FROM_SCRATCH

#ifdef _HANY_NET_LOAD_SAMPLE_FROM_PIC
#ifdef _HANY_NET_PRINT_MSG
	cout << "Loading samples..." << endl;
	time_cost = (double)getTickCount();
#endif

	for(int c = 0; c < net.class_count; c++) {

		for(int i = 0; i < sample_num; i++) {
			string file_name = sample_file_pre + to_string(c) + "_" + to_string(i) + ".jpg";
			Mat img_read = imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);
			if(img_read.data == NULL) {
				break;
			}
			Mat img_nor;
			resize(img_read, img_nor, Size(net.sample_width, net.sample_height));

			net.train_set.push_back(make_pair(img_nor, (uchar)(c)));
		}
	}

#ifdef _HANY_NET_PRINT_MSG
	time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
	cout << "Load samples done." << endl << "Time cost: " << time_cost << "s." << endl << endl;
#endif
#endif


#ifdef _HANY_NET_CAPTURE_FACE_FROM_CAMERA
#ifdef _HANY_NET_PRINT_MSG
	cout << "Capturing samples..." << endl;
	time_cost = (double)getTickCount();
#endif

	VideoCapture cap_in(0);
	if(!cap_in.isOpened()) {
		cout << "Cannot access camera. Press ANY key to exit." << endl;
		cin.get();
		exit(-1);
	}

	CascadeClassifier cascade_in;
	cascade_in.load(haar_file);

	Mat frame;
	int frame_count = 0;
	int capture_count = 0;
	int class_idx = 0;
	int class_count = 0;
	bool sample_suff = false;
	bool cap_sample = true;

	while(cap_in.read(frame)) {
		capture_count++;

		vector<Rect> faces;
		Mat frame_gray, img_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, img_gray);
		cascade_in.detectMultiScale(img_gray, faces, 1.1, 2, 0, Size(120, 120));

		int face_area = 0;
		int face_idx = 0;

		if(faces.size() > 0) {
			for(int f = 0; f < faces.size(); f++) {
				if(faces[f].area() > face_area) {
					face_area = faces[f].area();
					face_idx = f;
				}
			}

			rectangle(frame, faces[face_idx], Scalar(255, 0, 0), 3);

			if(frame_count % 5 == 0 && cap_sample && !sample_suff) {
				Mat face, face_nor;
				img_gray(faces[face_idx]).copyTo(face);

				resize(face, face_nor, Size(net.sample_width, net.sample_height));

				net.train_set.push_back(make_pair(face_nor, (uchar)class_idx));
				class_count++;
			}
		}

		putText(frame, "Class: " + to_string(class_idx), Point(50, 100), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
		putText(frame, "Sample: " + to_string(class_count), Point(50, 150), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);

		if(sample_suff) {
			putText(frame, "Enough samples. Press SPACE.", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
		}else {
			putText(frame, "Capturing...", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
		}
		if(!cap_sample) {
			putText(frame, "Wait for another person. Press SPACE.", Point(50, 200), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
		}

		imshow(camera_window_name, frame);

		if(class_count >= sample_num) {
			sample_suff = true;
		}

		frame_count++;
		int key = waitKey(20);
		if(key == 27){
			cap_in.release();
			break;
		} else if(key == ' ') {
			if(cap_sample && sample_suff) {
				cap_sample = false;
				continue;
			}
			if(!cap_sample && sample_suff) {
				cap_sample = true;
				sample_suff = false;
				class_idx++;
				class_count = 0;
				continue;
			}
		}
	}

#ifdef _HANY_NET_PRINT_MSG
	time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
	cout << "Load samples done." << endl << "Time cost: " << time_cost << "s." << endl << endl;
#endif
#endif

#endif


	//-------- CNN Initializing --------
	//----------------------------------

#ifdef _HANY_NET_PRINT_MSG
	cout << "Initializing neural networks..." << endl;
	time_cost = (double)getTickCount();
#endif

	//Initialize CNN with knowledge of samples
	net.initCNN();

#ifdef _HANY_NET_PRINT_MSG
	time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
	cout << "Total number of samples: " << (int)(net.train_set.size() + net.test_set.size()) << endl;
	cout << "Initializing neural networks done." << endl << "Time cost: " << time_cost << "s." << endl << endl;
#endif


	//Load pre-trained CNN parameters from file and continue to train
//	net.uploadCNN(pretrained_cnn_file);

	//-------- CNN Training ----------
	//--------------------------------

#ifdef _HANY_NET_TRAIN_FROM_SCRATCH
#ifdef _HANY_NET_PRINT_MSG
	cout << "Start training CNN..." << endl;
	time_cost = (double)getTickCount();
#endif

	//Train CNN with train sample set
	net.trainCNN();

#ifdef _HANY_NET_PRINT_MSG
	time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
	cout << "CNN training done." << endl << "Time cost: " << time_cost << "s." << endl << endl;
#endif

	for(int i = 0; i < net.time_ff.size(); i++) {
		cout << "FeedForward stage " << i << ":  " << net.time_ff[i] << "s" << endl;
	}
	for(int i = 0; i < net.time_bp.size(); i++) {
		cout << "BackPropagation stage " << i << ":  " << net.time_bp[i] << "s" << endl;
	}

	//Draw stage loss graph
	Mat stage_loss_graph = Mat::zeros(600, 1100, CV_8UC3);
	Point2d pt1, pt2;
	pt1 = Point2d(50.0, 50.0);
	for(int stage = 0; stage < net.stage_loss.size(); stage++) {
		pt2 = Point2d(50.0 + 1200.0 / net.stage_loss.size() * stage, 550.0 - 500.0 * net.stage_loss[stage] / net.stage_loss[0]);
		line(stage_loss_graph, pt1, pt2, Scalar(255, 255, 255));
		pt1 = pt2;
	}
	imshow("Stage Loss Graph", stage_loss_graph);
	imwrite("stage_loss_graph.jpg", stage_loss_graph);
	waitKey(10);

#endif


	//-------- Save Trained Network -----
	//-----------------------------------

#ifdef _HANY_NET_TRAIN_FROM_SCRATCH
#ifdef _HANY_NET_PRINT_MSG
	cout << "Dumping trained CNN parameters to file " << pretrained_cnn_file << "..." << endl;
#endif

	//Dump trained CNN parameters to file
	net.downloadCNN(trained_cnn_file);

#ifdef _HANY_NET_PRINT_MSG
	cout << "Dumping trained CNN parameters to file done." << endl << endl;
#endif
#endif


	//-------- Load Pre-trained Network -----
	//---------------------------------------

#ifndef _HANY_NET_TRAIN_FROM_SCRATCH
#ifdef _HANY_NET_PRINT_MSG
	cout << "Loading pre-trained CNN parameters from file " << pretrained_cnn_file << "..." << endl;
#endif

	//Load pre-trained CNN parameters from file
	net.uploadCNN(pretrained_cnn_file);

#ifdef _HANY_NET_PRINT_MSG
	cout << "Loading pre-trained CNN parameters from file done." << endl << endl;
#endif
#endif


	//-------- Predict New Samples-------
	//--------------------------------------

#ifdef _HANY_NET_PREDICT_MNIST
#ifdef _HANY_NET_PRINT_MSG
	cout << "Predicting MNIST test dataset..." << endl;
	time_cost = (double)getTickCount();
#endif

	//Calculate correctness ratio with test samples
	int total_correct_count = 0;
	for(int sample_idx = 0; sample_idx < net.test_set.size(); sample_idx++) {
		vector<Mat> input_sample;
		input_sample.push_back(net.test_set[sample_idx].first);
		vector<Mat> predict_result = net.predictCNN(input_sample);
		if((int)predict_result[0].ptr<uchar>(0)[0] == net.test_set[sample_idx].second) {
			total_correct_count++;
		}
	}
	double total_correct_ratio = (double)total_correct_count / net.test_set.size();

#ifdef _HANY_NET_PRINT_MSG
	time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
	cout << "MNIST testing done." << endl << "Time cost: " << time_cost << "s." << endl;
	cout << "Total correctness ratio: " << total_correct_ratio << endl << endl;
#endif
#endif

#ifdef _HANY_NET_PREDICT_IMAGE_SERIES
#ifdef _HANY_NET_PRINT_MSG
	cout << "Predicting from image series..." << endl;
#endif

//	VideoWriter wri(output_video_file, CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));

	for(int c = 0; c < net.class_count; c++) {

		for(int i = 0; i < sample_num; i++) {
			string file_name = sample_file_pre + to_string(c) + "_" + to_string(i) + ".jpg";
			Mat img_read = imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);
			if(img_read.data == NULL) {
				break;
			}
			Mat img_nor, img_show;
			resize(img_read, img_show, Size(400, 400));
			resize(img_read, img_nor, Size(net.sample_width, net.sample_height));

			vector<Mat> input_sample;
			input_sample.push_back(img_nor);

			vector<Mat> predict_result = net.predictCNN(input_sample);

			int pred_rst = (int)predict_result[0].ptr<uchar>(0)[0];
			if(pred_rst <= net.class_count)
				putText(img_show, label_list[pred_rst].second, Point(10, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);

			putText(img_show, to_string(c)+"-"+to_string(i), Point(img_show.cols-80, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);

			int frame_count = 25;
			while(--frame_count) {
//				wri.write(img_show);
			}
			imshow(camera_window_name, img_show);

			int key_get = waitKey(20);
			switch(key_get) {
			case 27:
//				wri.release();
				return 0;
			default:
				break;
			}
		}
	}

#endif


#ifdef _HANY_NET_PREDICT_VEDIO_SERIES
#ifdef _HANY_NET_PRINT_MSG
	cout << "Predicting from video series..." << endl;
#endif

	VideoWriter wri(output_video_file, CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));
	namedWindow(camera_window_name);

	CascadeClassifier cascade_out;
	cascade_out.load(haar_file);

	for(int c = 1; c <= net.class_count; c++) {
		string file_name = "path_to_face_videos\\" + to_string(c) + ".wmv";
		VideoCapture cap(file_name);
		if(!cap.isOpened())
			continue;

		Mat img_read;
		while(cap.read(img_read)) {
			Mat img_gray, nor_gray, img_show;
			img_read.copyTo(img_show);
			cvtColor(img_read, img_gray, CV_BGR2GRAY);

			vector<Rect> faces;
			equalizeHist(img_gray, img_gray);
			cascade_out.detectMultiScale(img_gray, faces, 1.1, 2, 0, Size(120, 120));

			for(int f = 0; f < faces.size(); f++) {
				rectangle(img_show, faces[f], Scalar(0, 255, 255), 3);

				resize(img_gray(faces[f]), nor_gray, Size(net.sample_width, net.sample_height));
				vector<Mat> input_sample;
				input_sample.push_back(nor_gray);

				vector<Mat> predict_result = net.predictCNN(input_sample);
				
				int pred_rst = (int)predict_result[0].ptr<uchar>(0)[0];
				if(pred_rst <= net.class_count)
					putText(img_show, to_string(pred_rst), Point(faces[f].x+faces[f].width, faces[f].y+faces[f].height), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
			}

			int frame_count = 2;
			while(--frame_count) {
				wri.write(img_show);
			}
			imshow(camera_window_name, img_show);

			int key_get = waitKey(20);
			switch(key_get) {
			case 27:
				wri.release();
				return 0;
			default:
				break;
			}
		}
	}
	wri.release();
#endif

#ifdef _HANY_NET_PREDICT_CAMERA
#ifdef _HANY_NET_PRINT_MSG
	cout << "Predicting from camera..." << endl;
#endif

	VideoCapture cap_out(0);
	if(!cap_out.isOpened()) {
		cout << "Cannot access camera." << endl;
		cin.get();
		exit(-1);
	}

	CascadeClassifier cascade_out;
	cascade_out.load(haar_file);

//	VideoWriter wri(output_video_file, CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));

	Mat src_frame;

	namedWindow(camera_window_name);

	Mat img_read;
	while(cap_out.read(img_read)) {
		Mat img_gray, nor_gray, img_show;
		img_read.copyTo(img_show);
		cvtColor(img_read, img_gray, CV_BGR2GRAY);

		vector<Rect> faces;
		equalizeHist(img_gray, img_gray);
		cascade_out.detectMultiScale(img_gray, faces, 1.1, 2, 0, Size(120, 120));

		for(int f = 0; f < faces.size(); f++) {
			rectangle(img_show, faces[f], Scalar(0, 255, 255), 3);

			resize(img_gray(faces[f]), nor_gray, Size(net.sample_width, net.sample_height));
			vector<Mat> input_sample;
			input_sample.push_back(nor_gray);

			vector<Mat> predict_result = net.predictCNN(input_sample);

			int pred_rst = (int)predict_result[0].ptr<uchar>(0)[0];
			if(pred_rst <= net.class_count)
				putText(img_show, label_list[pred_rst].second, Point(faces[f].x+faces[f].width, faces[f].y+faces[f].height), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);

		}

		int frame_count = 2;
		while(--frame_count) {
//			wri.write(img_show);
		}
		imshow(camera_window_name, img_show);

		int key_get = waitKey(20);
		if(key_get == 27) {
//			wri.release();
			cap_out.release();
			return 0;
		}
	}
#endif

	cout << "Press any key to quit..." << endl;
//	waitKey(0);
	cin.get();

	return 0;
}