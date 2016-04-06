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

CNN::CNN() {

	initial_done = false;

}


Mat CNN::sigmoid(Mat src) {

	Size size = src.size();
	Mat result(size, CV_64FC1);
	if(src.isContinuous() && result.isContinuous()) {
		size.width *= size.height;
		size.height = 1;
	}
	for(int j = 0; j < size.height; j++) {
		const double* pin = src.ptr<double>(j);
		double* pout = result.ptr<double>(j);
		for(int i = 0; i < size.width; i++) {
			pout[i] = 1.0 / (1.0 + exp(-pin[i]));
		}
	}
	return result;

}


Mat CNN::ReLU(Mat src) {

	Size size = src.size();
	Mat result(size, CV_64FC1);
	if(src.isContinuous() && result.isContinuous()) {
		size.width *= size.height;
		size.height = 1;
	}
	for(int j = 0; j < size.height; j++) {
		const double* pin = src.ptr<double>(j);
		double* pout = result.ptr<double>(j);
		for(int i = 0; i < size.width; i++) {
			pout[i] = (pin[i] > 0) ? pin[i] : 0;
		}
	}
	return result;

}

Mat CNN::softmax(Mat src) {

	Size size = src.size();
	Mat result(size, CV_64FC1);

	for(int j = 0; j < size.height; j++) {
		const double* pin = src.ptr<double>(j);
		double* pout = result.ptr<double>(j);
		double sum = 0.0;
		for(int i = 0; i < size.width; i++) {
			sum += exp(pin[i]);
		}
		for(int i = 0; i < size.width; i++) {
			pout[i] = exp(pin[i]) / sum;
		}
	}
	return result;

}

Mat CNN::parseMatFromXMLd(string mat_rows, string mat_cols, string mat_data) {

	int rows = stoi(mat_rows);
	int cols = stoi(mat_cols);
	Mat result(rows, cols, CV_64FC1);
	double tmp;

	stringstream data(mat_data);
	for(int i = 0; i < rows; i++) {
		double* prow = result.ptr<double>(i);
		for(int j = 0; j < cols; j++) {
			data >> tmp;
			prow[j] = tmp;
		}
	}

	return result;

}

void CNN::readPara(string file_name) {

	rapidxml::file<> fdoc(file_name.c_str());
	rapidxml::xml_document<> doc;
	doc.parse<0>(fdoc.data());

	rapidxml::xml_node<>* root = doc.first_node();
	if(string(root->name()) == "hanynet_parameters") {

		group_samples = stoi(root->first_node("group_samples")->value());
		epoch_num = stoi(root->first_node("epoch_num")->value());
		gradient_alpha = stod(root->first_node("gradient_alpha")->value());
		sample_width = stoi(root->first_node("sample_width")->value());
		sample_height = stoi(root->first_node("sample_height")->value());

		rapidxml::xml_node<>* node = root->first_node("net_struct");

		for(rapidxml::xml_node<>* node_layer = node->first_node(); node_layer; node_layer = node_layer->next_sibling()) {
			
			layer new_layer;
			
			if(string(node_layer->first_attribute("type")->value()) == "i") {
				new_layer.type = "i";
			} else if(string(node_layer->first_attribute("type")->value()) == "c") {
				new_layer.type = "c";
				new_layer.func = string(node_layer->first_node("func")->value());
				new_layer.maps = stoi(node_layer->first_node("maps")->value());
				new_layer.size = stoi(node_layer->first_node("size")->value());
			} else if(string(node_layer->first_attribute("type")->value()) == "s") {
				new_layer.type = "s";
				new_layer.scale = stoi(node_layer->first_node("scale")->value());
			} else if(string(node_layer->first_attribute("type")->value()) == "o") {
				new_layer.type = "o";
				new_layer.func = string(node_layer->first_node("func")->value());
				new_layer.classes = stoi(node_layer->first_node("classes")->value());
				class_count = new_layer.classes;
			}

			net_struct.push_back(new_layer);

		}
	}

}

void CNN::initCNN() {

	int layer_idx = 0;
	int weight_idx = 0;
	RNG rng(2345);

	for(vector<layer>::iterator layer_itr = net_struct.begin(); layer_itr != net_struct.end(); layer_itr++) {
		if((*layer_itr).type[0] == 'i') {

			time_ff.push_back(0.0);

			//Fill input layer with Mat::zeros
			Size sample_size(sample_width, sample_height);
			vector<Mat> new_nodes(group_samples, Mat::zeros(sample_size, CV_64FC1));
			vector<Mat> new_deltas(group_samples, Mat::zeros(sample_size, CV_64FC1));

			net_nodes.push_back(new_nodes);
			net_deltas.push_back(new_deltas);

			layer_weight_idx.push_back(weight_idx);

		} else if((*layer_itr).type[0] == 'c') {

			time_ff.push_back(0.0);
			time_bp.push_back(0.0);

			weight_idx++;

			if(layer_idx < 1) {
				cout << "First layer has to be an 'i' type." << endl;
				cin.get();
				exit(EXIT_CODE_PARA_INPUT_LAYER);
			}

			//Fill conv layer with Mat::zeros
			//size_of_conv_layer =  maps
			int border = (*layer_itr).size / 2;
			int prev_layer_size = (int)net_nodes[layer_idx - 1].size();
			Size prev_node_size = net_nodes[layer_idx - 1][0].size();
			vector<Mat> new_nodes(group_samples * (*layer_itr).maps,
				Mat::zeros(prev_node_size.height - border * 2, prev_node_size.width - border * 2, CV_64FC1));
			vector<Mat> new_deltas(group_samples * (*layer_itr).maps,
				Mat::zeros(prev_node_size.height - border * 2, prev_node_size.width - border * 2, CV_64FC1));

			net_nodes.push_back(new_nodes);
			net_deltas.push_back(new_deltas);

			//Initialize weights (convolution kernels)
			//size_of_weight_layer = prev_layer_maps * maps
			vector<Mat> new_weights;
			vector<Mat> new_biases;
			int maps_tmp = prev_layer_size / group_samples * (1 + (*layer_itr).maps);
			int prev_maps = net_weights.size() == 0 ? 1 : (int)net_weights.back().size();

			for(int map = 0; map < (*layer_itr).maps; map++) {
				for(int idx = 0; idx < prev_maps; idx++) {
					Mat kernel((*layer_itr).size, (*layer_itr).size, CV_64FC1);
					rng.fill(kernel, RNG::UNIFORM, Scalar(0.0), Scalar(1.0));
					kernel = 2.0 * sqrt(6.0 / maps_tmp / (*layer_itr).size / (*layer_itr).size) *
						(kernel - 0.5 * Mat::ones(kernel.size(), CV_64FC1));
					new_weights.push_back(kernel);
				}
				new_biases.push_back(Mat::zeros(prev_node_size.height - border * 2, prev_node_size.width - border * 2, CV_64FC1));
			}

			net_weights.push_back(new_weights);
			net_biases.push_back(new_biases);

			layer_weight_idx.push_back(weight_idx);

		} else if((*layer_itr).type[0] == 's') {

			time_ff.push_back(0.0);
			time_bp.push_back(0.0);

			if(layer_idx < 1) {
				cout << "First layer has to be an 'i' type." << endl;
				cin.get();
				exit(EXIT_CODE_PARA_INPUT_LAYER);
			}

			//Fill subs layer with Mat::zeros
			int prev_layer_size = (int)net_nodes[layer_idx - 1].size();
			Size prev_node_size = net_nodes[layer_idx - 1][0].size();
			vector<Mat> new_nodes(prev_layer_size, Mat::zeros(prev_node_size.height / 2, prev_node_size.width / 2, CV_64FC1));
			vector<Mat> new_deltas(prev_layer_size, Mat::zeros(prev_node_size.height / 2, prev_node_size.width / 2, CV_64FC1));

			net_nodes.push_back(new_nodes);
			net_deltas.push_back(new_deltas);

			layer_weight_idx.push_back(weight_idx);

		} else if((*layer_itr).type[0] == 'o') {

			time_ff.push_back(0.0);
			time_bp.push_back(0.0);

			weight_idx++;

			if(layer_idx < 2) {
				cout << "Network must contain at least 1 hidden layer." << endl;
				cin.get();
				exit(EXIT_CODE_PARA_HIDDEN_LAYER);
			}

			//Fill output layer with Mat::zeros
			vector<Mat> new_nodes, new_deltas;
			int prev_layer_size = (int)net_nodes[layer_idx - 1].size();
			Size prev_node_size = net_nodes[layer_idx - 1][0].size();
			new_nodes.push_back(Mat::zeros(group_samples, (*layer_itr).classes, CV_64FC1));
			new_deltas.push_back(Mat::zeros(group_samples, (*layer_itr).classes, CV_64FC1));

			net_nodes.push_back(new_nodes);
			net_deltas.push_back(new_deltas);

			vector<Mat> new_weights;
			vector<Mat> new_biases;
			Mat kernel(prev_node_size.width*prev_node_size.height*prev_layer_size / group_samples, (*layer_itr).classes, CV_64FC1);
			rng.fill(kernel, RNG::UNIFORM, Scalar(0.0), Scalar(1.0));
			kernel = 2.0 * sqrt(6.0 / (kernel.cols + kernel.rows)) * (kernel - 0.5 * Mat::ones(kernel.size(), CV_64FC1));

			new_weights.push_back(kernel);
			new_biases.push_back(Mat::zeros(group_samples, (*layer_itr).classes, CV_64FC1));
			net_weights.push_back(new_weights);
			net_biases.push_back(new_biases);

			layer_weight_idx.push_back(weight_idx);
		}
		layer_idx++;
	}

	initial_done = true;

}


void CNN::trainCNN() {

	if(!initial_done) {
		cout << "CNN not initialized." << endl << "Press any key to exit..." << endl;
		cin.get();
		exit(EXIT_CODE_CNN_NOT_INIT);
	}

	for(int epoch_idx = 0; epoch_idx < epoch_num; epoch_idx++) {

#ifdef _HANY_NET_PRINT_MSG
		if(epoch_idx % 1 == 0) {
			cout << "Epoch: " << epoch_idx+1 << " / " << epoch_num << "...";
		}
#endif

		//Randomize train samples set
		random_shuffle(train_set.begin(), train_set.end());

#ifdef _HANY_NET_LIMIT_SAMPLE_NUM
		for(int group_idx = 0; group_idx < _HANY_NET_LIMIT_SAMPLE_NUM; group_idx++) {
#else
		for(int group_idx = 0; group_idx < train_set.size() / group_samples; group_idx++) {
#endif

			Mat net_output_acti, net_output;
			int layer_idx = 0;
			int in_maps = 1;
			net_labels.clear();

			//------------
			//Feed Forward
			//------------

			for(vector<layer>::iterator layer_itr = net_struct.begin(); layer_itr != net_struct.end(); layer_itr++, layer_idx++) {
				if((*layer_itr).type[0] == 'i') {

					double time_cost = time_cost = (double)getTickCount();

					int node_idx = 0;
					for(vector<Mat>::iterator node_itr = net_nodes[layer_idx].begin(); node_itr != net_nodes[layer_idx].end(); 
					node_itr++, node_idx++) {
						Mat new_sample;
						train_set[group_idx * group_samples + node_idx].first.convertTo(new_sample, CV_64FC1, 1.0 / 255.0);
						*node_itr = new_sample;
						net_labels.push_back(train_set[group_idx * group_samples + node_idx].second);
					}

					time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
					time_ff[layer_idx] += time_cost;

				} else if((*layer_itr).type[0] == 'c') {

					double time_cost = time_cost = (double)getTickCount();

					int border = (*layer_itr).size / 2;
					Size prev_node_size = (*net_nodes[layer_idx - 1].begin()).size();
					vector<Mat>::iterator node_itr = net_nodes[layer_idx].begin();

					for(int sample_idx = 0; sample_idx < group_samples; sample_idx++) {

						for(int out_map_idx = 0; out_map_idx < (*layer_itr).maps; out_map_idx++, node_itr++) {

							Mat result_roi = Mat::zeros(prev_node_size.height - border * 2, prev_node_size.width - border * 2, CV_64FC1);
							Rect roi(border, border, prev_node_size.width - border * 2, prev_node_size.height - border * 2);

							for(int in_map_idx = 0; in_map_idx < in_maps; in_map_idx++) {
								Mat result;
								filter2D(net_nodes[layer_idx - 1][in_maps * sample_idx + in_map_idx], result, CV_64FC1,
									net_weights[layer_weight_idx[layer_idx] - 1][in_maps * out_map_idx + in_map_idx]);
								result_roi += result(roi);
							}
							result_roi += net_biases[layer_weight_idx[layer_idx] - 1][out_map_idx];

							if(strcmp((*layer_itr).func.c_str(), "sigmoid") == 0) {
								*node_itr = sigmoid(result_roi);
							} else if(strcmp((*layer_itr).func.c_str(), "relu") == 0) {
								*node_itr = ReLU(result_roi);
							} else {
								cout << "Activation function for conv layer must be sigmoid or relu." << endl;
								exit(EXIT_CODE_PARA_ACTIV_FUNC);
							}
						}
					}

					in_maps = (*layer_itr).maps;

					time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
					time_ff[layer_idx] += time_cost;

				} else if((*layer_itr).type[0] == 's') {

					double time_cost = time_cost = (double)getTickCount();

					vector<Mat>::iterator node_itr = net_nodes[layer_idx].begin();
					for(vector<Mat>::iterator prev_node_itr = net_nodes[layer_idx - 1].begin();
					prev_node_itr != net_nodes[layer_idx - 1].end(); prev_node_itr++, node_itr++) {
						Mat new_node;
						resize(*prev_node_itr, new_node,
							Size((*prev_node_itr).cols / (*layer_itr).scale, (*prev_node_itr).rows / (*layer_itr).scale));
						*node_itr = new_node;
					}

					time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
					time_ff[layer_idx] += time_cost;

				} else if((*layer_itr).type[0] == 'o') {

					double time_cost = time_cost = (double)getTickCount();

					int prev_node_idx = 0;
					int sample_stride = (int)net_nodes[layer_idx - 1].size() / group_samples;
					vector<Mat>::iterator prev_node_itr = net_nodes[layer_idx - 1].begin();
					int width = (*prev_node_itr).cols;
					int height = (*prev_node_itr).rows;
					Mat layer_value = Mat::zeros(group_samples, width*height*sample_stride, CV_64FC1);

					for(int sample_idx = 0; sample_idx < group_samples; sample_idx++) {
						double* prow = layer_value.ptr<double>(sample_idx);
						for(int node_idx = 0; node_idx < sample_stride; node_idx++, prev_node_itr++) {
							for(int j = 0; j < height; j++) {
								double* pdata = (*prev_node_itr).ptr<double>(j);
								for(int i = 0; i < width; i++) {
									prow[node_idx * height * width + j * width + i] = pdata[i];
								}
							}
						}
					}

					layer_value.copyTo(net_output_acti);
					net_output = sigmoid(layer_value * net_weights[layer_weight_idx[layer_idx] - 1][0] +
						net_biases[layer_weight_idx[layer_idx] - 1][0]);
					net_nodes[layer_idx][0] = net_output;

					time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
					time_ff[layer_idx] += time_cost;

				}
			}

			//----------------
			//Back Propagation
			//----------------

			layer_idx--;

			for(vector<layer>::reverse_iterator layer_itr = net_struct.rbegin(); layer_itr != net_struct.rend();
			layer_itr++, layer_idx--) {
				if((*layer_itr).type[0] == 'i') {

					//Do nothing

				} else if((*layer_itr).type[0] == 'c') {

					double time_cost = time_cost = (double)getTickCount();

					//Calculate delta values
					vector<Mat>::iterator node_itr = net_nodes[layer_idx].begin();
					vector<Mat>::iterator desc_delta_itr = net_deltas[layer_idx + 1].begin();
					for(vector<Mat>::iterator delta_itr = net_deltas[layer_idx].begin(); delta_itr != net_deltas[layer_idx].end();
					delta_itr++, desc_delta_itr++, node_itr++) {
						Mat delta_resize;
						resize(*desc_delta_itr, delta_resize, Size((*desc_delta_itr).cols * 2, (*desc_delta_itr).rows * 2));
						Mat new_delta = delta_resize.mul(*node_itr - (*node_itr).mul(*node_itr));
						*delta_itr = new_delta;
					}

					//Update weight values
					int in_maps = (int)net_nodes[layer_idx - 1].size() / group_samples;
					int out_maps = (*layer_itr).maps;

					vector<Mat>::iterator weight_itr = net_weights[layer_weight_idx[layer_idx] - 1].begin();
					for(int in_map_idx = 0; in_map_idx < in_maps; in_map_idx++) {

						for(int out_map_idx = 0; out_map_idx < out_maps; out_map_idx++, weight_itr++) {

							Mat weight_gradient = Mat::zeros((*weight_itr).size(), CV_64FC1);
							Mat bias_gradient = Mat::zeros(net_deltas[layer_idx][0].size(), CV_64FC1);

							for(int sample_idx = 0; sample_idx < group_samples; sample_idx++) {
								//Sum up weight gradient
								Mat result;
								filter2D(net_nodes[layer_idx - 1][sample_idx * in_maps + in_map_idx], result, CV_64FC1,
									net_deltas[layer_idx][sample_idx * out_maps + out_map_idx], Point(0, 0));
								Rect roi(Point(0, 0), weight_gradient.size());
								weight_gradient += result(roi);

								//Sum up bias gradient
								if(in_map_idx == 0) {
									bias_gradient += net_deltas[layer_idx][sample_idx * out_maps + out_map_idx];
								}
							}

							(*weight_itr) -= gradient_alpha * weight_gradient / group_samples;
							net_biases[layer_weight_idx[layer_idx] - 1][out_map_idx] -= 
								gradient_alpha * bias_gradient / group_samples;
						}
					}

					time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
					time_bp[layer_idx] += time_cost;

				} else if((*layer_itr).type[0] == 's') {

					double time_cost = time_cost = (double)getTickCount();

					if(layer_idx >= net_nodes.size() - 2) {
						continue;
					} else {
						in_maps = net_struct[layer_idx - 1].maps;
						int out_maps = net_struct[layer_idx + 1].maps;
						int border = net_struct[layer_idx + 1].size / 2;
						Size new_delta_size = net_deltas[layer_idx][0].size();

						vector<Mat> desc_deltas_expand(net_deltas[layer_idx + 1].size(), Mat::zeros(new_delta_size, CV_64FC1));

						vector<Mat>::iterator delta_itr = net_deltas[layer_idx].begin();
						for(int sample_idx = 0; sample_idx < group_samples; sample_idx++) {

							for(int in_map_idx = 0; in_map_idx < in_maps; in_map_idx++, delta_itr++) {

								Mat delta_result = Mat::zeros(new_delta_size, CV_64FC1);

								for(int out_map_idx = 0; out_map_idx < out_maps; out_map_idx++) {

									//Initialize desc_deltas_expand[]
									if(in_map_idx == 0) {
										Mat tmp(new_delta_size, CV_64FC1);
										copyMakeBorder(net_deltas[layer_idx + 1][sample_idx * out_maps + out_map_idx], tmp,
											border, border, border, border, BORDER_CONSTANT, Scalar(0.0));
										flip(tmp, desc_deltas_expand[sample_idx * out_maps + out_map_idx], -1);
									}

									Mat result;
									filter2D(desc_deltas_expand[sample_idx * out_maps + out_map_idx], result, CV_64FC1,
										net_weights[layer_weight_idx[layer_idx]][out_map_idx * in_maps + in_map_idx]);
									delta_result += result;
								}

								*delta_itr = delta_result;

							}
						}
					}

					time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
					time_bp[layer_idx] += time_cost;

				} else if((*layer_itr).type[0] == 'o') {

					double time_cost = time_cost = (double)getTickCount();

					Mat err;			//original loss

					net_nodes[layer_idx][0].copyTo(err);
					for(int sample_idx = 0; sample_idx < group_samples; sample_idx++) {
						err.at<double>(sample_idx, (int)net_labels[sample_idx]) += -1.0;
					}

					Mat output_delta = err.mul(net_output - net_output.mul(net_output));

					general_loss = sum(err.mul(err))[0] / group_samples / 2.0;
					if(stage_loss.size() == 0) {
						stage_loss.push_back(general_loss);
					} else {
						stage_loss.push_back(0.99 * stage_loss[stage_loss.size() - 1] + 0.01 * general_loss);
					}

					Mat output_weight_t;
					transpose(net_weights[layer_weight_idx[layer_idx] - 1][0], output_weight_t);
					Mat gradient_value = output_delta * output_weight_t;

					//Calculate delta for previous layer
					int sample_stride = (int)net_nodes[layer_idx - 1].size() / group_samples;
					vector<Mat>::iterator prev_delta_itr = net_deltas[layer_idx - 1].begin();
					int width = (*prev_delta_itr).cols;
					int height = (*prev_delta_itr).rows;

					for(int sample_idx = 0; sample_idx < group_samples; sample_idx++) {
						double* prow = gradient_value.ptr<double>(sample_idx);
						for(int node_idx = 0; node_idx < sample_stride; node_idx++, prev_delta_itr++) {
							Mat tmp((*prev_delta_itr).size(), CV_64FC1);
							for(int j = 0; j < height; j++) {
								double* pdata = tmp.ptr<double>(j);
								for(int i = 0; i < width; i++) {
									pdata[i] = prow[node_idx * width * height + j * width + i];
								}
							}
							*prev_delta_itr = tmp;
						}
					}

					//Calculate weight gradients for output layer and update its weights
					Mat net_output_acti_t;
					transpose(net_output_acti, net_output_acti_t);
					Mat output_gradient = net_output_acti_t * output_delta / 50.0;
					net_weights[layer_weight_idx[layer_idx] - 1][0] -= gradient_alpha * output_gradient;

					//Calculate bias gradients for output layer and update its weights
					Mat bias_gradient_vec = Mat::zeros(1, output_delta.cols, CV_64FC1);
					for(int class_idx = 0; class_idx < bias_gradient_vec.cols; class_idx++) {
						double sum = 0.0;
						for(int sample_idx = 0; sample_idx < group_samples; sample_idx++) {
							sum += output_delta.ptr<double>(sample_idx)[class_idx];
						}
						bias_gradient_vec.ptr<double>(0)[class_idx] = (double)sum / group_samples;
					}
					Mat bias_gradient;
					repeat(bias_gradient_vec, group_samples, 1, bias_gradient);
					net_biases[layer_weight_idx[layer_idx] - 1][0] -= gradient_alpha * bias_gradient;

					time_cost = ((double)getTickCount() - time_cost) / getTickFrequency();
					time_bp[layer_idx] += time_cost;

				}	//layer.type if

			}	//layer_itr loop

		}	//group_idx loop

#ifdef _HANY_NET_PRINT_MSG
		if(epoch_idx % 1 == 0) {
			cout << "\t\t" << "with error of: " << stage_loss.back() << endl;
		}
#endif

	}	//epoch_idx loop

}

void CNN::downloadCNN(string file_name) {

	FileStorage file(file_name, FileStorage::WRITE);

	int in_maps = 1;

	for(int layer_idx = 0; layer_idx < net_struct.size(); layer_idx++) {
		if(net_struct[layer_idx].type[0] == 'i') {
			//Do nothing.
		} else if(net_struct[layer_idx].type[0] == 'c') {
			for(int in_idx = 0; in_idx < in_maps; in_idx++) {
				for(int out_idx = 0; out_idx < net_struct[layer_idx].maps; out_idx++) {
					string node_name = "weight_" + to_string(layer_idx) + "_" + to_string(in_idx) + "_" + to_string(out_idx);
					file << node_name << net_weights[layer_weight_idx[layer_idx] - 1][in_maps * out_idx + in_idx];
				}
			}
			for(int out_idx = 0; out_idx < net_struct[layer_idx].maps; out_idx++) {
				string node_name = "bias_" + to_string(layer_idx) + "_" + to_string(out_idx);
				file << node_name << net_biases[layer_weight_idx[layer_idx] - 1][out_idx];
			}
			in_maps = net_struct[layer_idx].maps;
		} else if(net_struct[layer_idx].type[0] == 's') {
			//Do nothing.
		} else if(net_struct[layer_idx].type[0] == 'o') {
			string node_name = "weight_" + to_string(layer_idx);
			file << node_name << net_weights[layer_weight_idx[layer_idx] - 1][0];
			node_name = "bias_" + to_string(layer_idx);
			file << node_name << net_biases[layer_weight_idx[layer_idx] - 1][0];
		}
	}

}


void CNN::uploadCNN(string file_name) {

	rapidxml::file<> fdoc(file_name.c_str());
	rapidxml::xml_document<> doc;
	doc.parse<0>(fdoc.data());

	rapidxml::xml_node<>* root = doc.first_node();
	if(string(root->name()) == "opencv_storage") {

		int in_maps = 1;
		rapidxml::xml_node<>* node = root->first_node();

		for(int layer_idx = 0; layer_idx < net_struct.size(); layer_idx++) {
			if(net_struct[layer_idx].type[0] == 'i') {
				//Do nothing.
			} else if(net_struct[layer_idx].type[0] == 'c') {
				for(int in_idx = 0; in_idx < in_maps; in_idx++) {
					for(int out_idx = 0; out_idx < net_struct[layer_idx].maps; out_idx++) {
						string node_name = "weight_" + to_string(layer_idx) + "_" + to_string(in_idx) + "_" + to_string(out_idx);
						
						if(string(node->name()) == node_name &&
							string(node->first_attribute("type_id")->value()) == "opencv-matrix" &&
							string(node->first_node("dt")->value()) == "d") {

							string mat_rows(node->first_node("rows")->value());
							string mat_cols(node->first_node("cols")->value());
							string mat_data(node->first_node("data")->value());

							net_weights[layer_weight_idx[layer_idx] - 1][in_maps * out_idx + in_idx] =
								parseMatFromXMLd(mat_rows, mat_cols, mat_data);
							node = node->next_sibling();
						} else {
							cout << "Matrix parameters are invalid." << endl;
							exit(EXIT_CODE_PARSE_MAT_INVALID);
						}
					}
				}
				for(int out_idx = 0; out_idx < net_struct[layer_idx].maps; out_idx++) {
					string node_name = "bias_" + to_string(layer_idx) + "_" + to_string(out_idx);
					if(string(node->name()) == node_name &&
						string(node->first_attribute("type_id")->value()) == "opencv-matrix" &&
						string(node->first_node("dt")->value()) == "d") {

						string mat_rows(node->first_node("rows")->value());
						string mat_cols(node->first_node("cols")->value());
						string mat_data(node->first_node("data")->value());

						net_biases[layer_weight_idx[layer_idx] - 1][out_idx] =
							parseMatFromXMLd(mat_rows, mat_cols, mat_data);
						node = node->next_sibling();
					} else {
						cout << "Matrix parameters are invalid." << endl;
						exit(EXIT_CODE_PARSE_MAT_INVALID);
					}
				}
				in_maps = net_struct[layer_idx].maps;
			} else if(net_struct[layer_idx].type[0] == 's') {
				//Do nothing.
			} else if(net_struct[layer_idx].type[0] == 'o') {
				string node_name = "weight_" + to_string(layer_idx);
				if(string(node->name()) == node_name &&
					string(node->first_attribute("type_id")->value()) == "opencv-matrix" &&
					string(node->first_node("dt")->value()) == "d") {

					string mat_rows(node->first_node("rows")->value());
					string mat_cols(node->first_node("cols")->value());
					string mat_data(node->first_node("data")->value());

					net_weights[layer_weight_idx[layer_idx] - 1][0] =
						parseMatFromXMLd(mat_rows, mat_cols, mat_data);
					node = node->next_sibling();
				} else {
					cout << "Matrix parameters are invalid." << endl;
					exit(EXIT_CODE_PARSE_MAT_INVALID);
				}

				node_name = "bias_" + to_string(layer_idx);
				if(string(node->name()) == node_name &&
					string(node->first_attribute("type_id")->value()) == "opencv-matrix" &&
					string(node->first_node("dt")->value()) == "d") {

					string mat_rows(node->first_node("rows")->value());
					string mat_cols(node->first_node("cols")->value());
					string mat_data(node->first_node("data")->value());

					net_biases[layer_weight_idx[layer_idx] - 1][0] =
						parseMatFromXMLd(mat_rows, mat_cols, mat_data);
					node = node->next_sibling();
				} else {
					cout << "Matrix parameters are invalid." << endl;
					exit(EXIT_CODE_PARSE_MAT_INVALID);
				}
			}	//layer.type if
		}	//layer_itr loop
	}	//root->name() if

}


vector<Mat> CNN::predictCNN(vector<Mat>& input) {

	if(!initial_done) {
		cout << "CNN not initialized." << endl << "Press any key to exit..." << endl;
		cin.get();
		exit(EXIT_CODE_CNN_NOT_INIT);
	}

	int dupli_num, group_num;
	if(input.size() > group_samples) {
		dupli_num = 1;
		group_num = (int)input.size() / group_samples + 1;
	} else {
		dupli_num = group_samples / (int)input.size();
		group_num = 1;
	}

	vector<Mat> output;

	for(int group_idx = 0; group_idx < group_num; group_idx++) {

		Mat net_output;
		int layer_idx = 0;
		int in_maps = 1;
		net_labels.clear();

		for(vector<layer>::iterator layer_itr = net_struct.begin(); layer_itr != net_struct.end(); layer_itr++, layer_idx++) {
			if((*layer_itr).type[0] == 'i') {

				for(int node_idx = 0; node_idx < net_nodes[layer_idx].size(); node_idx++) {
					if(group_idx == group_num - 1) {
						if(node_idx < (input.size() - (group_num - 1) * group_samples) * dupli_num) {
							for(int dupli_idx = 0; dupli_idx < dupli_num; dupli_idx++, node_idx++) {
								Mat new_sample, nor_sample;
								input[group_idx * group_samples + node_idx - dupli_idx].convertTo(new_sample, CV_64FC1, 1.0 / 255.0);
								resize(new_sample, nor_sample, Size(sample_width, sample_height));
								net_nodes[layer_idx][node_idx] = nor_sample;
							}
						} else {
							for(int blank_idx = 0; blank_idx < group_samples - input.size() * dupli_num; blank_idx++, node_idx++) {
								Mat new_sample = Mat::zeros(Size(sample_width, sample_height), CV_64FC1);
								net_nodes[layer_idx][node_idx] = new_sample;
							}
						}
					} else {
						Mat new_sample, nor_sample;
						input[group_idx * group_samples + node_idx].convertTo(new_sample, CV_64FC1, 1.0 / 255.0);
						resize(new_sample, nor_sample, Size(sample_width, sample_height));
						net_nodes[layer_idx][node_idx] = nor_sample;
					}
				}

			} else if((*layer_itr).type[0] == 'c') {

				int border = (*layer_itr).size / 2;
				Size prev_node_size = (*net_nodes[layer_idx - 1].begin()).size();
				vector<Mat>::iterator node_itr = net_nodes[layer_idx].begin();

				for(int sample_idx = 0; sample_idx < group_samples; sample_idx++) {

					for(int out_map_idx = 0; out_map_idx < (*layer_itr).maps; out_map_idx++, node_itr++) {

						Mat result_roi = Mat::zeros(prev_node_size.height - border * 2, prev_node_size.width - border * 2, CV_64FC1);
						Rect roi(border, border, prev_node_size.width - border * 2, prev_node_size.height - border * 2);

						for(int in_map_idx = 0; in_map_idx < in_maps; in_map_idx++) {
							Mat result;
							filter2D(net_nodes[layer_idx - 1][in_maps * sample_idx + in_map_idx], result, CV_64FC1,
								net_weights[layer_weight_idx[layer_idx] - 1][in_maps * out_map_idx + in_map_idx]);
							result_roi += result(roi);
						}
						result_roi += net_biases[layer_weight_idx[layer_idx] - 1][out_map_idx];

						if(strcmp((*layer_itr).func.c_str(), "sigmoid") == 0) {
							*node_itr = sigmoid(result_roi);
						} else if(strcmp((*layer_itr).func.c_str(), "relu") == 0) {
							*node_itr = ReLU(result_roi);
						} else {
							cout << "Activation function for conv layer must be sigmoid or relu." << endl;
							exit(EXIT_CODE_PARA_ACTIV_FUNC);
						}
					}
				}

				in_maps = (*layer_itr).maps;

			} else if((*layer_itr).type[0] == 's') {

				vector<Mat>::iterator node_itr = net_nodes[layer_idx].begin();
				for(vector<Mat>::iterator prev_node_itr = net_nodes[layer_idx - 1].begin();
				prev_node_itr != net_nodes[layer_idx - 1].end(); prev_node_itr++, node_itr++) {
					Mat new_node;
					resize(*prev_node_itr, new_node,
						Size((*prev_node_itr).cols / (*layer_itr).scale, (*prev_node_itr).rows / (*layer_itr).scale));
					*node_itr = new_node;
				}

			} else if((*layer_itr).type[0] == 'o') {

				int prev_node_idx = 0;
				int sample_stride = (int)net_nodes[layer_idx - 1].size() / group_samples;
				vector<Mat>::iterator prev_node_itr = net_nodes[layer_idx - 1].begin();
				int width = (*prev_node_itr).cols;
				int height = (*prev_node_itr).rows;
				Mat layer_value = Mat::zeros(group_samples, width*height*sample_stride, CV_64FC1);

				for(int sample_idx = 0; sample_idx < group_samples; sample_idx++) {
					double* prow = layer_value.ptr<double>(sample_idx);
					for(int node_idx = 0; node_idx < sample_stride; node_idx++, prev_node_itr++) {
						for(int j = 0; j < height; j++) {
							double* pdata = (*prev_node_itr).ptr<double>(j);
							for(int i = 0; i < width; i++) {
								prow[node_idx * height * width + j * width + i] = pdata[i];
							}
						}
					}
				}

				net_output = sigmoid(layer_value * net_weights[layer_weight_idx[layer_idx] - 1][0] +
					net_biases[layer_weight_idx[layer_idx] - 1][0]);

				//Predict class based on 'net_output'
				Mat new_output(1, group_samples, CV_8UC1);
				if(group_idx == group_num - 1 && dupli_num != 1) {
					//Sum up all output of duplicated samples, and set index of maximum sum as the predict class
					//---- potential multi-threading ----
					for(int sample_idx = 0; sample_idx < input.size() - (group_num - 1) * group_samples; sample_idx++) {
						Mat prob_sum = Mat::zeros(1, net_output.cols, CV_64FC1);
						double max_predict = 0.0;
						Mat vote_sum = Mat::zeros(1, net_output.cols, CV_8UC1);
						int max_vote = 0;
						int predict_class = 0;

						for(int class_idx = 0; class_idx < net_output.cols; class_idx++) {
							for(int dupli_idx = 0; dupli_idx < dupli_num; dupli_idx++) {
								prob_sum.ptr<double>(0)[class_idx] += net_output.ptr<double>(sample_idx * dupli_num + dupli_idx)[class_idx];
								vote_sum.ptr<uchar>(0)[class_idx] += 1;
							}
							if(vote_sum.ptr<uchar>(0)[class_idx] > max_vote) {
								max_vote = vote_sum.ptr<uchar>(0)[class_idx];
								predict_class = class_idx;
							}
						}

						//If there are more than one predict classes having the same maximum votes,
						//take the one with maximum sum-up predict probability as the final predict class.
						vector<uchar> max_votes;
						max_votes.push_back(predict_class);
						for(int class_idx = 0; class_idx < net_output.cols; class_idx++) {
							if(vote_sum.ptr<uchar>(0)[class_idx] == max_vote) {
								max_votes.push_back(class_idx);
							}
							if(prob_sum.ptr<double>(0)[class_idx] > max_predict) {
								max_predict = prob_sum.ptr<double>(0)[class_idx];
								predict_class = class_idx;
							}
						}
						if(max_votes.size() == 1) {
							predict_class = max_votes[0];
						}
//						if(max_predict / (double)vote_sum.ptr<uchar>(0)[predict_class] > 0.7)
							new_output.ptr<uchar>(0)[sample_idx] = (uchar)predict_class;
//						else
//							new_output.ptr<uchar>(0)[sample_idx] = (uchar)net_output.cols + 1;
					}
				} else {
					//Set index of maximum value in each sample as its predict class
					//may be replaced with cv::reduce()
					//---- potential multi-threading ----
					for(int sample_idx = 0; sample_idx < group_samples; sample_idx++) {
						double* prow = net_output.ptr<double>(sample_idx);
						double max_predict = 0.0;
						int predict_class = 0;
						for(int class_idx = 0; class_idx < net_output.cols; class_idx++) {
							if(prow[class_idx] > max_predict) {
								max_predict = prow[class_idx];
								predict_class = class_idx;
							}
						}
						new_output.ptr<uchar>(0)[sample_idx] = (uchar)predict_class;
					}
				}
				output.push_back(new_output);
			}	//layer.type if
		}	//layer_itr loop
	}	//group_idx loop

	return output;

}

