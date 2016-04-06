# HanyNet

A naive implementation of Deep Convolutional Neural Networks.

Based on LeNet which is composed of several convolutional layers and subsampling layers followed by a fully connnected layer.

LeNet is a classic model which was proposed back in 1989.

A new version that supports modern Deep Learning models is currently underway.

一个卷积神经网络的初步实现，基于经典的LeNet模型。

中文文档请见[johnhany.net](http://johnhany.net/2016/04/naive-implementation-of-cnn/)。

更完备的版本正在开发中。

## Requirements

### Visual Studio
The solution is created with Visual Studio 2012. You are more than welcome to use other IDEs. All you need to do is to copy all the `cpp`, `h`, and `xml` files and link OpenCV correctly.

### [OpenCV](http://opencv.org/downloads.html)
I'm using `OpenCV 3.1`, while other version of OpenCV may work as well, since we only use its basic I/O and Mat functionalities.

### [RapidXML](http://rapidxml.sourceforge.net/)
A header-only open-source library for XML document I/O. I'm using `RapidXML 1.13`

## Notes
If you want to run full function that contains both training and reference, make sure `_HANY_NET_TRAIN_FROM_SCRATCH` is defined at line 36 in `HanyNet.h`.

If you have a pretrained xml file for HanyNet and configured the `parameters.xml` file right, and only want to perform reference with the networks, simply comment out `_HANY_NET_TRAIN_FROM_SCRATCH` at line 36 in `HanyNet.h`.

Note that in file `HanyNet-run.cpp`, only one of the following pregmas at line 17~19 should be defined:
* _HANY_NET_LOAD_MNIST
* _HANY_NET_LOAD_SAMPLE_FROM_PIC
* _HANY_NET_CAPTURE_FACE_FROM_CAMERA

Only one of the following pregmas at line 21~24 should be defined:
* _HANY_NET_PREDICT_MNIST
* _HANY_NET_PREDICT_IMAGE_SERIES
* _HANY_NET_PREDICT_VEDIO_SERIES
* _HANY_NET_PREDICT_CAMERA

## Applications
### Digit Recognition
By default, HanyNet will load MNIST dataset and perform classifier training and reference with it.

You can download MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/).

Extract and copy these four files to the source code path (where `HanyNet-run.cpp` is located): `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`, `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`.

If you don't want to load MNIST dataset and use custom datasets instead, simply apply these changes to `HanyNet-run.cpp`:
* Comment out `#define _HANY_NET_LOAD_MNIST` at line 17.
* Comment out `#define _HANY_NET_PREDICT_MNIST` at line 21.
* Set `string sample_file_pre` at line 35 as path to the folder containing your train set and test set.
* Change `int sample_num` at line 42 to the number of samples for each class. We assume that each class contains equal number of training samples.
* Note that filename of samples should follow format as `a_b.jpg`, in which `a` stands for class index, and `b` stands for sample index of each class. `a` and `b` both begin with 0.

### Face Recognition
HantNet is capable of recognizing different persons' faces. Samples can be either images or captured by live camera.

If you want to train a face recognizor with pictures, follow these steps and make changes to `HanyNet-run.cpp`:
* Comment out `#define _HANY_NET_LOAD_MNIST` at line 17.
* Uncomment `#define _HANY_NET_LOAD_SAMPLE_FROM_PIC` at line 18.
* Set `string sample_file_pre` at line 35 as path to the folder containing your train set and test set.
* Change `int sample_num` at line 42 to the number of samples for each class. We assume that each class contains equal number of training samples.
* Note that filename of samples should follow format as `a_b.jpg`, in which `a` stands for class index, and `b` stands for sample index of each class. `a` and `b` both begin with 0.

Samples can also be captured by a live camera. HanyNet will automatically detect face with Haar Cascade Classifier and capture `sample_num` pictures for each person. During capturing stage, when having finished capturing one person, the program will stop capturing and wait for you to press `SPACE` to continue to capture the next person. When you are done capturing all persons, press `ESC` to move to the training stage.

Haar Cascade Classifier is downloaded from [here](https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml).

If you want to train a facial recognizor with camera, follow these steps and make changes to `HanyNet-run.cpp`:
* Comment out `#define _HANY_NET_LOAD_MNIST` at line 17.
* Uncomment `#define _HANY_NET_CAPTURE_FACE_FROM_CAMERA` at line 19.
* Change `int sample_num` at line 42 to the number of samples for each class. We assume that each class contains equal number of training samples.
* If you have prepared a TXT label file for each person, uncomment `_HANY_NET_WITH_LABEL_NAMES` at line 26 and set `label_file` at line 23 as path to the label file.

When you are done changing the `HanyNet-run.cpp` file, follow these steps:
* Build and run.
* Pass alone the camera along you and your friends. Make sure only one person appear in the screen at one time.
* Wait for HanyNet to train the networks.
* When training is completed, the program is ready for real-time facial recognition through your camera.

Reference can work with images, videos and live camera. Choose your favorate one with similar method on train set.

Note that there are differences between three types of samples:
* Images contain only faces, without any surroundings around the faces;
* Videos can contain a bigger background, since we use Haar Cascade Classifier to detect faces beforehand.
* Live camera can contain a bigger background as well.

Currently, according to multiple experiments, HanyNet is capable of recognizing four different persons with accuracy of 100%. We can't assure the same performance with cases of more persons.

### Other Classification Applications
HanyNet can be also trainied for other classification problems such as vehicle classification. Yet samples can only be images.

Follow the same steps for Face Recognition with image samples.

## Parameters
Parameters for constructing networks are contained in file `parameters.xml`. Meaning of each parameter entry is as such:
* `group_samples` - Number of samples of input. Recommand `10` for facial recognition, and larger value for MNIST.
* `epoch_num` - Number of iteration epoches during training. Recommand more than `300` for facial recognition, and smaller value for MNIST.
* `gradient_alpha` - Convergence speed of training. Recommand `1.0`.
* `sample_width` - Image width of input sample. Recommand `28` for a 6-layer network, and `60` for a 8-layer network.
* `sample_height` - Image height of input sample. Recommand `28` for a 6-layer network, and `60` for a 8-layer network.
* `net_struct` - Define network structure.
*  `layer type="i"` - Define an input layer.
*  `layer type="c"` - Define a convolutional layer.
*   `func` - Activate function of convolutional layer. Can be `relu` or `sigmoid`.
*   `maps` - Number of convolution kernels of convolutional layer.
*   `size` - Size of convolution kernel of convolutional layer.	Value should be odd.
*  `layer type="s"` - Define a subsampling layer.
*   `scale` - Scale of subsampling. `2` means half the size. Recommand `2`.
*  `layer type="o"` - Define an output layer.
*   `func` - Activate function of output layer. Only supports `softmax` for now.
*   `classes` - Number of classes.

Structure of the networks is designed as such:

    i | c | s | c | s | o

or

    i | c | s | ... | c | s | o

The first layer must be input layer.

The last layer must be output layer.

Hidden layers are composed of pairs of one convolutional layer and one subsampling layer, specifically one subsampling layer after one convolutional layer.

## Results
For MNIST dataset, with 6-layer networks, HanyNet can reach accuracy of 83~88%.

For facial recognition, with 4-person scenario, HanyNet can accomplish accuracy of 100%. Results for cases with more than 10 persons are less satisfactory.

For vehicle classification, with 4 classes containing only 20 samples each, HanyNet classification accuracy is listed below:

    Sedan | SUV | Truck | Bus
     67%  | 87% |  73%  | 93%

## License
HanyNet is released under the MIT license. Considering two of the dependencies are both open-source, you are granted with the same rights as they offer.
