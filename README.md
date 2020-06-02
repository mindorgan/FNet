# FNet

FNet is a neural network designed for intensity based separation of overlapped transparent objects from time sequence images. It was developed to analyze Amyliod Beta fibril growth, but can be used to seperate any arbitrary objects.

## Dependencies
* Python 3.6+
* TensorFlow 1.14

## Architecture
![FNet](FNet.png)

The new neural network consists of four sub-networks: 1) Classification network, 2) Growth prediction network, 3) Background prediction network, and 4) Comparison network. The overall information flow is as follows: first the classification network encodes features from a new input image and decodes them into feature maps of different resolution. The growth prediction network encodes features of individual known molecules from the previous image, and then decodes them together with the feature maps of the same resolution from the classification network which contains the information of the new image. The background prediction network has the same structure with the prediction network but takes the previous background image as an input and use the weights different from those of the growth prediction network. The classification, the growth prediction and the background prediction networks generate single feature image outputs which stand for their prediction power of how many photons of each pixel result from new molecules, known molecules, and background, respectively. Then, the comparison network, which has 3 convolution layers and 3 activation layers, compares relative prediction powers from the prediction networks and generates photon count images of known fibrils, background and newly-appearing fibrils.

### Classification network and prediction network
The base structure of the classification network and the predictions networks is U-Net (44) like deep encoder-decoder networks. The classification network takes a photon count image as an input (160 × 160 × 1, 10 pixels were padded to the original image of 150 × 150 × 1 pixels in our analysis) and the growth prediction network takes photon count images of individual fibrils from the previous frame as inputs (126 images in our analysis). The encoding blocks (convolution, batch-normalization, rectified linear unit (ReLU) activation, and max pooling) of the classification network encodes 4 features (160 × 160 × 4), and the following encoder blocks reduce the image width and height by half and add 4 additional features. The decoder blocks are similar to encoder blocks, but double the image width and height and reduce the number of features by 4. This number of features can be adjusted by `base_feature` parameter. The decoder block also have skip connections with encoder blocks. The prediction networks have the same structure, but decoder blocks are connected with the decoder blocks of classification network. The background prediction network as the same structure as the growth prediction network, but with independent weights.

### Comparison Network
Each input from the classification network and the prediction networks generate single channel image output. The comparison network takes these images as an input and generates the final result using twice of a convolution, a batch-normalization, and an ReLU activation, followed by a convolution and a PReLU activation. The result is photon count prediction of newly-appearing fibrils, background, and known fibrils of new image frame. Then, each pixel of the result images was normalized to make the total number of photons of the result images be equal to the original input image for each pixel.

## Making a FNet model instance
```Python
from FNet import FNet
base_feature = 4 # depending on the complexity of objects
max_objects = 126 # maximum number of objects in an image
img_width = 160
img_height = 160
input_shape = (img_width, img_height, max_objects + 2)

model=FNet(base_feature, input_shape)
```
You can use this model instance for training and prediction using TensorFlow.