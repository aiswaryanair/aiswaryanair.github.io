## [Project1 - Flight Price Prediction](https://github.com/aiswaryanair/Linear-Regression---Flight-Price-Prediction)
### Objective
The objective of the project is to analyse the flight booking dataset obtained from “Ease My Trip” website and to conduct various statistical hypothesis tests in order to get meaningful information from it. 'Easemytrip' is an internet platform for booking flight tickets, and hence a platform that potential passengers use to buy tickets. A thorough study of the data will aid in the discovery of valuable insights that will be of enormous value to passengers
### Dataset
Dataset contains information about flight booking options from the website Easemytrip for flight travel between India's top 6 metro cities. There are 300261 datapoints and 11 features in the cleaned dataset.
### Algorithms
- KNeighborsRegressor
- Linear Regression
- XGBRegressor
- CatBoostRegressor

## [Project2 - Deep Learning:Image Classification using pre-trained models in Keras](https://github.com/aiswaryanair/Image-Classifier---ResNet50-and-VGG16---Evaluation-and-Testing-Performance)
### Objective
The objective is to build Image Classifiers using Resnet50 and VGG16 pre-trained models and compare the performances
### Dataset
The dataset utilized for this project was sourced from Concrete Crack Images for Classification. It presumably contains images relevant to concrete crack classification.
### Models
ResNet50 - Residual Network (ResNet) is a deep learning model used for computer vision applications. It is a Convolutional Neural Network (CNN) architecture designed to support hundreds or thousands of convolutional layers.
VGG16 - It is a convolution neural network (CNN) model supporting 16 layers.
### Required Libraries
`python'
'import keras'
'from keras.models import Sequential
from keras.layers import Dense
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator'


