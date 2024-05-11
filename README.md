## Projects
## [#1 - Flight Price Prediction](https://github.com/aiswaryanair/Linear-Regression---Flight-Price-Prediction)
### Objective
The objective of the project is to do exploratory data analysis of the flight booking dataset obtained from “Ease My Trip” website and to predict price of airline tickets to make better pricing decisions and optimize revenue using the best trained model.
### Dataset
Dataset contains information about flight booking options from the website Easemytrip for flight travel between 6 cities. There are 3,00,261 datapoints and 11 features in the cleaned dataset.
### Algorithms
- KNeighborsRegressor
- Linear Regression
- XGBRegressor
- CatBoostRegressor

### Required Libraries
```
import seaborn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
```
### Plots
- barplot
- violinplot
- regplot
- relplot
- scatterplot
- heatmap

## [#2 - Deep Learning:Image Classification using pre-trained models in Keras](https://github.com/aiswaryanair/Image-Classifier---ResNet50-and-VGG16---Evaluation-and-Testing-Performance)
### Objective
The objective is to build Image Classifiers using Resnet50 and VGG16 pre-trained models and compare their performances.
### Dataset
The dataset utilized for this project was sourced from Concrete Crack Images for Classification. It contains images relevant to concrete crack classification.
### Models
ResNet50 - Residual Network (ResNet) is a deep learning model used for computer vision applications. It is a Convolutional Neural Network (CNN) architecture designed to support hundreds or thousands of convolutional layers.

VGG16 - It is a convolution neural network (CNN) model supporting 16 layers.
### Required Libraries
```
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
```
## [#3 - Retail Price Optimization](https://github.com/aiswaryanair/Retail-Price-Optimization)
### Objective
The objective is to estimate the price elasticity of each item in a cafe and find the optimal price for maximum profit.
### Tasks
- Exploratory data analysis
- Data visualization
- Demand forecasting
- Price optimization
  
### Algorithm
Ordinary Least Square Regression
### Required Libraries
```
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
```
## [#4 - Decision Tree Classifier for Drug Classification](https://github.com/aiswaryanair/Decision-Tree---Drug-Classification)
### Objective
Build a model to find out which drug might be appropriate for a future patient with the same illness. The features of this dataset are Age, Sex, Blood Pressure, and the Cholesterol of the patients, and the target is the drug that each patient responded to.
### Datasource 
IBM

### Required Libraries
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
```



