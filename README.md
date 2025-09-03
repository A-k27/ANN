<<<<<<< HEAD
PROJECT TITLE:

Developing and Evaluating Neural Network Models for Real-World Data.

DESCRIPTION: 

This project aims to develop and evaluate two neural network models for classifying the iris dataset into 3 species. The goal is to compare these models based on their accuracy and performance for classification. The models that were chosen were Feedforward Neural Network (FNN) and Self-Organising Map (SOM). Both used different techniques. The FNN uses a supervised learning approach, and SOM uses an unsupervised learning approach, giving an insight on which approach is best used to complete this task. 
The code in the final called FINALCODE_FOR_ANN_MODELS classifies the data in the iris dataset into the 3 species (classes): 
	- Setosa
	- Versicolor 
	- Virginica 
using two Artificial Neural Networks  
	- Model 1: Feedforward Neural Network, trained using Backpropagation. 
	- Model 2: Self-organising maps, using Kohonen. 
Some of the key features within this code are that data has been pre-processed and split into X and Y. Where X is the factors and Y is the target. Data has then been converted into TensorFlow and split into training and testing sets ready to be used by the models to classify the flower samples into one of the three species. 

INSTALLATION: 

To run the code, import the Iris dataset from the link 'https://archive.ics.uci.edu/dataset/53/iris' and save it as 'iris_data.csv'. 
INSTALL: pip install tensorflow
INSTALL: pip install minisom

USAGE:

There are 9 blocks of code 

*BLOCK 1

The first block of code is Pre-processing of the dataset. It reads the dataset and interprets it by outputting the column names and dataset information. It also produces a summary of the dataset. The code checks for null values and duplicates in the dataset and, if present, removes them. 
Additionally, it visualises the data through Bar charts, histograms, scatter plots and a correlation heatmap. 
It converts non-numerical values into numerical values, Re-groups columns and drops additional columns. It splits the target from the features. 


*BLOCK 2

The second block of code converts the dataset into tensors so that TensorFlow’s libraries can work on it. 
The dataset is split into training and testing sets for the model. A Min-Max scaler is used to normalise the data for more accurate results. The shape of these tensors is then outputted. 

*BLOCK 3

This block of the code is where the first Neural Network is created. The First model is a  Feedforward Neural Network (FNN), and training uses backpropagation. 

The libraries that are needed are imported. 
Input size is defined. 
The model is created. It is a sequential model that has one input layer, 2 hidden layers, 2 dropout layers and an output layer.  
The model is then complied and run - it iterates the dataset over 500 times, outputting the 'accuracy', 'loss', accuracy value and loss value over each iteration. 
A summary is then created and outputted. 
The overall accuracy and loss value of the model are produced, alongside the predicted class values and actual class values. 
 

*BLOCK 4

This block is the code for evaluating the FNN model. 
For the evaluation of the model, a confusion matrix was created, and the calculation of the precision, Recall and F1-score was created. In Addition to these metrics for each class being produced, so was the overall average. This is to understand the model's performance completely. 

*BLOCK 5

This code is the pre-processing for the second Neural Network, self-organising map (SOM).
As the dataset is the same, the same process for the FNN model was used for the SOM. 

*BLOCK 6

This block of code converts the dataset for the SOM into a tensor for the uses of TensorFlow and splits the dataset into training and testing sets. 
This Process is again the same as the conversion of the dataset for the FNN, as the dataset is the same. 

*BLOCK 7

This is where the minisom library is installed to begin the creating of the SOM. 

*BLOCK 8

This block is the creation of the SOM model. 

Libraries needed to create the model are imported.
The grid size, which is the model size, is defined.  This is a 10 by 10 grid. 
The input length is defined. 

The model is initialised. This states the model size and the input size. The radius is sigma and is set to 1.0, and the learning rate is set to 0.01.
Weights have been randomly initialised. 
The creation of a decaying function is added. This slowly reduces the learning rate and radius over time. The weights are updated by getting the best-matching unit and updating it based on which is most similar to the input. 
The SOM grid is then visualised.

*BLOCK 9

This block codes the evaluation of the SOM model. It creates a confusion matrix and calculates the Precision, Recall and F1-score for each class. Also calculates the average of these metrics to gain a deeper insight into the model's performance.  

TECHNOLOGIES USED:  

This code is coded in python 
The libraries used in this code are: 
	- pandas
	- numpy
	- matplotlib.pyplot
	- seaborn
	- tensorflow_datasets
	- tensorflow.keras
	- minisom

=======
Final code for assessment can be found in Ipyby file called: FINALCODE_FOR_ANN_MODELS 
How to use the code is in the readme file called: README_FILE
>>>>>>> c37a37791dd5dc7142e201acfa081531019e9e61
