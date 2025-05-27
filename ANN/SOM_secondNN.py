Python 3.13.2 (tags/v3.13.2:4f8bb39, Feb  4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
# PROBLEM: CLASSIFY SMAPLES INTO ONE OF 3 SPECIES: SETOSA, VERSICOLOUR OR VIRGINICA
# USING TWO DIFFERENT NUMRAL NETWORKS 
# PART A
# data preprocessing - this code is adated form previous semester module code machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
# DATA PRE-PORCESSING
# Load and preprocess datase
df = pd.read_csv('iris_data.csv')
# describing data set
print(df.head())
print(df.info())
print(df.describe())
# check shape of datset
print(df.shape)
# check for missing values
df.isnull().any()
# check for duplicate data
df.duplicated().sum()
# drop duplicate data
df.drop_duplicates(inplace=True)
df.duplicated().sum()
#  DATA VSIUALISATION
# class distribution count
df['class'].value_counts()
plt.figure(figsize=(6,4))
df['class'].value_counts().plot(kind='bar', color=['purple', 'pink', 'red'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()
#featuer distribution and histogram
sns.pairplot(df, hue='class', diag_kind='hist')
plt.show()
#pair plot
#shows visuals between features and clss separation
df.hist(figsize=(10, 8), bins=20)
plt.suptitle("Feature Distributions")
plt.show()
plt.figure(figsize=(12, 6))
# converting non-numberical data into numerical
# converting  non-numerical data into numerical data - material taken from Lab 3
df = pd.get_dummies(df, columns=['class'], drop_first=False)
df.head()
# check to see if there is any null values in new columns
df.isnull().any()
# heatmap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Compute correlation matrix
corr = df.corr()
# Set up the matplotlib figure
plt.figure(figsize=(10, 12))
# Plot the heatmap with a color bar for reference
sns.heatmap(corr,vmax=.8,linewidth=.01, square = True, annot = True,cmap='YlGnBu',linecolor ='black')
plt.title("Feature Correlation Heatmap")
plt.show()
# DATA PREPROCESSING BEFORE SEPRATION AND SPLITTING
# Use idxmax to regroup one-hot encoded columns into class labels
df['class'] = df[['class_Iris-setosa', 'class_Iris-versicolor', 'class_Iris-virginica']].idxmax(axis=1)
# removes and white spaces and charactors
df['class'] = df['class'].str.replace('class_', '')
#Catergriacl values to numerical values using mapping
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['class'] = df['class'].map(class_mapping)  # Convert class labels to integers
df = pd.get_dummies(df, columns=['class'], drop_first=False)
# Remove the prefix 'class_' from the species names
# Drop the one-hot encoded columns
df.drop(['class_Iris-setosa', 'class_Iris-versicolor', 'class_Iris-virginica'], axis=1, inplace=True)
df['class'] = df[['class_0','class_1','class_2']].idxmax(axis=1)
df.drop(['class_0','class_1','class_2'], axis=1, inplace=True)
# removes and white spaces and charactors
df['class'] = df['class'].str.replace('class_', '').astype(int)
# separate targets and valuesb
x = df.drop(['class'],axis=1).values  # factors
y = df['class'].values  # Target
print(x)
print(y)
#labs and https://www.tensorflow.org/api_docs/python/tf/keras/utils/split_dataset, and ChatGPT to fix imablnacing issues when spltting data into traing and testing sets
import tensorflow as tf
import numpy as np
# Check unique class labels in y
print("Unique class labels:", np.unique(y))
# Step 1: Group the indices by class
import tensorflow as tf
# Convert NumPy arrays to TensorFlow tensors
# Convert NumPy arrays to TensorFlow tensors
X_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
class_0_indices = tf.where(y_tensor == 0)[:, 0]
class_1_indices = tf.where(y_tensor == 1)[:, 0]
class_2_indices = tf.where(y_tensor == 2)[:, 0]
# Step 2: Shuffle the indices for each class
class_0_indices = tf.random.shuffle(class_0_indices)
class_1_indices = tf.random.shuffle(class_1_indices)
class_2_indices = tf.random.shuffle(class_2_indices)
# Step 3: Define split index for 80-20 training and testing
split_index_0 = int(0.8 * len(class_0_indices))
split_index_1 = int(0.8 * len(class_1_indices))
split_index_2 = int(0.8 * len(class_2_indices))
# Step 4: Split the indices into training and testing sets
class_0_train_indices = class_0_indices[:split_index_0]
class_0_test_indices = class_0_indices[split_index_0:]
class_1_train_indices = class_1_indices[:split_index_1]
class_1_test_indices = class_1_indices[split_index_1:]
class_2_train_indices = class_2_indices[:split_index_2]
class_2_test_indices = class_2_indices[split_index_2:]
# Step 5: Concatenate the indices for training and test
train_indices = tf.concat([class_0_train_indices, class_1_train_indices, class_2_train_indices], axis=0)
test_indices = tf.concat([class_0_test_indices, class_1_test_indices, class_2_test_indices], axis=0)
# Step 6: Shuffle the final train and test sets
train_indices = tf.random.shuffle(train_indices)
test_indices = tf.random.shuffle(test_indices)
# Step 7: Gather the data using the indices
X_train = tf.gather(X_tensor, train_indices)
y_train = tf.gather(y_tensor, train_indices)
X_test = tf.gather(X_tensor, test_indices)
y_test = tf.gather(y_tensor, test_indices)
# Min-Max Scaling
X_min = tf.reduce_min(X_train, axis=0)  # Minimum for each feature
X_max = tf.reduce_max(X_train, axis=0)  # Maximum for each feature
X_train = (X_train - X_min) / (X_max - X_min)
X_test = (X_test - X_min) / (X_max - X_min)

# Convert to TensorFlow datasets for training
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(100)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
# Check class distribution in the training and test data
print("Class distribution in the training data:", np.sum(y_train.numpy(), axis=0))
print("Class distribution in the test data:", np.sum(y_test.numpy(), axis=0))
# Check shapes
print(f"Train data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")#PART C - SELF ORGAINISNG MAP - UNSUPERVISED LEARNING 
import numpy as np
import matplotlib.pyplot as plt
# code adapted form lab 3 auala and chatgpt 
# Kohonen SOM class
class kohonen:
    def __init__(self, number_of_neurons, input_dimension, learning_rate=0.001):
        # Initialize weights randomly
        self.weights = np.random.rand(number_of_neurons, input_dimension)
        self.number_of_neurons = number_of_neurons
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        
    def train(self, inputs, epochs=15):
        # Training process: Update weights based on input data
        for epoch in range(epochs):
            for input in inputs:
                distances = np.linalg.norm(self.weights - input, axis=1)
                winner_index = np.argmin(distances)  # Find the winning neuron
                self.weights[winner_index] += self.learning_rate * (input - self.weights[winner_index])

    def predict(self, input):
        # Predict the class based on the closest matching neuron
        distances = np.linalg.norm(self.weights - input, axis=1)
        return np.argmin(distances)

# Define the number of neurons (one per class in this case, so 3 neurons)
number_of_neurons = 3  # We have 3 classes: Setosa, Versicolor, Virginica

# Initialize and train the SOM
som_network = kohonen(number_of_neurons=number_of_neurons, input_dimension=X_train.shape[1])
som_network.train(X_train.numpy(), epochs=15)  # Training with the preprocessed data

# Loop through all test samples and print the results
for test_sample, true_label in zip(X_test, y_test):  # All test samples
    predicted_class = som_network.predict(test_sample.numpy())
    print(f"Test sample: {test_sample.numpy()}, True label: {true_label}, Predicted Class: {predicted_class}")import numpy as np
import matplotlib.pyplot as plt

# Predict classes for the entire test set using the trained SOM
y_pred_classes = [som_network.predict(test_sample.numpy()) for test_sample in X_test]

# Convert to numpy arrays for convenience
y_test_classes = y_test.numpy()  # True class labels
y_pred_classes = np.array(y_pred_classes)  # Predicted class labels

# Find unique classes in both y_test_classes and y_pred_classes
unique_classes = np.unique(np.concatenate((y_test_classes, y_pred_classes)))
num_classes = len(unique_classes)

# Create confusion matrix manually with the correct number of classes
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

# Map class labels to indices
class_to_index = {label: idx for idx, label in enumerate(unique_classes)}

# Increment the corresponding cell for each pair of true and predicted labels
for t, p in zip(y_test_classes, y_pred_classes):
    t_idx = class_to_index[t]  # Get the index for the true label
    p_idx = class_to_index[p]  # Get the index for the predicted label
    conf_matrix[t_idx, p_idx] += 1  # Increment the corresponding cell

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(5, 5))
cax = ax.matshow(conf_matrix, cmap='Blues')
plt.colorbar(cax)

# Add text annotations
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')
... 
... plt.xlabel("Predicted")
... plt.ylabel("Actual")
... plt.title("Confusion Matrix")
... plt.show()
... #https://www.nomidl.com/machine-learning/what-is-precision-recall-accuracy-and-f1-score/#:~:text=It%20can%20be%20calculated%20by%20dividing%20precision%20by,it%20takes%20into%20account%20both%20Precision%20and%20Recall.
... precision = {}
... recall = {}
... f1 = {}
... 
... # Calculate precision, recall, and F1 score for each class
... for i in range(num_classes):
...     TP = conf_matrix[i, i]  # True Positives for class i
...     FP = conf_matrix[:, i].sum() - TP  # False Positives for class i
...     FN = conf_matrix[i, :].sum() - TP  # False Negatives for class i
...     TN = conf_matrix.sum() - (TP + FP + FN)  # True Negatives for class i
... 
...     # Precision, recall, and F1 score calculations
...     # Handle division by zero by setting it to zero when necessary
...     precision[i] = TP / (TP + FP) if (TP + FP) != 0 else 0 #  presion equation = True positive/(True postive+ False positive)
...     recall[i] = TP / (TP + FN) if (TP + FN) != 0 else 0 #True postive/(True positive + False negative)
...     f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0 # 2*((precison*recall/precison+recall))
... 
...     # Print the results for each class
...     print(f"Class {i}: Precision = {precision[i]:.4f}, Recall = {recall[i]:.4f}, F1 Score = {f1[i]:.4f}")
... # Calculate average precision, recall, and F1 score
... average_precision = np.mean(list(precision.values()))
... average_recall = np.mean(list(recall.values()))
... average_f1 = np.mean(list(f1.values()))
... # Print average scores
... print(f"\nAverage Precision: {average_precision:.4f}")
... print(f"Average Recall: {average_recall:.4f}")
