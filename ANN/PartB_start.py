# PROBLEM: CLASSIFY SMAPLES INTO ONE OF 3 SPECIES: SETOSA, VERSICOLOUR OR VIRGINICA
# USING TWO DIFFERENT NUMRAL NETWORKS
# PART A
# data preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds
# Load and preprocess dataset
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
# one-hot encoded column to regroup them into sclass of species
df['class'] = df[['class_Iris-setosa', 'class_Iris-versicolor', 'class_Iris-virginica']].idxmax(axis=1)
df.drop(['class_Iris-setosa', 'class_Iris-versicolor', 'class_Iris-virginica'], axis=1, inplace=True)
# Check the result
df.head()
# separates x and y (taregt and features )
x = df.drop(['class'],axis=1).values  # factors
y = df['class'].values  # Target
# One-hot encode 'class' column
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['class'] = df['class'].map(class_mapping)  # Convert class labels to integers
import tensorflow as tf
# Convert NumPy arrays to TensorFlow tensors
X_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y, dtype=tf.bool)
#  Min-Max Scaling in TensorFlow
X_min = tf.reduce_min(X_tensor, axis=0)  # Minimum for each feature
X_max = tf.reduce_max(X_tensor, axis=0)  # Maximum for each feature
# Apply Min-Max Scaling
X_normalized = (X_tensor - X_min) / (X_max - X_min)
# Define split index
split_index = int(0.8 * len(x))
# Split data into test and training in tensorflow
X_train, X_test = tf.split(X_normalized, [split_index, len(x) - split_index], axis=0)
y_train, y_test = tf.split(y_tensor, [split_index, len(y) - split_index], axis=0)
# Convert to TensorFlow datasets for training
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(100)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
# Check shapes
print(f"Train data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")
# PART B
#ANN - FEED FORWARD NEURAL NETWORK USING BACKPROPGATION
import keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
# creating the model
# has three layers input, output and

# Creating the model using code froom https://medium.com/@elvenkim1/keras-for-dummies-simple-feedforward-neural-network-9b99fcd1df7b
model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(784,)))  # Use activation as a string
model.add(Dense(10, activation='softmax'))  # Use activation as a string
model.summary()
