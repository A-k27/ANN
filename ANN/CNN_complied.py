Python 3.13.2 (tags/v3.13.2:4f8bb39, Feb  4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
# PROBLEM: CLASSIFY SMAPLES INTO ONE OF 3 SPECIES: SETOSA, VERSICOLOUR OR VIRGINICA
# USING TWO DIFFERENT NUMRAL NETWORKS
# PART A
# data preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds
from tensorflow import keras
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
# converse dat a into tensorflow to start creating neraul network
import tensorflow as tf
# Convert NumPy arrays to TensorFlow tensors
# Convert NumPy arrays to TensorFlow tensors
X_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
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
#chack data type
print (f"Train data shape: {X_train.dtype}, {y_train.dtype}")
print(f"Test data shape: {X_test.dtype}, {y_test.dtype}")
# check the rank of tensor (number of deimentions )
print("Rank (order): ",tf.rank(X_train))
... print("Rank (order): ",tf.rank(y_train))
... #Creating a Convolutional neural network
... # reshape data for an image input where reshape matches height,width,channels
... x_train, x_test = X_train / 255.0, X_test / 255.0
... X_train_reshaped = tf.reshape(X_train, (-1, 2, 2, 1))  # Reshaping for CNN
... X_test_reshaped = tf.reshape(X_test, (-1, 2, 2, 1))
... print (f"Train data shape: {X_train_reshaped.shape}, {y_train.shape}")
... print(f"Test data shape: {X_test_reshaped.shape}, {y_test.shape}")
... # Convert labels to categorical format
... y_train = keras.utils.to_categorical(y_train, 3)
... y_test = keras.utils.to_categorical(y_test, 3)
... # PART B
... #ANN - FEED FORWARD NEURAL NETWORK USING BACKPROPGATION
... # FEED Fordwared neuron that need to still be trained using backprogation
... import keras
... from tensorflow import keras
... from keras.models import Sequential
... from tensorflow.keras import layers,models
... from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
... from keras.optimizers import SGD
... from keras.optimizers import Adam
... from keras.optimizers import RMSprop
... model = keras.Sequential([
...     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(2, 2, 1), padding='same'),  # Use padding='same'
...     layers.MaxPooling2D((2, 2)),  # Pooling layer
...     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  # Use padding='same'
...     layers.MaxPooling2D((2, 2)),  # Pooling layer
...     layers.Flatten(),  # Flatten the output
...     layers.Dense(128, activation='relu'),  # Fully connected layer
...     layers.Dense(10, activation='softmax')  # Output layer
... ])
... y_train = keras.utils.to_categorical(y_train, 3)
... y_test = keras.utils.to_categorical(y_test, 3)
... 
... # Compile and train the model
... # Compile the model
