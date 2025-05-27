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
# Min-Max Scaling
X_min = tf.reduce_min(X_tensor, axis=0)  # Minimum for each feature
X_max = tf.reduce_max(X_tensor, axis=0)  # Maximum for each feature
X_scaled = (X_tensor - X_min) / (X_max - X_min)
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
X_train = tf.gather(X_scaled, train_indices)
y_train = tf.gather(y_tensor, train_indices)
X_test = tf.gather(X_scaled, test_indices)
y_test = tf.gather(y_tensor, test_indices)
X_min = np.min(X_train.numpy(), axis=0)
X_max = np.max(X_train.numpy(), axis=0)
X_train_scaled = (X_train.numpy() - X_min) / (X_max - X_min)
X_test_scaled = (X_test.numpy() - X_min) / (X_max - X_min)
# One-hot encode the labels
y_train = tf.one_hot(y_train, depth=3)
y_test = tf.one_hot(y_test, depth=3)
# Convert to TensorFlow datasets for training
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(100)
... test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
... # Check class distribution in the training and test data
... print("Class distribution in the training data:", np.sum(y_train.numpy(), axis=0))
... print("Class distribution in the test data:", np.sum(y_test.numpy(), axis=0))
... # Check shapes
... print(f"Train data shape: {X_train.shape}, {y_train.shape}")
... print(f"Test data shape: {X_test.shape}, {y_test.shape}")
... print(f"Training class distribution: {dict(zip(range(3), np.sum(y_train.numpy(), axis=0)))}")
... print(f"Testing class distribution: {dict(zip(range(3), np.sum(y_test.numpy(), axis=0)))}")
... print(X_train_scaled.shape)
... # trying out a SOM using minSOm libries 
... # code adapted from ## https://medium.com/@a01620477/clustering-using-self-organizing-maps-ab7e1adfbc77
... # trying out a SOM using minSOm libries 
... # code adapted from ## https://medium.com/@a01620477/clustering-using-self-organizing-maps-ab7e1adfbc77 and chatGPT and https://www.datacamp.com/tutorial/self-organizing-maps?utm_source=chatgpt.com
... from minisom import MiniSom
... grid_size = (10, 10)  # 10x10 grid
... input_len = X_train_scaled.shape[1]  # Number of features (4 for Iris dataset)
... som = MiniSom(grid_size[0], grid_size[1], input_len, sigma=4, learning_rate=0.0, neighborhood_function='triangle') 
... som.pca_weights_init(X_train_scaled)
... som.train(X_test_scaled, 5000, random_order=True, verbose=True)
... plt.figure(figsize=(40,40))
... wmap = {}
... im = 0 
... for x, t in zip(X_test_scaled,y):
...   w = som.winner(x)
...   wmap[w] = im
...   plt. text (w[0]+0.5, w[1]+0.5, str(t), 
...            color = plt.cm.rainbow(t/10.), fontdict={'weight': 'bold', 'size': 11})
... 
...   im = im + 1
... plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])
