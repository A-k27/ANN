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
# reshape data
# reshape data for an image input where reshape matches height,width,channels
X_normalized  = tf.reshape(X_normalized, [-1, 2, 2, 1])
y_tensor = keras.utils.to_categorical(y_tensor, 3)
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
print("Rank (order): ",tf.rank(y_train))
#Creating a Convolutional neural network
# PART B
#ANN - CONVOLUTION NEURAL NETWORK
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(2, 2, 1), padding='same'),  # first convolution layer
    layers.MaxPooling2D((2, 2)),  # Pooling layer whic reduces the spatial dimensions
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  # second convolution layer - three pooling layer removed form code as it was not needed
    layers.Flatten(),  # Flatten the output into a one demitional array
    layers.Dense(128, activation='relu'),  # Fully connected layer
    layers.Dense(3, activation='softmax')  # Output layer (3 classes)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(train_dataset, epochs=25, validation_data=test_dataset)
# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# Evaluate model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
# If y_test is one-hot encoded, convert it to class labels
y_test_classes = np.argmax(y_test, axis=1)
# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert to class labels
# Create confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
# Plot confusion matrix code adapted form aula lab 6 
fig, ax = plt.subplots(figsize=(5, 5))
cax = ax.matshow(conf_matrix, cmap='Blues')
plt.colorbar(cax)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')

plt.xlabel("Predicted")
plt.ylabel("Actual")
... plt.title("Confusion Matrix")
... plt.show()
... # Manually calculate Precision, Recall, and F1 Score as not using sklearn 
... def calculate_metrics(conf_matrix):
...     precision = []
...     recall = []
...     f1_scores = []
...     for i in range(len(conf_matrix)):
...         TP = conf_matrix[i, i]  # True Positives
...         FP = np.sum(conf_matrix[:, i]) - TP  # False Positives
...         FN = np.sum(conf_matrix[i, :]) - TP  # False Negatives
...         TN = np.sum(conf_matrix) - (TP + FP + FN)  # True Negatives   
...         # Precision and Recall calculations
...         if TP + FP > 0:
...             p = TP / (TP + FP)
...         else:
...             p = 0     
...         if TP + FN > 0:
...             r = TP / (TP + FN)
...         else:
...             r = 0 
...         f1 = 2 * (p * r) / (p + r) if p + r > 0 else 0      
...         precision.append(p)
...         recall.append(r)
...         f1_scores.append(f1)
...     return precision, recall, f1_scores
... # Calculate metrics for each class
... precision, recall, f1_scores = calculate_metrics(conf_matrix)
... # Print the metrics for each class
... for i in range(len(precision)):
...     print(f"Class {i}: Precision = {precision[i]:.4f}, Recall = {recall[i]:.4f}, F1 Score = {f1_scores[i]:.4f}")
... # Calculate average Precision, Recall, and F1 Score (macro average)
... average_precision = np.mean(precision)
... average_recall = np.mean(recall)
... average_f1 = np.mean(f1_scores)
... print(f"Average Precision: {average_precision:.4f}")
... print(f"Average Recall: {average_recall:.4f}")
... print(f"Average F1 Score: {average_f1:.4f}")
