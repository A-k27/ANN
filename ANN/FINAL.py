Python 3.13.2 (tags/v3.13.2:4f8bb39, Feb  4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
### MODEL 1 FOR PART B&C ML technique ANN model: FEEDFORWARD NUREAL NETWORK trained using Backprogation ###

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
# code adated from labs and https://www.tensorflow.org/api_docs/python/tf/keras/utils/split_dataset, and ChatGPT to fix imablnacing issues when spltting data into traing and testing sets
import tensorflow as tf
import numpy as np
# Check unique class labels in y
print("Unique class labels:", np.unique(y))
# Step 1: Group the indices by class
import tensorflow as tf
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
# Step 3: Define split index for 80-20 training and testing (splitting done manually)
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
# Min-Max Scaling- SCALING/ NORMALISING DATA
X_min = tf.reduce_min(X_train, axis=0)  # Minimum for each feature
X_max = tf.reduce_max(X_train, axis=0)  # Maximum for each feature
X_train = (X_train - X_min) / (X_max - X_min)
X_test = (X_test - X_min) / (X_max - X_min)
# One-hot encode the labels - non-numerical to numerica
y_train = tf.one_hot(y_train, depth=3)
y_test = tf.one_hot(y_test, depth=3)
# Convert to TensorFlow datasets for training
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(100)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
# Check class distribution in the training and test data
print("Class distribution in the training data:", np.sum(y_train.numpy(), axis=0))
print("Class distribution in the test data:", np.sum(y_test.numpy(), axis=0))
# Check shapes
print(f"Train data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")
# code adated from labs and https://www.tensorflow.org/api_docs/python/tf/keras/utils/split_dataset, and ChatGPT to fix imablnacing issues when spltting data into traing and testing sets
import tensorflow as tf
import numpy as np
# Check unique class labels in y
print("Unique class labels:", np.unique(y))
# Step 1: Group the indices by class
import tensorflow as tf
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
# Step 3: Define split index for 80-20 training and testing (splitting done manually)
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
# Min-Max Scaling- SCALING/ NORMALISING DATA
X_min = tf.reduce_min(X_train, axis=0)  # Minimum for each feature
X_max = tf.reduce_max(X_train, axis=0)  # Maximum for each feature
X_train = (X_train - X_min) / (X_max - X_min)
X_test = (X_test - X_min) / (X_max - X_min)
# One-hot encode the labels - non-numerical to numerica
y_train = tf.one_hot(y_train, depth=3)
y_test = tf.one_hot(y_test, depth=3)
# Convert to TensorFlow datasets for training
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(100)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
# Check class distribution in the training and test data
print("Class distribution in the training data:", np.sum(y_train.numpy(), axis=0))
print("Class distribution in the test data:", np.sum(y_test.numpy(), axis=0))
# Check shapes
print(f"Train data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")
# PART B
#ANN - FEED FORWARD NEURAL NETWORK USING BACKPROPGATION
import keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout

# creating the model
# has three layers input, output and
# Creating the model using code froom https://medium.com/@elvenkim1/keras-for-dummies-simple-feedforward-neural-network-9b99fcd1df7b and https://www.youtube.com/watch?v=DjMElSxFFtQ and https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/
# Define input size based on dataset
input_size = X_train.shape[1]  # Ensure X_train is properly defined
# Creating the model
model = Sequential([
    Input(shape=(input_size,)),  # Explicit input layer
    Dense(64, activation='relu'),  # First hidden layer
    Dropout(0.3), # regularisation technique by 30%
    Dense(64, activation='relu'),  # Second hidden layer
    Dropout(0.3), # regularisation technique by 30%
    Dense(3, activation='softmax')  # Output layer with 3 classes
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # learning rate optimizer, loss function, specifies accuracy
# Train the model
history = model.fit(train_dataset, epochs=500, validation_data=test_dataset)
# Print model summary
model.summary()
# PART C- EVALUATION OF NEURAL NETWORK
#evalaution of model on test set showing accuracy score and doing prediction
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
y_train_pred = model.predict(X_train)
y_train_classes = np.argmax(y_train_pred, axis=1)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred_classes)
print("Predicted Classes:", y_pred_classes)
print("Actual Classes:", np.argmax(y_test, axis=1))  # Ensure y_test is one-hot encoded
SyntaxError: multiple statements found while compiling a single statement
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
#CODE ADAPTED FOR CNN LAB AND CHATGPT
import numpy as np
# Find unique classes in both y_test_classes and y_pred_classes
unique_classes = np.unique(np.concatenate((y_train_classes, y_pred_classes)))
num_classes = len(unique_classes)
# Create confusion matrix manually with the correct number of classes
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
# Map class labels to indices
class_to_index = {label: idx for idx, label in enumerate(unique_classes)}
# Increment the corresponding cell for each pair of true and predicted labels
for t, p in zip(y_train_classes, y_pred_classes):
    t_idx = class_to_index[t]  # Get the index for the true label
    p_idx = class_to_index[p]  # Get the index for the predicted label
    conf_matrix[t_idx, p_idx] += 1 # Increment the corresponding cell
# Plot confusion matrix
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))
cax = ax.matshow(conf_matrix, cmap='Blues')
plt.colorbar(cax)
# Add text annotations
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
#https://www.nomidl.com/machine-learning/what-is-precision-recall-accuracy-and-f1-score/#:~:text=It%20can%20be%20calculated%20by%20dividing%20precision%20by,it%20takes%20into%20account%20both%20Precision%20and%20Recall.
precision = {}
recall = {}
f1 = {}

# Calculate precision, recall, and F1 score for each class
for i in range(num_classes):
    TP = conf_matrix[i, i]  # True Positives for class i
    FP = conf_matrix[:, i].sum() - TP  # False Positives for class i
    FN = conf_matrix[i, :].sum() - TP  # False Negatives for class i
    TN = conf_matrix.sum() - (TP + FP + FN)  # True Negatives for class i

    # Precision, recall, and F1 score calculations
    # Handle division by zero by setting it to zero when necessary
    precision[i] = TP / (TP + FP) if (TP + FP) != 0 else 0 #  presion equation = True positive/(True postive+ False positive)
    recall[i] = TP / (TP + FN) if (TP + FN) != 0 else 0 #True postive/(True positive + False negative)
    f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0 # 2*((precison*recall/precison+recall))
    # Print the results for each class
    print(f"Class {i}: Precision = {precision[i]:.4f}, Recall = {recall[i]:.4f}, F1 Score = {f1[i]:.4f}")
# Calculate average precision, recall, and F1 score
average_precision = np.mean(list(precision.values()))
average_recall = np.mean(list(recall.values()))
average_f1 = np.mean(list(f1.values()))
# Print average scores
print(f"\nAverage Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")
print(f"Average F1 Score: {average_f1:.4f}")
SyntaxError: multiple statements found while compiling a single statement
#### MODEL 2 FOR PART B&C Unsupervised learning ANN model: SOM trained using Kohonen###
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
print(y)#labs and https://www.tensorflow.org/api_docs/python/tf/keras/utils/split_dataset, and ChatGPT to fix imablnacing issues when spltting data into traing and testing sets
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
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
# Check class distribution in the training and test data
print("Class distribution in the training data:", np.sum(y_train.numpy(), axis=0))
print("Class distribution in the test data:", np.sum(y_test.numpy(), axis=0))
# Check shapes
print(f"Train data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")
print(f"Training class distribution: {dict(zip(range(3), np.sum(y_train.numpy(), axis=0)))}")
print(f"Testing class distribution: {dict(zip(range(3), np.sum(y_test.numpy(), axis=0)))}")
!pip install minisom
SyntaxError: multiple statements found while compiling a single statement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom  # Self-Organizing Map
# code adapted from # code adapted from lab3 and Https://medium.com/@a01620477/clustering-using-self-organizing-maps-ab7e1adfbc77
#and https://github.com/PrachetasPathak/SOM-Iris-Visualization/blob/main/som.py and https://www.datacamp.com/tutorial/self-organizing-maps and chatGPt
# Define SOM parameters
grid_size = (10, 10)  # 10x10 grid
input_len = X_train_scaled.shape[1]  # Number of features (4 for Iris dataset)
# Initialize the SOM
som = MiniSom(grid_size[0], grid_size[1], input_len, sigma=1.0, learning_rate=0.01) # definition of grid, number, numb roof features in dataset(4), neighbouring function spread (radius),how fast weights learn )
som.random_weights_init(X_train_scaled) # added standard initialisation for better converges
# Creating custom training function - decaying learning rate and radius
def train_som_with_decay(som, data, num_epoch=5000, decay_factor=10):
    # Train with decay in learning rate and sigma
    for i in range(num_epoch):
        # Decay the learning rate and radius (sigma) over time
        learning_rate = 0.01 * np.exp(-i / (num_epoch / decay_factor))
        sigma = 1.0 * np.exp(-i / (num_epoch / decay_factor))
        # Update the SOM parameters
        som.learning_rate = learning_rate
        som.sigma = sigma
        # Train on each sample
        for x in data:
            winner = som.winner(x)  # Get the BMU (Best Matching Unit) is the node that eight is most similar top
            som.update(x, winner, learning_rate, sigma)  # Update the weights (neighborhood)
# Visualizing the SOM results
plt.figure(figsize=(10, 10))
for i, (x, label) in enumerate(zip(X_train_scaled, y)):
    w = som.winner(x)  # Find the BMU (Best Matching Unit)
    plt.text(w[0], w[1], str(label), color="red", fontsize=12,
             horizontalalignment='center', verticalalignment='center')

plt.title("Self-Organizing Map (SOM) for Iris Dataset")
plt.xticks(range(10))
plt.yticks(range(10))
plt.grid()
plt.show()
#codde adapted form ChatGPT CNN lab and  #https://www.nomidl.com/machine-learning/what-is-precision-recall-accuracy-and-f1-score/#:~:text=It%20can%20be%20calculated%20by%20dividing%20precision%20by,it%20takes%20into%20account%20both%20Precision%20and%20Recall.
# Step 1: Create a function to map SOM units to class labels
def get_som_labels(som, X_test, y_test): # chenged evaluation of dataset form whole that ase=t to evaluation only on teasing data
    # Get the winning units for each input
    y = np.array(y_test)
    predicted_labels = []
    for x in X_scaled:
        w = som.winner(x)  # Get the BMU for the input
        # Find the class with the most samples in the winning unit's cluster
        # Assuming the SOM grid is (10, 10), so 100 possible units
        cluster_indices = np.where(np.array([som.winner(x) == w for x in X_scaled]))[0]
        cluster_labels = y[cluster_indices]
        majority_label = np.argmax(np.bincount(cluster_labels))  # Most frequent label in cluster
        predicted_labels.append(majority_label)
    return np.array(predicted_labels)
... # Step 2: Map SOM outputs to predicted labels
... predicted_labels = get_som_labels(som, X_scaled, y)
... # Step 3: Create confusion matrix using TensorFlow
... def create_confusion_matrix(true_labels, predicted_labels, num_classes):
...     # Create a confusion matrix
...     confusion_matrix = tf.math.confusion_matrix(true_labels, predicted_labels, num_classes=num_classes)
...     return confusion_matrix
... # Step 4: Set number of classes (3 for Iris dataset)
... num_classes = len(np.unique(y))
... # Step 5: Calculate confusion matrix
... conf_matrix = create_confusion_matrix(y, predicted_labels, num_classes)
... # Step 6: Visualize confusion matrix using Seaborn heatmap
... plt.figure(figsize=(8, 6))
... sns.heatmap(conf_matrix.numpy(), annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
... plt.title("Confusion Matrix for SOM (Iris Dataset)")
... plt.xlabel("Predicted")
... plt.ylabel("Actual")
... plt.show()
... # Calculate precision, recall, and F1 score for each class, dictionaries for doing manually
... precision = {}
... recall = {}
... f1 = {}
... # Convert confusion matrix to NumPy array for easier indexing
... conf_matrix_np = conf_matrix.numpy()
... 
... # Calculate precision, recall, and F1 score for each class
... for i in range(num_classes):
...     TP = conf_matrix_np[i, i]  # True Positives for class i
...     FP = conf_matrix_np[:, i].sum() - TP  # False Positives for class i
...     FN = conf_matrix_np[i, :].sum() - TP  # False Negatives for class i
...     TN = conf_matrix_np.sum() - (TP + FP + FN)  # True Negatives for class i
...     # Precision, recall, and F1 score calculations
...     # Handle division by zero by setting it to zero when necessary
...     precision[i] = TP / (TP + FP) if (TP + FP) != 0 else 0  # Precision equation = True Positive / (True Positive + False Positive)
...     recall[i] = TP / (TP + FN) if (TP + FN) != 0 else 0     # Recall equation = True Positive / (True Positive + False Negative)
...     f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0  # F1 score
...     # Print the results for each class
...     print(f"Class {i}: Precision = {precision[i]:.4f}, Recall = {recall[i]:.4f}, F1 Score = {f1[i]:.4f}")
... # Calculate average precision, recall, and F1 score
... average_precision = np.mean(list(precision.values()))
... average_recall = np.mean(list(recall.values()))
... average_f1 = np.mean(list(f1.values()))
... # Print average scores
... print(f"\nAverage Precision: {average_precision:.4f}")
... print(f"Average Recall: {average_recall:.4f}")
