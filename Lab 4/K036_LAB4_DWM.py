#!/usr/bin/env python
# coding: utf-8

# # Python Libraries: Numpy, Pandas, NLTK, Scikit-learn
# 
# This notebook provides an overview of four essential Python libraries for data science and machine learning: **Numpy**, **Pandas**, **NLTK**, and **Scikit-learn**. Each section covers the significance, key functions, and example code for using these libraries.
# 
# ## 1. Numpy
# 
# ### Significance
# Numpy is the core library for numerical and matrix operations in Python. It supports large multi-dimensional arrays and matrices, and provides a collection of mathematical functions to operate on these arrays. It is heavily used in data science, machine learning, and scientific computing.
# 
# ### Functions/Features
# - Array Creation (ndarray)
# - Mathematical operations on arrays
# - Linear Algebra, Trigonometric, and Statistical functions
# - Random sampling and number generation
# - Array manipulation (reshaping, stacking, etc.)
# 
# ### Example Code:
# ```python
# import numpy as np
# 
# # Creating a 2D Numpy array
# array = np.array([[1, 2, 3], [4, 5, 6]])
# 
# # Array operations
# sum_array = np.sum(array)  # Summing all elements in the array
# mean_array = np.mean(array)  # Calculating the mean of array elements
# 
# print("Array:\n", array)
# print("Sum of all elements:", sum_array)
# print("Mean of array elements:", mean_array)
# 

# In[1]:


import numpy as np

# Creating a 2D Numpy array
array = np.array([[1, 2, 3], [4, 5, 6]])

# Array operations
sum_array = np.sum(array)  # Summing all elements in the array
mean_array = np.mean(array)  # Calculating the mean of array elements

print("Array:\n", array)
print("Sum of all elements:", sum_array)
print("Mean of array elements:", mean_array)


# In[2]:


import pandas as pd

# Creating a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)

# Basic operations
mean_age = df['Age'].mean()  # Calculating mean age
city_count = df['City'].value_counts()  # Count occurrences of cities

print("DataFrame:\n", df)
print("Mean Age:", mean_age)
print("City count:\n", city_count)


# In[4]:


import nltk
nltk.download('punkt')


# In[5]:


import nltk
from nltk.tokenize import word_tokenize

# Example text
text = "Hello, how are you doing today?"

# Tokenize text into words
words = word_tokenize(text)

print("Tokenized Words:", words)


# In[6]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a SVM classifier
clf = SVC()

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

