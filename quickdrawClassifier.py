# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Intro to Computer Vision
# In this notebook you will use the [quickdraw](https://quickdraw.withgoogle.com/data) dataset to build a simple image recongnition neural net.
#
# We will be using matplotlib for this assignment so start with installing it via pip.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt


# %% [markdown]
# # Step 1: Data
#
# Quickdraw supports A LOT of images. Rather than training on all 50 million, we will start with just a few. 
# Your model should train on at least 5 categories of drawings. 
# This will be more fun if they look similar to each other. 
#
# Go to [quickdraw](https://quickdraw.withgoogle.com/data) and select at least 5 categories to use for your model. These are stored in a Google Cloud storage bucket called quickdraw_dataset.
#
# Load each of your categories by these steps:
#
# 1. ` pip install gsutil`
#
# 2.  `gsutil -m cp 'gs://quickdraw_dataset/full/numpy_bitmap/<your category>.npy' .`
#
#
# Do step 2 once per category. 
#
# *You will need to do this both locally and in your gcloud vm. (they are too big to push to github)

# %% [markdown]
# ## Load Your Data
#
# Your data will be in a .npy format (numpy array). We will load it differently this time: for example
#
# `data_airplanes = np.load("airplane.npy")`

# %%
# Helper function to show images from your dataset
def show_image(data):
    index = random.randint(0, len(data)-1)
    img2d = data[index].reshape(28, 28)
    plt.figure()
    plt.imshow(img2d)
    plt.colorbar()
    plt.grid(False)
    plt.show()


# %%
# Load your first cateogry of images
data_apple = np.load("apple.npy")

# %%
# Now use the function above to show an image from your data
show_image(data_apple)

# %%
# Now load your second category of images
data_airplane = np.load("airplane.npy")

# %%
# visualize your second category of images
show_image(data_airplane)

# %%
# Now load your third category of images
data_angel = np.load("angel.npy")

# %%
# visualize your third category of images
show_image(data_angel)

# %%
# Now load your 4th category of images
data_anvil = np.load("anvil.npy")

# %%
# visualize your 4th category of images
show_image(data_anvil)

# %%
# Now load your 5th category of images
data_book = np.load("book.npy")

# %%
# visualize your 5th category of images
show_image(data_book)

# %% [markdown]
# ## Preprocess your data
#
# Now it's time to define X and y. We want our X and y data to include ALL of the categories you loaded above.
#
# You can combine np.arrays together using something like `np.vstack`.
#

# %%
# define X by combining all loaded data from above
X = np.vstack((data_book, data_airplane, data_angel, data_anvil, data_apple))

X = X.reshape(-1, 28, 28, 1)

# %%
# verify the X was defined correctly

# %% [markdown]
# Now it's time to define y. Recall that y is an array of "correct labels" for our data. For example, ['airplane', 'airplane', 'cat', 'bird', ....]
#
#
# In the above step you created an array of images stored in X of 5 different categories. Now, create an array of labels to match those images. 

# %%
# define y by creating an array of labels that match X
y = np.concatenate([
    np.full(len(data_airplane), 0),
    np.full(len(data_angel), 1),
    np.full(len(data_anvil), 2),
    np.full(len(data_apple), 3),
    np.full(len(data_book), 4)
]
)

# %%
# verify that y is the same length as X
assert len(y) == len(X)

# %% [markdown]
# ## Split your data
#
# Split your data is 80 training/ 20 testing as usual. 

# %%
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# # Define your model
#
# Now define your neural network using tensorflow. 

# %%
# Define your model with the correct input shape and appropriate layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax'),
])

# %%
# Compile your model
lr = 0.001

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.AdamW(learning_rate=lr),
    metrics=['accuracy']
)

# %% [markdown]
# ## Train your model locally
#
# Train your model. 

# %%
# Fit your model 
history = model.fit(X_train, y_train, epochs=15)

# %% [markdown]
# # Train your model on gcloud
#
# Your model likely is a bit slow... move it to the Gcloud gpu. Since we can't run notebooks in gcloud, you can convert it to a python file first by running:
#
# `jupytext --to py quickdrawClassifier.ipynb`
#
# Then push the generated quickdrawClassifier.py file to github and then pull it down to your gcloud laptop to run.
