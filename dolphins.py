import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

print("yes")

# read data
data = pd.read_csv("all_classifcation_and_seqs_aln.csv")
data = data.dropna() # drop na values

# initialize
sequence = data["sequence"]
X = []

# encode DNA sequences into numbers 
encoder = LabelEncoder()
encoder = encoder.fit(list(sequence[0]))
for i in range(58699): 
    encoded = encoder.transform(list(sequence[i]))
    X.append(encoded)

# standardization
X = np.array(X, dtype=np.float32)
X = (X - X.mean()) / X.std()

# encode species lables 
y = data["species"]
species_encoder = LabelEncoder()
y_encoded = species_encoder.fit_transform(y)

#split train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

sequence_length = X.shape[1]
num_classes = len(species_encoder.classes_)

# build model 
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(46, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

# assign learning rate 
lr = 0.0001

# compile model 
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=['accuracy']
)

# train the model
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))

# plot graph 
df = pd.DataFrame(history.history)

"""
What different settings did you experiment with, and how did each one affect your model’s performance?
i mostly modified the validation split, learning rate, and the number of layers and neurons. The validation split was what brought my val_accuracy score from 87 to 90. 


Describe which choices ultimately worked best, which did not, and provide reasoning for why you think those outcomes occurred.
The validation split was what made the biggest impact along with the number of epochs and learning rate. 
There wasnʻt really anything that didnʻt make a big impact but sometimes i would try something new and it would cap my val accuracy at 58 again. This was probably due to overfitting on the training set. 


"""