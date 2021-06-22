# Importing required librariries
import urllib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split


def buildModel():

	dataset = pd.read_csv("./spam.csv", encoding="latin-1")
	sentences = dataset['v2']
	labels = np.where(dataset['v1'] == "ham", 0, 1)

	# Initializing Model Parameters
	vocab_size = 1000
	embedding_dim = 16
	max_length = 120
	trunc_type = "post"
	padding_type = "<OOV>"
	training_size = 4500


	
    # Data Preprocessing
	tokenizer = Tokenizer(num_words=vocab_size, oov_token=padding_type)
	tokenizer.fit_on_texts(sentences)
	sequences = tokenizer.texts_to_sequences(sentences)
	sequences = pad_sequences(sequences, maxlen=max_length, padding=trunc_type)
    
    # Splitting Data into train/test sets
	X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2)
    
    
    # Model Architecture
	model = tf.keras.models.Sequential([
		
		tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
        
		tf.keras.layers.Dense(1, activation="sigmoid")
	])
    
    # Model Training
	model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    
	model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
    
	return model


# Saving model as .h5 format
if __name__ == "__main__":
	model = buildModel()
	model.save("mymodel.h5")
