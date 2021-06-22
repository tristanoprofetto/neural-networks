# Imporing required libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import image


# Loading dataset from tensorflow datasets package
dataset, info = tfds.load(name='colorectal_histology', split=tfds.Split.TRAIN, with_info=True)

# Preprocess data 
def preprocess(features):
	# YOUR CODE HERE
	image = features['image']
	label = features['label']

	image = tf.cast(image, tf.float32)
	label = tf.cast(label, tf.float32)

	image /= 255.0

	image = tf.image.resize(image, (150, 150))

	return image, label


def buildModel():
	train = dataset.map(preprocess).batch(32)
    
    # Model Architecture
	model = tf.keras.models.Sequential([
		
		tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=(150, 150, 3)),
		tf.keras.layers.MaxPool2D(2, 2),
		tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
		tf.keras.layers.MaxPool2D(2, 2),
		tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"),
		tf.keras.layers.MaxPool2D(2, 2),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(units=512, activation="relu"),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(units=256, activation="relu"),
		tf.keras.layers.Dense(8, activation="softmax")
        
	])
    
    # Model training
	model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
	model.fit(train, epochs=10)

	return model


# Saving model in .h5 format
if __name__ == "__main__":
	model = solution_model()
	model.save("mymodel.h5")
