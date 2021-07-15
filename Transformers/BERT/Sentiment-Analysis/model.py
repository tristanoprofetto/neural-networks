from transformers import TFBertForSequenceClassification

# Loading pretrained BERT model 
model = TFBertForSequenceClassification.from_pretrained("nateraw/bert-case-uncased-imdb", from_pt=True)

# Saving the model as a TensorFlow Keras model
model.save_pretrained("/bert", saved_model=True)