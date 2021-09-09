# Recurrent Neural Networks (RNN's)

### Architecture
RNN's are a special form of neural networks which leverage their "memory" property in order allow previous outputs to be used as sequential inputs throughout the hidden layers. In a RNN, the information cycles through a loop, it takes into consideration the current input aswell as previous inputs. This ability to process sequential data has allowed for serious advances in the field of AI, more specifically in NLP (natural language processing).

Components:
*  Encoder: transforms the inputs into the required sequential format
*  Hidden State: representation of previous inputs
*  Backpropagation Through Time (BPTT): special training algorithm for weights to be optimized in a way of best handling sequential data such as; text and time-series

### Applications:
* Speech Recognition
* Stock Predictions
* Language Translation
* Sentiment Analysis

 
