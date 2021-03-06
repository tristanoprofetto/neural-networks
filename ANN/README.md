# Artificial Neural Networks (ANN's)

Also known as **Vanilla** Neural Networks, ANN's are the simplest implementation of deep learning model architectures.

Inspired from the human brain, ANN's were first designed for the purpose of mimicing the way human brain operates in order to learn and solve problems.

Network Architecture: two main components
* **FeedForward Mechanism**: a sequence of linear, and non-linear functions applied to the input vector for a given number of hidden layers for producing an output.
* **Backpropagation**: an algorithm which optimizes the model weights through a recursive formula of partial derivatives allowing the network to "learn".

### Understanding the FeedForward Mechanism

Necessary components of producing an output:
* Input Vector **X**
* Weights Matrices **w** (randomly intialized)
* Non-Linear Activation Function **h(X)** (typically; sigmoid, tanh, ore relu)

Consider a single hidden layer ANN....
* First Operation: a = w.X + b
* Output: h(a)


### Understanding Backpropagation

Backpropagation is the backbone of deep learning models today, without the ability for these models to learn, we would not have advanced in the field as much as we have today (thank sir Geoffrey Hinton).

Components for performing Backpropagation:
* Predictions Vector **y^**
* Response variable vector **Y**
* Derivative of the Activation Function **g(a)**
* Cost function **C** (typically; quadratic or cross-entropy)
