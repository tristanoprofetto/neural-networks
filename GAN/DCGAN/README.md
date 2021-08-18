# Deep Convolutional GAN (DCGAN)

Main features of DCGAN:
* uses convolutions without pooling layers
* batchnorm present in both the generator and the discriminator
* Use ReLU activation in the generator for all hidden layers (except output)
* generator output layer uses a Tanh activation
* LeakyReLU activation in the discriminator for all hidden layers (except output)
