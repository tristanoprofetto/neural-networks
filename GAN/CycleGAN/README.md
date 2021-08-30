# CycleGAN's

Capable of taking of generating a new image from a subset of images. The cool part of CycleGANs is you do not need paired images. This is really useful for scenarios where you may not have paired images. For example, if you want to convert pictures of zebras into pictures of horses. This kind of data would probably be impossible to collect, unless I guess if you painted zebras and horses.

With enough creativity, they also turn out to be useful for creating computer-generated art human made


### Architecture
A notable difference between CycleGAN and other variations of the GAN is the use of residual blocks
A residual block is a stack of layers set in such a way that the output of a layer is taken and added to another layer deeper in the block. In CycleGAN, after the expanding blocks, there are convolutional layers where the output is added to the original input os the variation can be minimized with respect to the images.

