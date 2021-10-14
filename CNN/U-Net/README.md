# U-Net

U-Net is a CNN architecture designed for handling biomedical image segmentation tasks ... the architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization

Similar to a classifier, the model is responsible for labeling neurons

### Architecture
* Multi-Channel Feature Maps: takes in a tensor with arbitrarily many tensors and produces a tensor with the same number of pixels but with the correct number of output channels
* Contracting Path: the encoder section of the U-Net, which has several downsampling steps as part of it ...
* Expanding Path: the decoding section of U-Net which has several upsampling steps
