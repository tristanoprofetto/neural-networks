# Generative Adversarial Networks (GAN's)

GAN's are a special type of neural network optimized for computer vision tasks. This architecture involves two models against each other, one tasked with creating something based on existing data and the other tasked with finding flaws in the new output. When flaws are discovered, the creative network (known as the “generator”) learns from its mistakes and eventually becomes capable of creating data that its opponent (the “discriminator” network) finds increasingly hard to distinguish from the existing data.

Components:
* Generator: learns how to generate images from a random noise vector ...  the generator’s goal is to trick the discriminator
* Discriminator: trained to recognize real images from the fake images provided by the generator


### Challenges when Evaluating GAN's
One aspect that makes evaluating GANs challenging is that the loss tells us little about their performance. Unlike with classifiers, where a low loss on a test set indicates superior performance, a low loss for the generator or discriminator suggests that learning has stopped.

### Detecting Bias in GAN's
One major challenge in assessing bias in GANs is that you still want your generator to be able to generate examples of different values of a protected class—the class you would like to mitigate bias against. While a classifier can be optimized to have its output be independent of a protected class, a generator which generates faces should be able to generate examples of various protected class values.

When you generate examples with various values of a protected class, you don’t want those examples to correspond to any properties that aren’t strictly a function of that protected class. This is made especially difficult since many protected classes (e.g. gender or ethnicity) are social constructs, and what properties count as “a function of that protected class” will vary depending on who you ask. It’s certainly a hard balance to strike.

Moreover, a protected class is rarely used to condition a GAN explicitly, so it is often necessary to resort to somewhat post-hoc methods (e.g. using a classifier trained on relevant features, which might be biased itself).
