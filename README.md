# DCGAN: A strong candidate for unsupervised learning

In the recent years, Supervised learning with CNNs has very good applications in the field of computer vision. Compared to it, Unsupervised learning with CNNs has not received the required attention which it deserves. To reduce the gap between the supervised and unsupervised learning, a new class of CNNs called deep convolutional generative adversarial networks (DCGANs) was introduced. 

We know that there is an unlimited amount of unlabeled data. But with the help of computer
vision, one can utilize the data to learn the immediate representation of the dataset and extract
the beneficial information which can be used for a variety of supervised learning tasks such as
image classification, object detection, and image segmentation. The traditional generative methods often rely on Maximum Likelihood Estimation (MLE) such as Gaussian Mixture Models (GMMs) and Variational Autoencoders (VAEs) but they sometimes face difficulty due to high-dimensional data and produce vague output. The methods and learning process used by GAN and the
absence of heuristic cost function make it attractive for representation learning. Very limited
research has been done to understand and visualize what exactly the GANs are learning and
how they are learning.

- Such an architecture and constraints have been established in the deep convolutional generative adversarial networks(DCGANs) which make them stable to learn for most of the datasets.
- The discriminator which has been trained for images classification and segmentation tasks is giving competitive results with unsupervised algorithms.
- After visualizing the filters learned by the GANs we got to know that they have become proficient at drawing particular objects.

![i1](https://drive.google.com/uc?id=1SkBhH62zVd6mZN0DINdcbY7hWoLydhBf)

#### Modifications which have been made to DCGANs from the original GAN :

- Replacing all max pooling, min pooling, average pooling with convolutional stride
- Using transposed convolution for upsampling and downsampling
- Eliminated fully connected layers which increased the the model's stability
- Using Batch normalization which transforms each feature into zero mean and unit variance except the output layer for the generator and the input layer of the discriminator
- The ReLU activation function is used in the discriminator, except for the output layer where
the tanh function is used
- Using LeakyReLU in the discriminator

> Note: Batch Normalization is a method to transforms the data at each layer adaptively to zero mean and unit variance. During training, At each step of gradient descent, a small batch of data is sampled form the dataset.
It is used to stabilize training, accelerate convergence and redice sensitivity effect and improve gradient effect.

![i1](https://drive.google.com/uc?id=1esh5C8uUon1B-QMeP1qD5Bh9KhlDUisS)

#### Related work

- The traditional and classic approach is to do clustering on data using K means. Coats
and Ng have suggested doing the hierarchical clustering of image patches to learn
powerful image representations.
- Another method is to train the auto-encoders which compresses input data into its
most important features and then reconstructs the original input from that compressed
representation.
- Another approach is the ladder structures that transform the image into compact
code and re-transform it into the image as accurately as possible.
- It has also been found that Deep belief networks are also working well in learning
hierarchical representations.

## Detail of Adversarial Training

Training was performed by mini batch stochastic gradient descent with mini-batch size of 128
All weights were initialized from zero-centered Normal distribution with standard deviation 0.02
LeakyReLU slope was set to 0.2 and learning rate to 2e-4
Adam Optimizer beta parameter 0.5 where it was tuned through grid search CV
The suggested momentum parameter value was 0.9, but it was later set to 0.5 due to unstable and oscillating training

> Note: The Adam(Adaptive Moment Estimation) optimizer is an iterative optimization algorithm, which take momentum into consideration, is used to minimize the loss function during the training of neural networks. Given below, g is the mean of each gradient. β1 is the decay rate of the mean and β2 is decay rate for the squared gradients.Here are the key equations used in the Adam optimizer:
     <pre>
   m<sub>t</sub> = β<sub>1</sub> * m<sub>t-1</sub> + (1 - β<sub>1</sub>) * g<sub>t</sub>
   v<sub>t</sub> = β<sub>2</sub> * v<sub>t-1</sub> + (1 - β<sub>2</sub>) * g<sub>t</sub><sup>2</sup>
   </pre>
    <pre>
   m̂<sub>t</sub> = m<sub>t</sub> / (1 - β<sub>1</sub><sup>t</sup>)
   v̂<sub>t</sub> = v<sub>t</sub> / (1 - β<sub>2</sub><sup>t</sup>)
   </pre>
Where:
- <code>θ<sub>t</sub></code>: Parameters at time step <code>t</code>
- <code>α</code>: Learning rate
- <code>β<sub>1</sub></code>: Exponential decay rate for the first moment estimate
- <code>β<sub>2</sub></code>: Exponential decay rate for the second moment estimate
- <code>ε</code>: Small constant to prevent division by zero

### DCGAN Equations and Formulas


1. **Generator**: The objective function for the generator is:
   <pre>
   L<sub>G</sub> = -E<sub>x~p<sub>data</sub></sub>[log(D(G(z)))]
   </pre>
   Where:
   - <code>G(z)</code>: New data generated from noise vector <code>z</code>
   - <code>D(G(z))</code>: Discriminator's classifiyng the data
   - <code>E<sub>x~p<sub>data</sub></sub>[ ]</code>: Expectedvalue over the data distribution

2. **Discriminator Objective**:The objective function for the discriminator is:
   <pre>
   L<sub>D</sub> = -E<sub>x~p<sub>data</sub></sub>[log(D(x))] - E<sub>z~p<sub>z</sub></sub>[log(1 - D(G(z)))]
   </pre>
   Where:
   - <code>D(x)</code>: Discriminator classifiyng the data
   - <code>D(G(z))</code>: Discriminator's prediction of generated data being real

3. **Training Process**:
   - **Discriminator Update**:
     <pre>
     θ<sub>D</sub> ← θ<sub>D</sub> - α<sub>D</sub> ∇<sub>θ<sub>D</sub></sub> L<sub>D</sub>
     </pre>
   - **Generator Update**:
     <pre>
     θ<sub>G</sub> ← θ<sub>G</sub> - α<sub>G</sub> ∇<sub>θ<sub>G</sub></sub> L<sub>G</sub>
     </pre>
   Where:
   - <code>θ<sub>D</sub></code>,<code>θ<sub>G</sub></code>: Parameters
   - <code>α<sub>D</sub></code>, <code>α<sub>G</sub></code>: Learning rates for discriminator and generator
   - <code>∇<sub>θ<sub>D</sub></sub></code>, <code>∇<sub>θ<sub>G</sub></sub></code>: Loss function's gradient with respect to the parameters

4. **Minimax Approach**:
   <pre>
   min<sub>G</sub> max<sub>D</sub> L<sub>D</sub>
   </pre>
   Here the generator tries to minimize the probability of the generator classifying the data correctly. It implies that the discriminator tries to minimize its classification error rate.

![i1](https://drive.google.com/uc?id=12OquuqsZW7AM8SddlJvPoQa3JUk2U3wd)

With more accurate models, the visual quality of the sample dataset has been improved, but
there was concern over the overfitting and memorization of the training samples. To check how
our models performs on huge data and for higher resolution generation, the model was trained
on the LSUN bedroom dataset, which contains more than 3 million images of bedroom. Data
augmentation was not applied to the images.


#### Experiment and Results performed on various datasets

Experiment was performed on CIFAR10 in which which was utilizing K-means as a feature learning algorithm and with a huge amount of feature maps (4800), it gives 80.6% accuracy. An unsupervised multi-layered extension of this algorithm takes the accuracy to 82.0%. Then the DCGANs + L2-SVM achieves 82.8% accuracy and they have completely out-performed the all K-means based approaches.

![i1](https://drive.google.com/uc?id=1LUyi4It-zgQFIP65QEaobOQUOq_gGflp)

Street View House Numbers(SVHN) dataset was divided into validation set of 10,000 examples, which was used for model selection and selecting the best hyper-parameters. 1,000 training examples were randomly chosen and used to train on the same L2-SVM classifier was used for CIFAR-10. Hhere DCGANs with L2-SVM give an error-rate 22.48% only.

![i1](https://drive.google.com/uc?id=1UsbV-XAAw0OMfoqz8lm46T5g5iGKJBJM)

Let us end then with the conclusion here, It introduced a breakthrough in generating high-quality and high-resolution images using neural networks. It tells that how the convolutional netwroks can be effectively used in the architecture of GANs where we replaced fully connceted networks with convoutional networks in the generator and discriminator which gives more stable training and produced high quality images. It has the ability to capture the more complex patterns in the data leading to more realistic and detailed images. The success of DCGANs has laid the foundation for numerous advancements in GAN research, which includes the architectures like StyleGAN, CycleGAN etc. It had proved that convolutional architectures could be helpful in generative models and motivated the exploration of various modifications and improvements in GAN training and architecture design.


