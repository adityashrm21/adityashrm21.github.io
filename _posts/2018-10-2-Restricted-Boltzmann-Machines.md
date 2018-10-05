---
layout: post
title: Demystifying Restricted Boltzmann Machines
---

In this post, I will try to shed some light on the intuition about Restricted Boltzmann Machines and the way they work. This is supposed to be a simple explanation without going too deep into mathematics and will be followed by a post on an application of RBMs. So let's start with the origin of RBMs and delve deeper as we move forward.

## What are Boltzmann Machines?

Boltzmann machines are stochastic and generative neural networks capable of learning internal representations, and are able to represent and (given sufficient time) solve difficult combinatoric problems.

They are named after the [Boltzmann distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution) (also known as Gibbs Distribution) which is an integral part of Statistical Mechanics and helps us to understand impact of parameters like Entropy and Temperature on Quantum States in Thermodynamics. That's why they are called as Energy Based Models (EBM). They were invented in 1985 by Geoffrey Hinton, then a Professor at Carnegie Mellon University, and Terry Sejnowski, then a Professor at Johns Hopkins University

### How do Boltzmann Machines work?

A Boltzmann Machine looks like this:

<center> <img src = "https://upload.wikimedia.org/wikipedia/commons/7/7a/Boltzmannexamplev1.png" width = "300"> </center>


Boltzmann machines are non-deterministic (or stochastic) generative Deep Learning models with only two types of nodes - `hidden` and `visible` nodes. There are no output nodes! This may seem strange but this is what gives them this non-deterministic feature. They don't have the typical 1 or 0 type output through which patterns are learnt and optimized using Stochastic Gradient Descent. They learn patterns without that capability and this is what makes them so special!

One difference to note here is that unlike the other traditional networks (A/C/R) which don't have any connections between the input nodes, a Boltzmann Machine has connections among the input nodes. We can see from the image that all the nodes are connected to all other nodes irrespective of whether they are input or hidden nodes. This allows them to share information among themselves and self-generate subsequent data. We only measure what's on the visible nodes and not what's on the hidden nodes. When the input is provided, they are able to capture all the parameters, patterns and correlations among the data. This is why they are called `Deep generative models` and fall into the class of `Unsupervised Deep Learning`.

## What are Restricted Boltzmann Machines?

RBMs are two-layered artificial neural network with generative capabilities. They have the ability to learn a probability distribution over its set of input. RBMs were invented by Geoffrey Hinton and can be used for dimensionality reduction, classification, regression, collaborative filtering, feature learning and topic modeling.

RBMs are a special class of [Boltzmann Machines](https://en.wikipedia.org/wiki/Boltzmann_machine) and they are restricted in terms of the connections between the visible and the hidden units. This makes it easy to implement them when compared to Boltzmann Machines. As stated earlier, they are a two-layered neural network (one being the visible layer and the other one being the hidden layer) and these two layers are connected by a fully bipartite graph. This means that every node in the visible layer is connected to every node in the hidden layer but no two nodes in the same group are connected to each other. This restriction allows for more efficient training algorithms than are available for the general class of Boltzmann machines, in particular the [gradient-based](https://en.wikipedia.org/wiki/Gradient_descent) contrastive divergence algorithm.

A Restricted Boltzmann Machine looks like this:
<center> <img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Restricted_Boltzmann_machine.svg/440px-Restricted_Boltzmann_machine.svg.png" width = "300"> </center>

### How do Restricted Boltzmann Machines work?


In an RBM, we have a symmetric bipartite graph where no two units within a same group are connected. Multiple RBMs can also be `stacked` and can be fine-tuned through the process of gradient descent and back-propagation. Such a network is called as a Deep Belief Network. Although RBMs are occasionally used, most people in the deep-learning community have started replacing their use with General Adversarial Networks or Variational Autoencoders.

RBM is a Stochastic Neural Network which means that each neuron will have some random behavior when activated. There are two other layers of bias units (hidden bias and visible bias) in an RBM. This is what makes RBMs different from autoencoders. The hidden bias RBM produce the activation on the forward pass and the visible bias helps RBM to reconstruct the input during a backward pass. The reconstructed input is always different from the actual input as there are no connections among the visible units and therefore, no way of transferring information among themselves.

<center> <img src = "https://skymind.ai/images/wiki/multiple_inputs_RBM.png" width = "400"> </center>

The above image shows the first step in training an RBM with multiple inputs. The inputs are multiplied by the weights and then added to the bias. The result is then passed through a sigmoid activation function and the output determines if the hidden state gets activated or not. Weights will be a matrix with number of input nodes as the number of rows and number of hidden nodes as the number of columns. The first hidden node will receive the vector multiplication of the inputs multiplied  by the first column of weights before the corresponding bias term is added to it.

And if you are wondering what a sigmoid function is, here is the formula:

$$ S(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{1 + e^{x}} $$

So the equation that we get in this step would be,

$$ \textbf{h}^{(1)} = S(\textbf{v}^{(0)T}W + \textbf{a})$$

where $$\textbf{h}^{(1)}$$ and $$\textbf{v}^{(0)}$$ are the corresponding vectors (column matrices) for the hidden and the visible layers with the superscript as the iteration ($$\textbf{v}^{(0)}$$ means the input that we provide to the network) and $$\textbf{a}$$is the hidden layer bias vector.

(Note that we are dealing with vectors and matrices here and not one-dimensional values.)


<center> <img src = "https://skymind.ai/images/wiki/reconstruction_RBM.png" width = "500"> </center>

Now this image show the reverse phase or the **reconstruction** phase. It is similar to the first pass but in the opposite direction. The equation comes out to be:

$$\textbf{v}^{(1)} = S(\textbf{h}^{(1)}W^T + \textbf{a})$$

where $$\textbf{v}^{(1)}$$ and $$\textbf{h}^{(1)}$$ are the corresponding vectors (column matrices) for the visible and the hidden layers with the superscript as the iteration and $$\textbf{b}$$ is the visible layer bias vector.


#### The learning process

Now, the difference $$\textbf{v}^{(0)} - \textbf{v}^{(1)}$$ can be considered as the reconstruction error that we need to reduce in subsequent steps of the training process. So the weights are adjusted in each iteration so as to minimize this error and this is what the learning process essentially is.
Now, let us try to understand this process in mathematical terms without going too deep into the mathematics. In the forward pass, we are calculating the probability of output $$\textbf{h}^{(1)}$$ given the input  $$\textbf{v}^{(0)}$$ and the weights $$W$$ denoted by:

$$ p(\textbf{h}^{(1)} \mid \textbf{v}^{(0)};W)$$

and in the backward pass while reconstructing the input, we are calculating the probability of output $$\textbf{v}^{(1)}$$ given the input $$\textbf{h}^{(1)}$$ and the weights $$W$$ denoted by:

$$p(\textbf{v}^{(1)} \mid \textbf{h}^{(1)};W)$$

The weights used in both the forward and the backward pass are the same. Together, these two conditional probabilities lead us to the joint distribution of inputs and the activations:

$$p(\textbf{v}, \textbf{h})$$

Reconstruction is different from regression or classification in that it estimates the probability distribution of the original input instead of associating a continuous/discrete value to an input example. This means it is trying to guess multiple values at the same time. This is known as generative learning as opposed to discriminative learning that happens in a classification problem (mapping input to labels).

Let us try to see how the algorithm reduces loss or simply put, how it reduces the error at each step. Assume that we have two normal distributions, one from the input data (denoted by $$p(x)$$) and one from the reconstructed input approximation (denoted by $$q(x)$$). The difference between these two distributions is our error in the graphical sense and our goal is to minimize it, i.e., bring the graphs as close as possible. This idea is represented by a term called the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). KL-divergence measures the non-overlapping areas under the two graphs and the RBM's optimization algorithm tries to minimize this difference by changing the weights so that the reconstruction closely resembles the input. The graphs on the right hand side show the integration of the difference in the areas of the curves on the left.

<img src = "https://upload.wikimedia.org/wikipedia/en/a/a8/KL-Gauss-Example.png">

This gives us an intuition about our error term. Now, to see how actually this is done for RBMs, we will have to dive into how the loss is being computed.  All common training algorithms for RBMs approximate the log-likelihood gradient given some data and perform gradient ascent on these approximations.

#### Contrastive Divergence

Boltzmann Machines (and RBMs) are Energy based models and a joint configuration, $$(\textbf{v}, \textbf{h})$$ of the visible and hidden units has an energy given by:

$$ \displaystyle E(\textbf{v}, \textbf{h}) = − \sum_{i∈visible}
a_i v_i − \sum_{j∈hidden} b_jh_j − \sum_{i,j} v_ih_jw_{ij}$$

where $$v_i$$, $$h_j$$ are the binary states of visible unit $$i$$ and hidden unit $$j$$, $$a_i$$, $$b_j$$ are their biases and $$w_{ij}$$ is the weight between them.

The probability that the network assigns to a visible vector, $$v$$, is given by summing over all possible hidden vectors:

$$\displaystyle p(\textbf{v}) = \frac{1}{Z} \sum_{\textbf{h}} e^{-E(\textbf{v},\textbf{h})}$$

$$Z$$ here is the partition function and is given by summing over all possible pairs of visible and hidden vectors:

$$\displaystyle Z = \sum_{\textbf{v}, \textbf{h}} e^{-E(\textbf{v},\textbf{h})}$$

This gives us:

$$ p(\textbf{v}) = \frac{\displaystyle \sum_{\textbf{h}} e^{-E(\textbf{v},\textbf{h})}}{\displaystyle \sum_{\textbf{v}, \textbf{h}} e^{-E(\textbf{v},\textbf{h})}}$$

The log-likelihood gradient or the derivative of the log probability of a training vector with respect to a weight is surprisingly simple:

$$\frac{\partial log p(\textbf{v})}{\partial w_{ij}} = \langle v_i h_j \rangle_{data} - \langle v_i h_j \rangle_{model}$$

where the angle brackets are used to denote expectations under the distribution specified by the subscript that follows. This leads to a very simple learning rule for performing stochastic steepest ascent in the log probability of the training data:

$$\Delta w_{ij} = \alpha(\langle v_i h_j \rangle_{data} - \langle v_i h_j \rangle_{model})$$

where $$\alpha$$ is a learning rate. For more information on what the above equations mean or how they are derived, refer to the [Guide on training RBM by Geoffrey Hinton](https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf). The important thing to note here is that because there are no direct connections between hidden units in an RBM, it is very easy to get an unbiased sample of $$\langle v_i h_j \rangle_{data}$$. Getting an unbiased sample of $$\langle v_i h_j \rangle_{model}$$, however, is much more difficult. This is because it would require us to run a Markov chain until the stationary distribution is reached (which means the energy of the distribution is minimized - equilibrium!) to approximate the second term. So instead of doing that, we perform [Gibbs Sampling](https://en.wikipedia.org/wiki/Gibbs_sampling) from the distribution. It is a Markov chain Monte Carlo (MCMC) algorithm for obtaining a sequence of observations which are approximated from a specified multivariate probability distribution, when direct sampling is difficult (like in our case). The Gibbs chain is initialized with a training example $$\textbf{v}^{(0)}$$ of the training set and yields the sample $$\textbf{v}^{(k)}$$ after $$k$$ steps. Each step $$t$$ consists of sampling $$\textbf{h}^{(t)}$$ from $$p(\textbf{h} \mid \textbf{v}^{(t)})$$ and sampling $$\textbf{v}^{(t+1)}$$ from $$p(\textbf{v} \mid \textbf{h}^{(t)})$$ subsequently (the value $$k = 1$$ surprisingly works quite well). The learning rule now becomes:

$$\Delta w_{ij} = \alpha(\langle v_i h_j \rangle_{data} - \langle v_i h_j \rangle_{recon})$$

The learning works well even though it is only crudely approximating the gradient of the log probability of the training data. The learning rule is much more closely approximating the gradient of another objective function called the **Contrastive Divergence** which is the difference between two Kullback-Liebler divergences.

When we apply this, we get:

$$ \textbf{CD}_{k}(W, \textbf{v}^{(0)}) = -\displaystyle \sum_{\textbf{h}} p(\textbf{h} \mid \textbf{v}_k)\frac{\partial  E(\textbf{v}_k, \textbf{h})}{\partial W} + \displaystyle \sum_{\textbf{h}} p(\textbf{h} \mid \textbf{v}_k)\frac{\partial  E(\textbf{v}_k, \textbf{h})}{\partial W}$$

where the second term is obtained after each $$k$$ steps of Gibbs Sampling. Here is the pseudo code for the CD algorithm:

<center><img src = "https://cdn-images-1.medium.com/max/1600/1*cPYfytQ30HP-2rpe_NKqmg.png"> </center>

### Conclusion

What we discussed in this post was a simple Restricted Boltzmann Machine architecture. There are many variations and improvements on RBMs and the algorithms used for their training and optimization (that I will hopefully cover in the future posts). I hope this helped you understand and get an idea about this awesome generative algorithm. In the next post, we will apply RBMs to build a recommendation system for books!

#### Sources:
* [Wikipedia - Restricted Boltzmann Machine](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)
* [Wikipedia - Boltzmann Machine](https://en.wikipedia.org/wiki/Boltzmann_machine)
* [Guide on training RBM by Geoffrey Hinton](https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf)
* [Skymind - RBM](https://skymind.ai/wiki/restricted-boltzmann-machine)
* [https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf](https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf)
* [Artem Oppermann's Medium post on understanding and training RBMs]( https://towardsdatascience.com/deep-learning-meets-physics-restricted-boltzmann-machines-part-ii-4b159dce1ffb)
* [Medium post on Boltzmann Machines by Sunindu Data](https://medium.com/@neuralnets/boltzmann-machines-transformation-of-unsupervised-deep-learning-part-1-42659a74f530)
