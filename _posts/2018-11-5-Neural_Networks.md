---
layout: post
title: Neural Networks - Explained, Demystified and Simplified
---

Everyone who wants to learn neural networks is new to them at some point in their lives. It seems really intuitive to understand that neural networks behave just like human brain with all the convoluted connections and neurons and whatnot! But when it comes to actually understanding the math behind certain concepts, our brain fails to create new connections to understand the equations easily unless you have a hefty math background. See the irony? (Just kidding!).

Let's try to break it down into pieces and understand it step by step.

## What is an Artificial Neuron?

An artificial neuron is a mathematical function conceived as a model of biological neurons, a neural network. Artificial neurons are elementary units in an Artificial Neural Network (ANN). The artificial neuron receives one or more inputs and sums them to produce an output (or activation). Usually each input is separately weighted, and the sum is passed through a non-linear function known as an activation function (more on this coming later) or transfer function.

In a simpler form, the flow in an ANN looks like this:

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/nn/simple_nn.png?raw=true"></center>

But this doesn't look like a network, right? This diagram is just for understanding the flow of information in a neural network. We will look at better network representations later.

## What is a Neural Network?

Neural networks are the computing systems vaguely inspired by biological neurons, have connections similar to the connections in animal brain and are made up of multiple artificial neurons arranged in layers.

In case of supervised learning, we need to provide labeled examples to our neural network $$(x^{(i)}, y^{(i)})$$ as the training data. If you are not familiar with supervised learning, I would suggest you look it up but simply put, $$y^{(i)}$$ is the output of the observation when the input provided to the supervised learning algorithm is $$x^{(i)}$$ and these input-output pairs are provided to the algorithm in order for it to learn the pattern and form a relation between the input and the output variables. This helps the algorithm to predict the output on new unseen values of the input.

Neural networks have the ability to learn non-linear relationships from the data due to their special architecture and this is not possible in many of the traditional machine learning algorithms like regression. They are able to do so by fitting a combination of the parameters $$W$$ and $$b$$ (the weights and bias) to our data to produce the output. Don't worry about what weights and bias are for now as we will be looking at each one of them in detail.

### Architecture

Let's learn about neural networks by understanding them through a simple network with a single neuron as shown below. The network in the figure has a single neuron with $$n$$ inputs $$(a_1, a_2...,a_n)$$, $$n$$ weights $$(w_1, w_2...,w_n)$$ and one output (a complex term which we will break down and simplify).

<center> <img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/nn/nn.gif?raw=true"> </center>

As explained earlier in the flow diagram for an ANN, the weights are multiplied by the inputs and a bias term is added to the sum. This quantity is then passed through an activation function such as a Sigmoid function, a Rectified Linear Unit or a $$tanh$$ function (also known as the Hyperbolic Tangent Function). This gives us the output of the network which is also called the activation of the output unit. In general, each node $$i$$ in each layer $$l$$ in the neural network will have an activation from the previous layer (will be more clear later).

#### Sigmoid Function

A sigmoid function is often used as an activation function for neural networks and is defined as follows:

$$ f(x) = \frac{1}{1 + e^{-x}}$$

This is the same function which is used in logistic regression (if you remember that) and thus, our single neuron corresponds exactly to an input-output mapping defined by logistic regression. The output of this function lies in the range [0, 1].

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/nn/sigmoid.png?raw=true" width = "400"></center>

#### Hyperbolic Tangent Function

It is a function similar to a sigmoid function and simply put, it is a rescaled version of the sigmoid function. It is defined as follows:

$$ f(x) = tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

The range for this function is [-1, 1] and we should keep the range of our activation function in mind when using them.

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/nn/tangent.png?raw=true" width = "400"></center>

#### Rectified Linear Unit

In the context of ANN, a rectifier is an activation function defined as the positive part of its argument and is given by:

$$ f(x)=x^{+}=\max(0,x)$$

where $$x$$ is the input to a neuron. This activation function has been shown to enable better training of deep networks as opposed to its earlier counterparts which we discussed above. A unit employing the rectifier is also called a rectified linear unit (ReLU).

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/nn/relu.png?raw=true" width = "400"></center>

Here are some advantages of ReLU as an activation function over the traditional Sigmoid function:
-  The ReLU has a constant gradient which results in faster learning. Sigmoids, on the other hand, have diminishing gradient. The constant gradient in a ReLU results in a reduced likelihood of vanishing gradient.
- Another benefit of ReLU over sigmoids is sparsity which arises when the input to the activation function is non-positive. The more such units exist, the more sparse the representation. Sigmoids, on the other hand, always generate some non-zero value which results in dense representations which are less beneficial than sparse representations.

Dense and Sparse are not some fancy words. In our context, a sparse representation simply means that the representation of the matrices or vectors involved in the calculations have more $$0$$s while the dense ones don't (and have a lot of small values instead). They may also be described in terms of activations of particular layers or in terms of a small subset of connections in the network as opposed to all possible connections. Sparse representations are easy to store and deal with than the dense ones.

### Neural Network Formulation

Let us now talk about the math and how information if propagated through a neural network. For this task, we will consider the network shown below with 3 layers.

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/nn/nn_form.png?raw=true" width = "500"></center>

The leftmost layer (layer $$L_1$$) is called the input layer. The $$+1$$ term in the input layer corresponds to the bias or the intercept term. The middle layer (layer $$L_2$$) is called the hidden layer. This is because it is hidden and its values are not observed in the training dataset. The third layer (layer $$L_3$$) is the output layer and it has only one node (note that the output layer can have multiple nodes). We also say that our network has $$3$$ input units, $$3$$ hidden units and $$1$$ output unit (the bias terms are not counted in this).

_Now, I need you to be attentive and concentrate on this part as there are a lot of notations and it is easy to get confused and lost in between._

We will denote the number of layers in our network with $$n_l$$ (in this case, $$n_l = 3$$). The output from each layer are the activations from this layer. The activation for unit $$i$$ in layer $$l$$ would be denoted by $$a_i^{(l)}$$ (which is the output for this unit). For the first layer, the activations are the inputs themselves ($$x_1, x_2, x_3$$) and therefore, $$a_1^{(1)} = x_1$$ and so on.

The parameters weights and bias are denoted by $$W, b$$ and in our case we have $$W, b = (W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)})$$. The weight corresponding to the connection which connects unit $$j$$ in layer $$l$$ and unit $$i$$ in layer $$l+1$$ is denoted by $$W_{ij}^{(l)}$$. $$b_i^{(l)}$$ is the bias associated with unit $$i$$ in layer $$l+1$$. Take a moment here and digest this before we move on.

_It is easy to confuse the order of $$i$$ and $$j$$ in the weight matrix with the corresponding layers in the network and to confuse the bias for a unit in layer $$l$$ with the bias for layer $$l+1$$._

In simple terms, a weight is a number which tells us the the contribution of a particular unit when being used in the calculation of an input to another unit inside a neural network. If the weight for connection $$c1$$ from a unit $$x$$ to a unit $$y$$ is more than the weight for connection $$c2$$ from unit $$z$$ to unit $$y$$, it means that the contribution of unit $$x$$ is more in the input to $$y$$ when compared to the contribution of unit $$z$$ and $$x$$ will play a greater role in the activation of $$y$$ than $$z$$. This should give a basic idea of the role of weights in a network.

We will denote the number of nodes in layer $$l$$ by $$s_l$$ (without counting the bias unit). In our example, we have $$W^{(1)} \in \mathbb{R}^{3\times3}, W^{(2)} \in \mathbb{R}^{1\times3}$$. Bias units don't have inputs going into them since they always output a value of $$+1$$ (they are like the constants in equations which play the role of an intercept, displacing the curve form the origin).

Let's also talk a bit more about bias units since they are not easily understood. In the ANN terminology, the bias will help the neurons which are not able to fire when the weighted input is passed through the activation function to fire (get activated). This helps the model to become more flexible in learning over a broader range of values and allows for a better fitting of the input data. We have a bias value of $$+1$$ because our activation threshold is $$\geq0$$. Simply, the bias will have a value negative of the threshold that we choose for our neurons to get activated or fire. For example, if the threshold is $$+5$$, the bias value would be $$-5$$. For more information on what bias is, watch [this cool video](https://www.youtube.com/watch?v=HetFihsXSys).

### Feed-Forward Neural Network

Now that we have defined almost everything (just a little more coming), let us see the computation steps in the neural network:

$$a_1^{(2)} = f(W_{11}^{(1)}x_1 + W_{12}^{(1)}x_2 + W_{13}^{(1)}x_3 + b_1^{(1)}) \hspace{2cm} (1)$$

$$a_2^{(2)} = f(W_{21}^{(1)}x_1 + W_{22}^{(1)}x_2 + W_{23}^{(1)}x_3 + b_2^{(1)}) \hspace{2cm} (2)$$

$$a_3^{(2)} = f(W_{31}^{(1)}x_1 + W_{32}^{(1)}x_2 + W_{33}^{(1)}x_3 + b_3^{(1)}) \hspace{2cm} (3)$$

$$h_{W,b}(x) = f(W_{11}^{(2)}a_1 + W_{12}^{(2)}a_2 + W_{13}^{(2)}a_3 + b_1^{(2)}) \hspace{1.4cm} (4)$$

where $$h_{W,b}(x)$$ is the output (a real number) of the network.

To digest these equations, let us do some mental representation and manipulation of the weight matrix, input vector and the bias vector. Imagine the first row vector of the weight matrix as being multiplied by the column vector of input $$x$$ which gives us equation $$(1)$$. Similarly, we get equations $$(2)$$ and $$(3)$$ using the second and the third rows of the weight matrix. We now get a column vector of size $$(1 \times3)$$ which we then add to the bias column vector which has the same shape and apply the activation function to the result which gives us the activation for this layer and input for the next layer. We now multiply the second component of the weight matrix which has the size $$(1\times3)$$ with this output to get a real value. We add the second component of the bias and pass the value to the activation function. This finally gives us the activation of the output unit. I like to imagine matrix/vector calculations in my head so as to get an idea of what's actually going on instead of breaking my head in the complex mathematical notations.

Moving forward, here comes another notation for you, $$z_i^{(l)}$$ which denotes the total weighted sum of the inputs to unit $$i$$ in layer $$l$$:

$$z_i^{(2)} = \displaystyle \sum_{j = 1}^n W_{ij}^{(1)}x_j + b_i^{(1)}$$

which gives us $$a_i^{(l)} = f(z_i^{(l)})$$. This will help us in writing the above set of complex looking equations into a nicer matrix and vector format as we will see below (we will apply the function $$f(.)$$ to a vector in an element-wise fashion, i.e., $$f([z_1, z_2, z_3]) = [f(z_1), f(z_2), f(z_3)]$$).

The above set of equations for the computation steps become:

$$z^{(2)} = W^{(1)}x + b^{(1)}$$

$$a^{(2)} = f(z^{(2)})$$

$$z^{(3)} = W^{(2)}a^{(2)} + b^{(2)}$$

$$h_{W,b}(x) = a^{(3)} = f(z^{(3)})$$

I guess you can now look at the above equations and relate them to what we discussed above in the mental exercise to understand what was going on inside the network. We recall that the input $$x$$ can be written as $$a^{(1)}$$ and therefore, we can generalize to compute layer $$l+1$$'s activation by using:

$$z^{(l+1)} = W^{(l)}a^{(l)} + b^{(l)}$$

$$a^{(l+1)} = f(z^{(l+1)})$$

This is the beauty of linear algebra and matrix computations in the field of deep learning as they help us take the advantage of fat linear algebra routines to quickly perform calculations in our network. This is also where the role of GPUs comes in. You can read more about how GPUs help in faster matrix calculations [here](https://graphics.stanford.edu/papers/gpumatrixmult/gpumatrixmult.pdf).

This is a typical example of a feed-forward neural network but it is more easy to imagine a feedforward network with more than 1 hidden layers. The information will propagate in the forward direction as the activations of one layer feed into another till we reach the output. The word feedforward is used because there are no directed loops or cycles in the network (yes neural networks can have loops/cycles, more on them in later posts).

In case we have more than two units in the output layer (which as discussed earlier, is possible), we will have training examples of the form $$(x^{(i)}, y^{(i)})$$ where $$y^{(i)} \in \mathbb{R}^2$$. This sort of network is useful if there are multiple outputs that we are interested in predicting!

## Conclusion
I hope this post was helpful in explaining neural networks in simple terms along with the required mathematics! Neural networks are not difficult to understand. The fact that the math involves a lot of notations in matrix and vector forms makes it a little bit tricky to understand at one go. If you don't understand it on one go, read again and try to break down stuff with pen and paper and I am sure you will be comfortable with neural networks in no time!

#### Resources:
1. [Notes on Sparse Autoencoder by Andrew Ng](http://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf)
2. Wikipedia
