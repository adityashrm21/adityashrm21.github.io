---
layout: post
title: Why Rectified Linear Units (ReLUs) actually work?
author: Aditya Sharma
date: Feb 12, 2019
---

## Introduction

**NOTE**: This post assumes a basic knowledge of artificial neural networks.

If you have a basic idea of how neural networks work, you know that they can model really complex non-linear functions and this ability is possessed by them through the use of non-linear activation functions such as the sigmoid function or the hyperbolic-tangent function. For a detailed list of activation functions used in neural networks, visit [this Wikipedia page](https://en.wikipedia.org/wiki/Activation_function). In this post, we will talk about why we need activation functions and how do ReLUs perform this job even when they themselves are linear in structure.

## Why do we need Activation Functions?

The activation functions that you saw in the list above on Wikipedia help in deciding whether a neuron will fire or not after all the weighted matrix multiplication and the addition of a bias.

$$Y = W^TX + b$$

**BUT HOW?**
Let's see!

The above equation is the linear part of what happens in a neural network. If we just make a neural network with equations like the above, we will keep getting linear combinations of equations and the function obtained would just be a polynomial of `degree 1`. So a non-linear activation is what is needed in order to convert the output $$Y$$ into a value that can be interpreted as the output of a neuron in the network just like how neurons get activated in the brain. This helps model non-linearities in the input and create complex non-linear functions which are simply not possible to be modeled only using polynomial functions. Without the activation, the output value $$Y$$ is unrestricted and can range from $$-\infty$$ to $$+\infty$$. The activation functions help us restrict the output value of a neuron so that it can be decided whether it will get activated (will fire) or not. Once this non-linearity is added, the result is then sent as an input to the next layer (if there is any) in the network.

Now you must be thinking "I know all of this! What's the point of this post?" which brings us to the main section of this post in which we discuss some properties of Rectified Linear Units and contrast them with other activations such as the sigmoid and the hyperbolic tangent function.

## Rectified Linear Units

Rectified Linear Units (ReLUs) are a relatively newer type of activation functions used in neural networks and they have a simple mathematical form $$g(z) = max\{0, z\}$$. Graphically, they look like the plot below:

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/relu/relu.png?raw=True" width = "500"></center>

It was not quite long ago (2011) when it was demonstrated that ReLUs were better for the training of deep neural networks when compared to the traditional smoother sigmoid or tangent activation functions. In general, it is difficult to predict in advance which activation function would work best in a given scenario and hence, trial and error is involved to a certain degree to figure out the best activation function. This is done by intuiting that a certain activation function would work well, and then testing the performance of the network using that function and evaluating the performance on a validation set.

### How do ReLUs model non-linearity?
Now looking at the linear structure of ReLUs, it is not very intuitive to understand how they help us add a non-linearity. To understand this, think of a non-linear function being made up of small segments of linear functions. To be more simplistic, imagine a non-linear curve in 2D. Break the curve down into small pieces which can be easily approximated by lines. Each of these small line segments can be now approximated using lines parametrized by the weights of the network with ReLU as the activation function. If this was still not so clear to imagine, let us see a working example on a synthetic/toy dataset.

Let us first generate a synthetic dataset to be modeled and plot it:

<code data-gist-id="6c2d1050f16ecc74cc41833c323292b6" data-gist-line="8-12"></code>

The data looks something like this:

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/relu/toy_data.png?raw=True" width = "500"></center>

Now, we will build a simple 1 hidden layer network on this data using ReLU activation and see what we get:

<code data-gist-id="6c2d1050f16ecc74cc41833c323292b6" data-gist-line="19-32"></code>

After running the code above we obtain a model of the data which very well captures its non-linearity. After plotting the same above the data, we get something like this:

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/relu/model_relu.png?raw=True" width = "500"></center>

We can see how small straight lines are joined to model the non-linear curve made by the data.

Here is the complete code for you:

{% gist 6c2d1050f16ecc74cc41833c323292b6 %}

## More on ReLUs

Now that you have seen and understood how this linear looking activation function can model non-linearities in the data, let us talk a bit more about them!

Though ReLUs seem to work quite well ([and in fact better than standard activation functions like sigmoid or tanh](http://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)), there are some drawbacks in their use:
 - The range of these activation functions is unrestricted [0, $$\infty$$). Though this is good and helps in addressing the problem of vanishing gradients which occurs with restricted activation functions like `sigmoid` or `tanh` (very slow learning for large values of x as gradient values are small), this becomes a problem as they tend to blow up the activation because there is no mechanism to constrain the output (activation).
 - ReLUs simply make all the non-negative activations zero. Though sparsity can be good for learning, it can cause problems when too many activations are being zeroed out. This prohibits learning and therefore, is a source of problems. This is also known as the Dying ReLU problem. A ReLU neuron is "dead" if it is stuck in the negative side and always outputs 0.

To address the above problems with ReLU, people have come up with its variants that try to mitigate these drawbacks:
 - Leaky ReLU: Instead of the activation being zero for negative input values, it has a very small positive slope for negative values.
 - PReLU (parametric ReLU): This function allows to control the slope of the line in the Leaky ReLU.

<center><img src = "https://i.stack.imgur.com/1BX7l.png" width = "700"></center>

There are many other variants and improvements over ReLU which you can learn more about at these places:

- [A practical guide to ReLU by Danqing Liu](https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7)
- [Searching for Activation Functions, Ramachandran et al. (Google)](https://arxiv.org/pdf/1710.05941.pdf)
- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUS), Clevert et al.](https://arxiv.org/pdf/1511.07289.pdf)

### Conclusion

I hope this post gave you an idea and a better understanding of Rectified Linear Units and activation functions in general. I have tried to cover as much information as possible without making the post too long. If I missed something or if you find any mistakes, let me know in the comments and I will be happy to address them!
See you in the next post!
