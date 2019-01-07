---
layout: post
title: Image Compression: Seam Carving and Clustering
author: Aditya Sharma
date: Jan 05, 2019
---

## Motivation

We often want to resize images (say, for different screen sizes/resolutions). A standard solution to make an image smaller would be to just discard some of the pixels. For example, if we want to make the image exactly half its original width and height, we can cut up the images into $2\times 2$ squares and just keep the upper-left pixel in each square. But what if you want to reduce the size of the image by 30%? Or, what if you want to change the aspect ratio of the image without distorting the image contents? What if we want to compress so that the image takes up only a specific number of bits to store the image? Now things get more complicated. Let us explore some techniques which help us do the above tasks.

In this post I will use three algorithms to compress images:
1. Using Dynamic Programming
2. Using Integer Linear Programming
3. Using K-Means Clustering

This would be a short post with emphasis only on how the above techniques can be used for image compression followed by the Python code snippets for the same. So let's get started!


## Seam Carving

Here, we will explore some techniques for image compression first of which is called [Seam Carving](https://en.wikipedia.org/wiki/Seam_carving). This concept is based on the [2007 paper by Shai Avidan and Ariel Shamir "Seam Carving for Content-Aware Image Resizing"](https://perso.crans.org/frenoy/matlab2012/seamcarving.pdf). As the title of the paper suggests, this algorithm is about "content-aware" image resizing, which means that the pixels we add/remove are chosen based on the content itself. With seam carving, we can resize to any size, not just nice integer fractions like double or half. Seam carving can also be used to make an image larger.

A seam is an optimal 8-connected path of pixels (pixels that are connected to each other sharing an edge or a corner) on a single image from top to bottom, or left to right, where optimality is defined by an image energy function. A seam will contain one pixel per row if we carve out a vertical seam or one pixel per column if we carve out a horizontal seam. Look at the picture below for an example of a vertical and a horizontal seam.

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/ic/seam_carve.png?raw=True"></center>
<br>

In seam carving, we remove one seam at a time until the desired compression is achieved. The energy function in seam carving defines the importance of pixels. This ensures that while removing seams, we remove most of the low energy pixels and while enlarging the images, we add pixels such that there is a balance between the original pixels and the ones artificially inserted. as mentioned in the paper, seam carving can support several types of energy functions such as gradient magnitude, entropy, visual saliency, eye-gaze movement, and more. The basic idea about energy functions is that when pixel intensities vary more rapidly, this indicates a more important part of the image, and thus higher energy. Let us now move on to the actual implementation using Dynamic Programming.

### Seam Carving using Dynamic Programming

As is obvious from the term Dynamic Programming, this process can be done using a recursive approach but it is too slow to be useful for real images. Using DP, we bring this time complexity down to $O(M*N)$. Assuming we already have the energy function and a function to remove the seam found using DP, we just need to look at the code for getting the seam. Also, if we write a function to find a vertical seam, we can simply rotate the image by 90 degrees to get a horizontal seam by applying the same function.

Let us now look at the function `find_vertical_seam` using a recursive implementation first.

```python
def find_vertical_seam(energy):
    costs = dict()
    for i in range(energy.shape[1]):
        best_seam, best_cost = fvs(energy, [i], 0.0)
        costs[tuple(best_seam)] = best_cost
    return min(costs, key=costs.get) # the best out of the M best seams

def fvs(energy, seam, cost):
    row = len(seam)-1
    col = seam[-1]

    # if out of bounds on one of the two sides, return infinite energy
    if col < 0 or col >= energy.shape[1]:
        return seam, np.inf

    cost += energy[row,col]

    # regular base case: reached bottom of image
    if len(seam) == energy.shape[0]:
        return seam, cost

    return min((fvs(energy, seam+[col], cost),
                fvs(energy, seam+[col+1],cost),
                fvs(energy, seam+[col-1],cost)),key=lambda x:x[1])
```

The time complexity of this recursive approach is O(M * $3^N$) where M and N are the width and the height of the original image respectively. Now let us use DP to bring this time complexity down.

```python
def find_vertical_seam(energy):
    """
    Function to get the vertical seam in a picture

    The function takes an energy array for a picture and
    finds the vertical seam with the minimum energy path
    from the top to the bottom of the image.
    """
    row = energy.shape[0]
    col = energy.shape[1]
    dp_energy = []
    dp_energy.append(energy[0].tolist())

    for i in range(1, row):
        temp = []
        for j in range(col):
            if j == 0:
                temp.append(energy[i][j] + min(dp_energy[i-1][j], dp_energy[i-1][j+1]))
            elif j == col - 1:
                temp.append(energy[i][j] + min(dp_energy[i-1][j], dp_energy[i-1][j-1]))
            else:
                temp.append(energy[i][j] + min(dp_energy[i-1][j-1], dp_energy[i-1][j], dp_energy[i-1][j+1]))
        dp_energy.append(temp)

    # backtracking to get the seam
    cost = [0] * row
    cost[row-1] = np.argmin(dp_energy[row-1])
    for i in range(row-2, -1, -1):
        j = cost[i+1]
        if j == 0:
            cost[i] = np.argmin(dp_energy[i][j:j+2]) + j
        elif j == col - 1:
            cost[i] = np.argmin(dp_energy[i][j-1:j+1]) + j-1
        else:
            cost[i] = np.argmin(dp_energy[i][j-1:j+2]) + j-1
    return cost
```

Here is a small example image with one seam carved out using both the recursive and the DP approaches.

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/ic/ex_sc.png?raw=True" width = "350"></center>
<br>

Here is another example with an image of a Mandrill (a monkey breed) carved out using DP. The image width and height are halved and we find some information being lost in the process.

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/ic/dp_mand.png?raw=True" width = "350"></center>
<br>

You can try out different width and height compressions using the [actual code](https://github.com/adityashrm21/image-compression-techniques/blob/master/image-compression.ipynb) from my [github repository](https://github.com/adityashrm21/image-compression-techniques).


### Image Compression using Integer Linear Programming

An integer programming problem is a mathematical optimization or feasibility program in which some or all of the variables are restricted to be integers. In many settings the term refers to integer [linear programming](https://en.wikipedia.org/wiki/Linear_programming) (ILP), in which the objective function and the constraints (other than the integer constraints) are linear.

Integer programming is [NP-complete](https://en.wikipedia.org/wiki/NP-completeness). In particular, the special case of 0-1 integer linear programming, in which unknowns are binary, and only the restrictions must be satisfied, is one of [Karp's 21 NP-complete problems](https://en.wikipedia.org/wiki/Karp%27s_21_NP-complete_problems).

If some decision variables are not discrete the problem is known as a mixed-integer programming problem.

This approach in image compression will take the advantage of the linearity of the problem. To pick a seam, we define $NM$ binary variables $x_{ij}$, where $i$ represents the row and $j$ represents the column, so that we have one variable for each pixel.

A vertical seam can only have one pixel per row. If pixel $(i,j)$ is in the seam and $i \lt N$, then one of $(i+1,j-1)$, $(i+1,j)$, or $(i+1,j+1)$ must be in the seam. We can convert these into math using our $x_{ij}$ variables. The first one becomes $\sum_{j=0}^{N-1}x_{ij}=1$ for all $i$. The second constraint comes from the connectivity of the seam: $x_{ij} - x_{i+1j-1} - x_{i+1j} - x_{i+1j+1} \leq 0$. We use the `pulp` package from Python for this implementation. To read more about the package and the documentation, go to [this link](https://pythonhosted.org/PuLP/).

```python
import pulp

def find_vertical_seam(energy):
    N, M = energy.shape

    # initialize the optimization problem, give it a name
    prob = pulp.LpProblem("Seam carving", pulp.LpMinimize)

    # create the x_ij variables
    x = pulp.LpVariable.dicts("pixels",(list(range(N)),list(range(M))),0,1,pulp.LpInteger)

    # The objective function is being built here. The objective is the sum of energies in the seam.
    objective_terms = list()
    for i in range(N):
        for j in range(M):
            objective_terms.append(energy[i][j] * x[i][j])
    prob += pulp.lpSum(objective_terms) # adds up all the terms in the list

    # Constraint #1: one pixel per row
    for i in range(N):
        prob += pulp.lpSum(x[i][j] for j in range(M)) == 1

    # Constraint #2: connectivity of seam
    for i in range(N-1):
        for j in range(M): # below: this says: x(i,j) - x(i+1,j-1) - x(i+1,j) - x(i+1,j+1) <= 0
            prob += pulp.lpSum([x[i][j]]+[-x[i+1][k] for k in range(max(0,j-1),min(M,j+2))]) <= 0

    # Solve the problem
    prob.solve()

    # Build up the seam by collecting the pixels in the optimal seam
    # Note: you can access the values (0 or 1) of the variables with pulp.value(x[i][j])

    seam = []
    for i in range(N):
        for j in range(M):
            if pulp.value(x[i][j]) == 1.0:
                seam.append(j)        
    return seam
```
Here is an example with a random image of width and height 31.

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/ic/ilp.png?raw=True" width = "350"></center>
<br>

The same example was run using DP and it was ~2000 times faster than this ILP implementation. You can try it out yourself using the code from [the repository](https://github.com/adityashrm21/image-compression-techniques).

Now, we will try compressing images using an Unsupervised Learning algorithm: K-Means Clustering. How this is accomplished is pretty straightforward. We select a suitable number of clusters of pixels in an image as prototypes and then use the prototypes selected instead of the cluster points in the image. Sounds pretty cool right? It is indeed!

Just show me the code!

```python
def quantize(img, b):
    """
    Quantizes an image into 2^b clusters

    Parameters
    ----------
    img : a (H,W,3) numpy array
    b   : an integer

    Returns
    -------
    quantized_img : a (H,W) numpy array containing cluster indices
    colours       : a (2^b, 3) numpy array, each row is a colour
    """

    H, W, _ = img.shape
    model = cluster.KMeans(n_clusters=2**b)

    img = img.reshape(298*298, 3)
    model.fit(img)
    labels = np.array(model.labels_)
    quantized_img = labels.reshape(298, 298)

    return quantized_img, model.cluster_centers_.astype('uint8')

def dequantize(quantized_img, colours):
    H, W = quantized_img.shape
    img = np.zeros((H,W,3), dtype='uint8')

    for i in range(H):
        for j in range(W):
            for k in range(3):
                img[i, j, k] = colours[quantized_img[i, j], k]

    return img
```

We implemented image quantization using the quantize and dequantize functions below. The quantize function takes in an image and, using the pixels as examples and the 3 colour channels as features, runs $k$-means clustering on the data with $2^b$ clusters for some hyperparameter $b$. The code stores the cluster means and returns the cluster assignments. The dequantize function returns a version of the image (the same size as the original) where each pixel's original colour is replaced with the nearest prototype colour.

To understand why this is compression, consider the original image space. Say the image can take on the values $0,1,\ldots,254,255$ in each colour channel. Since $2^8=256$ this means we need 8 bits to represent each colour channel, for a total of 24 bits per pixel. Using our method, we are restricting each pixel to only take on one of $2^b$ colour values. In other words, we are compressing each pixel from a 24-bit colour representation to a $b$-bit colour representation by picking the $2^b$ prototype colours that are "most representative" given the content of the image. The above implementation uses `KMeans` implementation from the `scikit-learn` library in Python.

Here are some sample compression images for different values of the hyperparameter `b`:

<center><img src = "https://github.com/adityashrm21/adityashrm21.github.io/blob/master/_posts/imgs/ic/km_mand.png?raw=True" width = "600"></center>
<br>

You can find the jupyter notebook with the full code [here](https://github.com/adityashrm21/image-compression-techniques/blob/master/image-compression_kmeans.ipynb). We clearly see how increasing `b` improves the quality of the image and I found that at `b = 6`, the compressed image was almost indistinguishable from the original image.

## Conclusion

I found these techniques of image compression pretty slick! While studying Dynamic Programming, ILP or K-Means clustering, I never imagined that these algorithms could be used in such a creative way. I hope you liked these approaches too. In one of the upcoming posts, I will do the same task using Deep Learning. One approach to do so using DL can be in [this paper from Google](https://arxiv.org/abs/1608.05148). They use RNNs to compress the images which is pretty cool once again! I am looking forward to implementing that, so see you soon in the next post!
