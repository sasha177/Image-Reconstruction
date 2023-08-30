# A Survey of Image Reconstruction Techniques
EECS 442: Computer Vision Final Project

Vishal Chandra (chandrav), Sasha Mittal (msasha), Abhinav Thakur (tabhinav)


### File Guide
- fbp.ipynb explores Filtered Back Projection (FBP) and fbp.py is the library of extracted code from that notebook

- art.ipynb explores the Algebraic Reconstruction Technique (ART) and depends on the functions in fbp.py as lambdas. Begins with a mathematical derivation of  algorithm from [here](https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique)

- test.ipynb contains test code for the fbp.py library

- util.py contains adapted code from [here](https://github.com/mjDelta/Filtered-Backprojection-python) that takes care of generating sinograms and padding images.

- cnn.ipynb contains all code for the Deep Learning part of this project. Dataset used is specifc EECS 445 and can be provided upon request. Each image in the dataset was transformed to a sinogram before being fed to our network for training. We did not submit the dataset as it contains a prohibitive number of images.

- /gifs folder contains animations related to the FBP and ART algorithms, which appeared in our video submission.


### Next Steps

We now have two good ways to go from a sinogram to an image with FBP and ART. Can we use a CNN as a third way? Said similarly to the math in the ART notebook, we have two ways to approximate the inverse operation of projection, can a CNN approximate it too by training on image-sinogram pairs? The process might look like taking a common dataset (like MNIST maybe), creating sinograms for every image, then training a CNN with sinograms as input and original image as expected output.

Paper implementing similar idea is in the repo [here](./neural-reconstruction.pdf). We only need the parallel-beam architecture (fig. 1). The other architectures are for more complex projection operations and in 3D.