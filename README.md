# The Perceptron Concept (Or All Is A Perceptron Concept)
# An efficent neural network concept for the C-Family
# Supported language: C#, CPP, JAVA

The u[] array describes the network, I take u[] because it got a good position on the keyboard, the reference to learn the concept will be the { 3, 5, 5, 5, 2 } deep neural network. 

The demo NN_001 for each language uses a { 784, 25, 25, 25, 10 } NN and as inputs the MNIST data comes in, 
to make everything comparable, the network works with pseudo random values. 
Rectified linear units (ReLU) are used as an activation function,
and the output neurons are activated with softmax and cross entropy loss.
The quick demo also supports batch, mini-batch and stochastic gradient descent (SGD). 

The weight initialization is a pseudo random function with known numbers. 
"weightInitRange" set the range of the weights, 0.33 seems a nice start value, 
so the weight init works only with one hyperparameter, thus.

A special feature ist the bounce restriction method.
The method in the trivial form is simpel, if a delta^2 breach the limit, the network waits till the delta is cooling down before the NN updates the weight again, but the weight itself continues to work in the network all the time. This makes the NN much more robust.

With the perceptron concept its possible to use only one array for every value. The data will be updated just in time.
The biggest step to understand the concept in addition to the understanding of how a perceptron works, 
is to understand why the j index need to be initialized with the size of the input neurons.

Feed Forward (FF) and Backpropagation (BP) treat the current neuron in FF and the gradient in the BP excatly in the same way, forward and backward. This means FF needs the products of their neurons times their weights and optionally the bias, then we got the netinput and the neuron can be activated.
The same thing with the gradient, here the step is a little bit trickier, because the gradient calculation treats the output neurons with (target - output) in a different way as the hidden neurons, where the gradient is the sum of the products from the gradients on the layer in direction to the output with their weights.




![alt text](https://user-images.githubusercontent.com/53048236/61723001-99813b00-ad6b-11e9-81ea-aaa683a98b4f.png)
