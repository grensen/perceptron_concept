# The Perceptron Concept (Or All Is A Perceptron Concept)
# A efficent neural network concept for the c-family
# Supported language: C#, CPP, JAVA

The u[] array describes the network, I take u[] because it got a good position on the keyboard, the reference to learn the concept will be the { 3, 5, 5, 5, 2 } deep neural network. 

The demo NN_001 for each language uses a { 784, 25, 25, 25, 10 } network with pseudo random values, 
that makes everything comparable.
Rectified linear units (ReLU) are used as an activation function,
and the output neurons activated with softmax with cross entropy loss.
The quick demo also supports batch, mini-batch and stochastic gradient descent (SGD). 

A special feature ist the bounce restriction method.
The method in the trivial form is simpel, if a delta^2 breach the limit, the network waits till the delta is cooling down before the NN update the weight, but the weight itself continues to work in the network all the time. This makes the NN much more robust.

With the perceptron concept its possible to use only one array for every value. The data treated just in time.






![alt text](https://user-images.githubusercontent.com/53048236/61723001-99813b00-ad6b-11e9-81ea-aaa683a98b4f.png)
