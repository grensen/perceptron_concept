# The Perceptron Concept (Or All Is A Perceptron Concept)
# An efficent neural network concept for the C-Family
# Supported language: C#, CPP, JAVA

The u[] array describes the network, I take u[] because it got a good position on the keyboard, the reference to learn the concept will be the { 3, 5, 5, 5, 2 } deep neural network. 

The demo NN_001 for each language uses a { 784, 25, 25, 25, 10 } NN and as inputs the MNIST data comes in.
To make everything comparable, the network works with pseudo random values. 
Rectified linear units (ReLU) are used as an activation function,
and the output neurons are activated with softmax and cross entropy loss.
The quick demo also supports batch, mini-batch and stochastic gradient descent (SGD). 

The weight initialization is a pseudo random function with known numbers. 
"weightInitRange" set the range of the weights, 0.33 seems a nice start value, 
so the weight init works only with one hyperparameter, thus.

A special feature ist the bounce restriction method.
The method in the trivial form is simpel, if a delta^2 breach the limit, the network waits till the delta is cooling down before the NN updates the weight again, but the weight itself continues to work in the network all the time. This makes the NN much more robust.

With the perceptron concept its possible to use only one array for every neuron. Every data will be updated just in time.
The biggest step to understand the concept in addition to the understanding of how a perceptron works, 
is to understand why the j index need to be initialized with the size of the input neurons.

Feed Forward (FF) and Backpropagation (BP) treat the current neuron in FF and the gradient in the BP excatly in the same way, forward and backward. This means FF needs the products of their neurons times their weights and optionally the bias, then we got the netinput and the neuron can be activated.
The same thing with the gradient, here the step is a little bit trickier, because the gradient calculation treats the output neurons with (target - output) in a different way as the hidden neurons, where the gradient is the sum of the products from the gradients on the layer before times their own weights.

On the picture below you can see the identical results in C#, CPP and JAVA. JAVA was tested with Eclipse, C#  and CPP with Visual Studio 2017.

![alt text](https://user-images.githubusercontent.com/53048236/61723001-99813b00-ad6b-11e9-81ea-aaa683a98b4f.png)

The concept consists of a heart that is the same in every language. The other part is language specific.
More specifically, the whole training code, except for the function of the filestream, is the same in all languages.
Also arrays, math functions, print functions and some include stuff need special attention for every language, but that was it.
The complicated algorithms can be easily ported.

Requirement for the demo code in every language is the unzipped MNIST dataset, I would prefer 7zip for this work, the default path for the dataset is "C:\mnist\", one language was bleating, so I was taking a folder. In the demo the trainingset with their label are activ. 
That includes a total of 60,000 images and their labels. This results in a maximum number of 60,000 training examples that must not be exceeded.


And so I've explained the concept myself, ignore the math, here I was using (Output - Target) which is just as possible.

![WP_20190423_00_35_17_Pro](https://user-images.githubusercontent.com/53048236/61755635-ca3b9180-adb8-11e9-99a6-adfce47950a5.jpg)

All neurons, input, hidden and output neurons are in the neuron[] array, the green part shows the FF process, the red part shows the BP and the weight update in gold shows which part from FF and BP it needs.

Ok ok, step by step, because the steps are the key. With MNIST the input would be 784 for each pixel, but the reference works only with 3 inputs, so we just imagine a picture with three pixels for the reference example.

All the pixel need to connect to the first neuron on the next layer, this process repeats until the last neuron on the current layer is activated. The algorithm repeats this till the output neurons, thats a special case because we dont activate the output neurons with Relu! So we just let them pass on the outputlayer with neuron 18 and 19.



![numberWeights](https://user-images.githubusercontent.com/53048236/61751317-3d88d780-ada8-11e9-9e50-9e1a95055e4d.png)
