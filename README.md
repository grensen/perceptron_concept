# The Perceptron Concept (Or All Is A Perceptron Concept)
# An efficent neural network concept for the C-Family
# Supported language: C#, CPP, JAVA

The u[] array describes the network, I take u[] because it got a good position on the keyboard, the reference to learn the concept will be the { 3, 5, 5, 5, 2 } deep neural network. 

The demo NN_001 for each language uses a { 784, 25, 25, 25, 10 } NN and as inputs the MNIST data comes in.
To make everything comparable, the network works with pseudo random values. 
Rectified linear units (ReLU) are used as an activation function,
and the output neurons are activated with softmax and cross entropy loss.
The quick demo also supports batch, mini-batch and online-trainig / stochastic gradient descent (SGD). 

The weight initialization is a pseudo random function with known numbers. 
"weightInitRange" set the range of the weights, 0.314 seems a nice start value, 
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

Let's focus on the green index on the Image, input, hidden and output neurons are in only one array, the neuron[], so the network uses one index for all neurons. And here we have to talk about the j index, thats not the whole index, because j is the activation index after the inputs and starts on the first neuron with index 3 on hidden layer 1. So lets talk about the gradient array in red, which starts with index 0 on the green index position 3, these is the index for the gradient[], but it's also the index for the bias[], or the netinput[] we dont need in this concept. 

To realize the idea we need three loops, the outer i loop for the layer, then the middel k loop for every neuron we need to activate for our output (right sided k) operations, and the inner n loop for the input (left sided n) neurons and every weight, to calc the products we add to the net variable intead the netinput[] array after we leave the n loop.

Ok ok, step by step, because the steps are the key. Here we watch the NN u[] = {3,5,5,5,2} again, the first step goes from 0 to 3 and represents the input neurons we add to k, the k loop is u[i + 1], because we start with i = 0, and add to the next layer, so its i + 1 with 5 neurons in this case.

With MNIST the input would be 784 for each pixel, but the reference works only with 3 inputs, so we just imagine a image with three pixels for the reference example.
Think about, we add the left side n first with the 3 input neurons with index 0, 1, 2 to the right side on our first output neuron[3] k, till we activate the last neuron on the layer h1 from 3 to 7 in the k loop, then if i = 1 h1 represent our inputs (left sided n), and h2 our outputs (right sided k) till we reach the final output layer, here we do not activate the output neurons and just let them pass.

Keep in  mind, n goes to k!

Some termenology alert, a layer means normaly the connection between the input and their outputs (input * output = layer), but it's also common to name the layer as input, hidden or output. Correctly I would name it connection layer, which need 2 parts.
So in the i loop a layer means (u layer: i0 = (3 * 5) i1 = (5 * 5) i2 = (5 * 5) i3 = (5 * 2)) which results in 4 layer, 20 neurons and 75 weights.

In pseudo it could look like this for FF:
```
dnn = u.len - 1 // = 4 on the reference

inputs = u[0] // = 3

output = u[dnn] // = 2

nns = sumUp(u) // size of the neurons is the sum of u[] = 3 + 5 + 5 + 5 + 2 = 20*

wnn = sumProducts(u) // size of weights = sum of the products of u = 3 * 5 + 5 * 5 + 5 * 5 + 5 * 2 = 75*
```
*the gradient, bias and netinput index is just the (nns-inputs), as example I added a bias for FF*

```
for (int j = inputs, w = 0, t = 0; i < dnn; i++, t += u[i - 1], w += u[i] * u[i - 1])  // layer

   for (int k = 0; k < u[i+1]; j++, k++) // neuron

      net = bias[j-inputs]

      for (int n = t; n < t + u[i + 1]; n++, m += u[i + 1]) // weight

         net += neuron[n] * weight[m]

      if(net more then 0 (relu) or i is outputlayer (output for softmax))

       neuron[j] = net

      else

       neuron[j] = 0
```

The output layer will be activated with softmax. One special treatment, but only for a neural networks, is only one calculation for the max value of the output layer for the cross entropy each run, because the other calculations multiplys with 0, so I let them out and no loop is needed.

The hardest part is the backpropagation, here we go just backwards.

```
  for (int i = dnn, j = nns - 1, ls = output, wd = wnn - 1, wg = wd, us = nns - output - 1, gs = nns - inputs - 1;
      i != 0; i--, wd -= u[i + 1] * u[i + 0], us -= u[i], gs -= u[i + 1])
     /*
       lets describe the new ones:
       ls = loss iterator with the size of the output neurons, thats what we need
       wd = weight delta starts on the last index array position
       wg = weight gradient = wd
       us = neuron steps, we start on the last neuron of the last hidden-layer, because we need the product from neuron[n] * gra 
       gs = gradient steps, here we start without the inputs, because on the FF we start activation on the first hidden neuron 
     */
      for (int k = 0; k != u[i]; k++, j--)
      {
          float gra = 0;
         //--- first check if output or hidden layer
          if (i == dnn) // calc gradient with (t - o)
              gra = target[--ls] - neuron[j];
          else if(neuron[j] > 0) // calc the gradient for hidden with respect of the derivative for ReLU
              for (int n = gs + u[i + 1]; n > gs; n--, wg--)
                  gra += weight[wg] * gradient[n];
          else wg -= u[i + 1]; // substract the skipped iterations           
          for (int n = us, w = wd - k; n > us - u[i - 1]; w -= u[i], n--)
              delta[w] += gra * neuron[n]; // calcs the deltas from the currenct gradient
          gradient[j - inputs] = gra;
      }
```

First we check if we are on the output layer and than we calc the gradient with (target - output) and update the deltas for the weights of this gradient. Thats the strongest "just in time" component in this concept, because we dont need more code and resources to handle this operation. If we done with the output, we calc the gradient for the hidden nerurons, and this process ist just the FF, but backwards.
The loops do not look very attractive, thats true. Instead of the long loops we could use arrays for the steps, that looks sexier, but for the understanding it seems better to show the calc on their place.

Before we finish we need one more step, the weight update. This is easy, the simple way is one loop.
```
for (int m = 0; m < wnn; m++)
   // update weights
```

Another nice way is to use the FF or BP as a dummy of the 3 loops, remove the calculation and replace them in the n loop with the weight update. With this dummy its possible to update the layer with seperate learning rates, whats seems a nice finetune at the end, or do some other stuff over the neurons.   

```
for (int j = inputs, w = 0, t = 0; i < dnn; i++, t += u[i - 1], w += u[i] * u[i - 1]) 
   for (int k = 0; k < u[i+1]; j++, k++)
      for (int n = t; n < t + u[i+1]; n++, m += u[i + 1]) 
         // update weights
```

Here is a visualisation of the whole process:

[Perceptron Concept Visualisation](https://www.youtube.com/watch?v=jZgb3-W7BpQ)

Hope that helps.


At the end, the understanding of the reference is the key to work with massiv huge networks. 

With a good understanding of the indices, its really easy to implement helpful techniques like dropout with nice efficency.

![numberWeights](https://user-images.githubusercontent.com/53048236/61751317-3d88d780-ada8-11e9-9e50-9e1a95055e4d.png)


