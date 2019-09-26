# The Perceptron Concept (Or All Is A Perceptron Concept)
# An efficent neural network concept for the C-Family
# Supported languages: C#, CPP, JAVA, MQL4


In most articles, the process itself is usually briefly explained.

Here I will try to simplify everything as well as possible in order to explain the non-changeable, very complex parts as simply as possible. 

I express the whole process as A (inputs) + B (hidden connected weights) = C (outputs).

We want to make predictions, no matter what kind, we give A with any data to the NN in form of input neurons, calculate plus B and get C as output neurons. 

Our goal is a perfect prediction by correcting the value of B by computing (our wanted C) - A = (new B),
thats what we do after every new input A and take the new B to calculate the next prediction.

A + (new B) = (better C prediction). 
The C is our classification result, and can contain any number of classifiers.

With real numbers, it may be easier to understand.
We calculate 4 + (-2) = 2, but our goal for C was a prediction of 1 in this case.
So we correct the B with 1 - 4 = -3 because we tell the NN C = 1 now.
The next calculation will be with 4 + (-3) = 1, so the NN makes better predictions if the input is A = 4.

The weights stand for B and are our constants in the NN, we change B a bit to create better C predictions.
This idea of the process should help to understand the core of this topic, the implementation of a deep neural network or also called a multilayer perceptron (MLP), so lets go deeper.

## *okay, but why it named perceptron concept?*

James D. McCaffrey wrotes:

["A perceptron is code that models the behavior of a single biological neuron. A neural network can be thought of as a collection of perceptrons."](https://jamesmccaffrey.wordpress.com/2013/04/17/classification-using-perceptrons/)

Lets create the simpelest perceptron, with two inputs and one output. 
We calculate neuron1 * weight1 + neuron2 * weight2 = output1. So we can add more input neruons, 
or we calculate more output neurons neuron1 * weight3 + neuron2 * weight4 = output2.
After the layer is calculated the output neurons can act as input neurons for new output neurons if their is a next layer. So the process is nearly the same all the time. The algorithm treat every neuron as a perceptron. That is why the concept bears this name.  

The u[] array describes the neural network (NN), I take u[] because it got a good position on the keyboard, the reference to learn the concept will be the { 3, 5, 5, 5, 2 } deep neural network. 

The demo NN_001 for each language used a { 784, 25, 25, 25, 10 } NN and as inputs the MNIST data comes in.
To make everything comparable, the network works with pseudo random values. 
Rectified linear units (ReLU) are used as an activation function with a optimized implementation,
the output neurons are activated with softmax and cross entropy to evaluate the model performance.
The quick demo also supports batch, mini-batch and online-trainig / stochastic gradient descent (SGD). 

The weight initialization is a pseudo random function with known numbers. 
"weightInitRange" set the range of the weights, 0.314 seems a nice start value, 
so the weight init works only with one hyperparameter.

Here is a lot more to say, the reason for this weight initialization is not the performence, this is absolutly not the best way, but it's a good way to keep everthing more comparable. 
If you want a better way, take a look at the glorot initialization: 
["The Five Neural Network Weight Initialization Algorithms"](https://jamesmccaffrey.wordpress.com/2018/12/06/the-five-neural-network-weight-initialization-algorithms/)

The code could look like this:
```
      for(int i=0, w=0; i < dnn; i++, w+=u[i]*u[i+1])
      { 
       float sd = (float)sqrt(6.0f / (u[i] + u[i+1]));
       for(int m=w; m < w+u[i]*u[i+1]; m++)
         weight[m] = rnd.nextFloat(-sd, sd);
      }  
```

A special feature in this concept is the bounce restriction method.
The method in the trivial form is simpel, if a delta^2 breach the limit, the network waits till the delta is cooling down before the NN updates the weight again, but the weight itself continues to work in the network all the time. This makes the NN much more robust. 

This line works as cheap momentum:
```
delta[m] *= 0.5f;
```

With the perceptron concept its possible to use only one array for every neuron. Every data will be updated just in time.
The biggest step to understand the concept in addition to the understanding of how a perceptron works, 
is to understand why the j index need to be initialized with the size of the input neurons, remember that.

Feed Forward (FF) and Backpropagation (BP) treat the current neuron in FF and the gradient in the BP excatly in the same way, forward and backward. This means FF needs the products of their neurons times their weights and optionally the bias, then we got the netinput and the neuron can be activated.
The same thing with the gradient, here the step is a little bit trickier, because the gradient calculation treats the output neurons with (target - output) in a different way as the hidden neurons, where the gradient is the sum of the products from the gradients on the layer before times their own weights.

On the picture below you can see the identical results in C#, CPP and JAVA. JAVA was tested with Eclipse, C# and CPP with Visual Studio 2017.

![alt text](https://user-images.githubusercontent.com/53048236/61723001-99813b00-ad6b-11e9-81ea-aaa683a98b4f.png)

This article assumes you have at least intermediate programming skills and a basic idea of neural networks and machine learning(ML), but doesn’t assume you know anything about the perceptron concept. The demo code contains only the essetials to keep the main ideas as clear as possible and the size of the code small. 

To better understand deep neural networks, I can recommend the work of Professor James D. McCaffrey. His blog is the first address with Google and stack overflow, which I inform myself.
His articles written in different languages (C#, Python, JavaScript, R) gave me a good understanding of the very complex systems.

In addition, he has published short but really good books that help a lot in understanding of ML.

His life's work has taught me almost everything that's important in ML, and I learn more every day.

The perceptron concept consists of a heart that is the same in every language. The other part is language specific.
More specifically, the whole training code, except for the function of the filestream, is the same in all languages.
Also arrays, math functions, print functions and some include stuff need special attention for every language, but that was it.
The complicated algorithms can be easily ported.

Requirement for the demo code in every language is the unzipped MNIST dataset, I would prefer 7zip for this work, the default path for the dataset is "C:\mnist\", one language was bleating, so I was taking a folder. In the demo the trainingset with their label are activ. 
That includes a total of 60,000 images and their labels. This results in a maximum number of 60,000 training examples that must not be exceeded.


And so I've explained the concept myself, ignore the math, here I was using (Output - Target) which is just as possible.

![WP_20190423_00_35_17_Pro](https://user-images.githubusercontent.com/53048236/61755635-ca3b9180-adb8-11e9-99a6-adfce47950a5.jpg)

Let's focus on the green index, input, hidden and output neurons are in only one array, the neuron[], so the network used one index for all neurons, the i loop seperates the layer for input, hidden(1...n) or output neurons. And here we have to talk about the j index, thats not the whole index, because j is the activation index after the inputs and starts on the first neuron with index 3 on hidden layer 1. So lets talk about the gradient array in red, which starts with index 0 on the green index position 3, that is the index for the gradient[], but it's also the index for the bias[], or the netinput[] we dont need in this concept. 

The process I show here is not very accurate, but thats ok, think about the left side in green as the FF operation, and the right side should show the BP in red with the components we need from the FF process to calculate the delta's in gold for the new weight. 

It may not be obvious, but once you understand the FF process, you have understood the BP process for calculating the gradient. 
To become aware of this, you only have to turn the BP process 180 degrees. The initialization of the inputs and the difference of the error are the start for FF and BP, and the product summation for netinput and gradient are reflected. If you rotate the image 180 degrees, the end is the beginning and the plus becomes minus. Three Inputs for FF becomes 2 outputs for BP. From FF 4 [3] to 20 [19] goes from 20 [19] to 4 [3] with BP. 

## Lets start to build our NN with the layers in the training cycle

With MNIST the input would be 784 for each pixel, but the reference works only with 3 inputs, so we just imagine a image with three pixels for the reference example.

To realize the idea we need three loops, the outer i loop for the layer, the middel k loop for every neuron we need to activate for our output (right sided k) operations, and the inner n loop for the input (left sided n) neurons and every weight, to calc the products we add to the net variable instead the netinput[] array after we leave the n loop.

First we need to add our prepared input neurons like this:

```
      for (int n = 0; n < inputs; ++n) 
         // neuron[n] = insert the prepared inputs!
```
Here we can do operations on the input layer, thats good to know.

After we got the input neurons we can calculate the FF, here we start just with the layer loop.

```
      for (int i = 0, j = inputs, w = 0, t = 0; i < dnn; i++, t += u[i - 1], w += u[i] * u[i - 1])  // layer
      {
         // calc the NN loops with neurons, weights, etc, bam...
      }
```

So j starts with the size of the input neurons, thats because we want to activate all neurons till we end with the activated output neuron 20 in array position 19 (neuron[19]) as seen in the picture above. Here t (t: i0=0, i1=3, i2=8, i3=13) saves the neuron steps and serves the n-sided neurons, same with w (w:i0=0, i1=15, i2=40, i3=65), saves the weight steps.

Now the neurons!
```
      for i in layers
            for (int k = 0; k < u[i + 1]; k++, j++) // neurons
            {
                  // 1. add bias to netinput
                  float net = bias[j - inputs];
                   
                  // 2. calculate the inner n loop
                  
                  // 3. after the n loop activate all hidden neurons and let the outputs pass like this:                
                  neuron[j] = i == dnn - 1 ? net : net > 0 ? net : 0;                                       
            }//--- k ends     
```

j starts with inputs = 3 and end on the last output neuron[nns-1] = 19.

k goes the steps seperate (5, 5, 5, 2) because u[] starts with u[i+1]=5,
so if the k loop is done, j ends with 3+5+5+5+2=20.

The last operation in the k loop is to add the netinput[16] to neuron[19].

Ehm, netinput[16]???, thats one of the clues of this concept, because we dont need a netinput array and take instead a fast efficency variable to sum up the products + the bias to the net variable. 

Thats massiv, the NN needs only 2 arrays and one variable instead of 3 arrays in the inner n loop.
Lets finish this with the inner weight loop ;)

``` 
      for i in layer
            for k in neurons                    
                  for (int n = t, m = w + k; n < t + u[i]; n++, m += u[i + 1]) // weights
                     // sum the products to the netinput
                     net += neuron[n] * weight[m];                
```
n starts with 0 because on i0 t is 0 and the loop end on i0 is t + u[i] = 3, to run the the n-sided neuron[0, 1, 2] for every k neuron.

m = w + k, here k adds k+=1 after every k loop, w add the steps after every i loop,
the weights m=0, m=5, m=10 connects to neuron[3] see the picture above.

If k increments the first time, the next weights are w+k(1) = m=1, m=6, m=11 for neuron[4],
we end on the first layer with the last 3 weights with w+k(4) = m=4, m=9, m=14 for neuron[7].

After the first layer is done, w starts on i1 with w=15, weight[15] is the first weight on the next layer!
Lets think about n again, n -> k means we need to add the complete n side for every k sided neuron.

Keep in mind, n goes to k!

Some termenology alert, a layer means normaly the connection between the input and their outputs (input * output = layer), but it's also common to name the layer as input, hidden or output. Correctly I would name it connection layer, which need 2 parts.
So in the i loop a layer means (u layer: i0 = (3 * 5) i1 = (5 * 5) i2 = (5 * 5) i3 = (5 * 2)) which results in this case in 4 layer, 20 neurons and 75 weights.


The result of the three loops could look like this:

```
dnn = u.len - 1 // = 4 on the reference 3,5,5,5,2

inputs = u[0] // = 3

output = u[dnn] // = 2

nns = sumUp(u) // size of the neurons is the sum of u[] = 3 + 5 + 5 + 5 + 2 = 20

wnn = sumProducts(u) // size of weights = sum of the products of u = 3 * 5 + 5 * 5 + 5 * 5 + 5 * 2 = 75
```
*the gradient, bias and netinput index is just the (nns-inputs) or (j-inputs), as example I added a bias for FF*

```
for (int j = inputs, w = 0, t = 0; i < dnn; i++, t += u[i - 1], w += u[i] * u[i - 1])  // layer

   for (int k = 0; k < u[i+1]; j++, k++) // neuron

      net = bias[j-inputs]

      for (int n = t, m = w + k; n < t + u[i + 1]; n++, m += u[i + 1]) // weight

         net += neuron[n] * weight[m]

      if(net more then 0 (relu) or i is outputlayer (output for softmax with maxtrick))

       neuron[j] = net

      else

       neuron[j] = 0
```

The output layer will be activated with softmax. One special treatment, but only for a neural networks, is only one calculation for the max value of the output layer for the cross entropy each run, because the other calculations multiplys with 0, so I let them out and no loop is needed.

The hardest part is the backpropagation, here we go just backwards.

**What does this mean exactly?**
 
for clarity:

ls = loss iterator with the size of the output neurons, thats what we need = (output - 1)

wd = weight delta starts on the last index array position = (wnn - 1)

wg = weight gradient = wd

us = neuron steps, we start on the last neuron of the last hidden-layer = (nns - output - 1) 

gs = gradient steps = (nns - inputs - 1) because on the FF we start activation on the first hidden neuron 

```
  for (int i = dnn, j = nns - 1, ls = output, wd = wnn - 1, wg = wd, us = nns - output - 1, gs = nns - inputs - 1;
      i != 0; i--, wd -= u[i + 1] * u[i + 0], us -= u[i], gs -= u[i + 1])
      for (int k = 0; k != u[i]; k++, j--)
      {
          float gra = 0;
         //--- first check if output or hidden layer
          if (i == dnn) // calc output gradient with (t - o)
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

First we check if we are on the output layer and than we calc the gradient with (target - output) and update the deltas for the weights of this gradient. Thats the strongest "just in time" component in this concept, because we dont need more code and resources to handle this operation. If we done with the output, we calc the gradient for the hidden nerurons, and this process ist just the FF with the net variable, but backwards.
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


### Warning!

*The Python version is only for fun here, the code works hundred times slower than the other versions!*


