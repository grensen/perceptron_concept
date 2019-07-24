#include "stdafx.h"
#include <iostream>
#include <fstream>

#ifndef M_PI 
#define M_PI 3.14159265358979323846 
#endif 
#ifndef M_E 
#define M_E 2.71828182845904523536
#endif 

typedef unsigned char byte;
using namespace std;

int main()
{
	std::cout << "running..." << endl << endl;

       //--- create neural network 
	int u[]               = { 784, 25, 25, 25, 10 };
	float learningRate    = 0.0067f;
	float bounceResRate   = 50.0f;
	float weightInitRange = 0.35f;
	int runs              = 10000;
	int miniBatch         = 8;
	int networkInfoCheck  = 10;

	int dnn = (sizeof(u) / sizeof(*u)) - 1, nns = 0, wnn = 0, inputs = u[0], output = u[dnn], correct = 0;
	float ce = 0, ce2 = 0;

	for (int n = 0; n < dnn + 1; n++) nns += u[n]; // num of neurons
	for (int n = 1; n < dnn + 1; n++) wnn += u[n - 1] * u[n]; // num of weights

	float* neuron   = new float[nns]{0};
	float* gradient = new float[nns - inputs]{0};
	float* weight   = new float[wnn]{0};
	float* delta    = new float[wnn]{0};
	float* target   = new float[output]{0};

	std::fstream ig("C:\\mnist\\train-images.idx3-ubyte", std::ifstream::in | std::ifstream::binary);
	std::fstream lab("C:\\mnist\\train-labels.idx1-ubyte", std::ifstream::in | std::ifstream::binary);
	
	ig.seekp(16 * sizeof(char));
	lab.seekp(8 * sizeof(char));
	
       //--- get pseudo random init weights
	for (int n = 0, p = 314; n < wnn; n++)
		weight[n] = (float)((p = p * 2718 % 2718281) / (2718281.0 * M_E * M_PI * weightInitRange));

       //--- start training
	for (int x = 1; x < runs + 1; x++) 
	{

	       //+----------- 1. MNIST as Inputs --------------------------------------+      
		for (int n = 0; n < inputs; ++n)
		{
			byte pixel = 0;
			ig.read((char*)&pixel, 1);		
			neuron[n] = pixel / 255.0f;
		 }
		byte label = 0;
		lab.read((char*)&label, sizeof(label));
		int targetNum = label; 

	       //+----------- 2. Feed Forward -----------------------------------------+            
		for (int i = 0, j = inputs, t = 0, w = 0; i < dnn; i++, t += u[i - 1], w += u[i] * u[i - 1])
			for (int k = 0; k < u[i + 1]; k++, j++)
			{
				float net = gradient[j - inputs] = 0;
				for (int n = t, m = w + k; n < t + u[i]; n++, m += u[i + 1])
					net += neuron[n] * weight[m];
				neuron[j] = i == dnn - 1 ? net : net > 0 ? net : 0;
			}//--- k ends    

	       //+------------ 3. NN prediction ---------------------------------------+
		int outMaxPos = nns - output;
		float outMaxVal = neuron[nns - output], scale = 0;
		for (int i = nns - output + 1; i < nns; i++)
			if (neuron[i] > outMaxVal) 
			{
				outMaxPos = i; 
				outMaxVal = neuron[i];
			}
		if (targetNum + nns - output == outMaxPos) correct++;

	       //+----------- 4. Loss / Error with Softmax and Cross Entropy ----------+                    
		for (int n = nns - output; n != nns; n++)
			scale += exp(neuron[n] - outMaxVal);
		for (int n = nns - output, m = 0; n != nns; m++, n++)
			neuron[n] = exp(neuron[n] - outMaxVal) / scale;
		ce2 = (ce -= log(neuron[outMaxPos])) / x;

	       //+----------- 5. Backpropagation --------------------------------------+    
		target[targetNum] = 1.0f;
		for (int i = dnn, j = nns - 1, ls = output, wd = wnn - 1, ws = wd, us = nns - output - 1, gs = nns - inputs - 1;
			i != 0; i--, wd -= u[i + 1] * u[i + 0], us -= u[i], gs -= u[i + 1])
			for (int k = 0; k != u[i]; k++, j--) 
			{
				float gra = 0;
				//--- first check if output or hidden, calc delta for both
				if (i == dnn)
					gra = target[--ls] - neuron[j];
				else if (neuron[j] > 0) 
					for (int n = gs + u[i + 1]; n > gs; n--, ws--)
						gra += weight[ws] * gradient[n];
				else ws -= u[i + 1]; 
				for (int n = us, w = wd - k; n > us - u[i - 1]; w -= u[i], n--)
					delta[w] += gra * neuron[n];
				gradient[j - inputs] = gra;
			}
		target[targetNum] = 0;

	       //+----------- 6. update Weights ----------------------------------------+         
		if ((x % miniBatch == 0) || (x == runs - 1)) 
		{
			for (int m = 0; m < wnn; m++) 
			{
				//--- bounce restriction
				if (delta[m] * delta[m] > bounceResRate) continue;
				//--- update weights
				weight[m] += learningRate * delta[m];
				delta[m] *= 0.67f;
			}
		} //--- batch end

		if (x % (runs / networkInfoCheck) == 0)
			std::cout << "runs: " << x << " accuracy: " << (correct * 100.0f / x) << endl;
	} //--- runs end

	std::cout << endl << "neurons:  " << nns << " weights: " << wnn << " batch: " << miniBatch << endl;
	std::cout << "accuracy: " << (correct * 100.0 / (runs * 1.0f)) << " cross entropy: " << ce2 << endl;
	std::cout << "correct: " << (correct) << " incorrect: " << (runs - correct) << endl;
	
	ig.close();lab.close();

	std::system("pause");
   
   return 0;
}
