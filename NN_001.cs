using System;
using System.IO;

namespace NN_001
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("running...\n");
                
               //--- create neural network 
                int[] u               = { 784, 25, 25, 25, 10 };
                float learningRate    = 0.0067f;
                float bounceResRate   = 50.0f;
                float weightInitRange = 0.35f;
                int runs              = 10000;
                int miniBatch         = 8;
                int networkInfoCheck  = 10;
                
                int dnn = u.Length - 1, nns = 0, wnn = 0, inputs = u[0], output = u[dnn], correct = 0;
                float ce = 0, ce2 = 0;

                for (int n = 0; n < dnn + 1; n++) nns += u[n]; // num of neurons
                for (int n = 1; n < dnn + 1; n++) wnn += u[n - 1] * u[n]; // num of weights

                float[] neuron   = new float[nns];
                float[] gradient = new float[nns - inputs];
                float[] weight   = new float[wnn];
                float[] delta    = new float[wnn];
                float[] target   = new float[output];

               //--- testdata "t10k-images.idx3-ubyte" , "t10k-labels.idx1-ubyte" 
                FileStream MNISTlabels = new FileStream(@"C:\mnist\train-labels.idx1-ubyte", FileMode.Open);
                FileStream MNISTimages = new FileStream(@"C:\mnist\train-images.idx3-ubyte", FileMode.Open);

                MNISTimages.Seek(16, 0);
                MNISTlabels.Seek(8, 0);
              
               //--- get pseudo random init weights
                for (int n = 0, p = 314; n < wnn; n++)
                    weight[n] = (float)((p = p * 2718 % 2718281) / (2718281.0 * Math.E * Math.PI * weightInitRange));

               //--- start training
                for (int x = 1; x < runs + 1; x++)
                {
                   
                   //+----------- 1. MNIST as Inputs ---------------------------------------+      
                    for (int n = 0; n < inputs; ++n)
                        neuron[n] = MNISTimages.ReadByte() / 255.0f;
                    int targetNum = MNISTlabels.ReadByte();

                   //+----------- 2. Feed Forward ------------------------------------------+            
                    for (int i = 0, j = inputs, t = 0, w = 0; i < dnn; i++, t += u[i - 1], w += u[i] * u[i - 1])
                        for (int k = 0; k < u[i + 1]; k++, j++)
                        {
                            float net = gradient[j - inputs] = 0;
                            for (int n = t, m = w + k; n < t + u[i]; n++, m += u[i + 1])
                                net += neuron[n] * weight[m];
                            neuron[j] = i == dnn - 1 ? net : net > 0 ? net : 0;
                        }//--- k ends    

                   //+------------ 3. NN prediction ----------------------------------------+
                    int outMaxPos = nns - output;
                    float outMaxVal = neuron[nns - output], scale = 0;
                    for (int i = nns - output + 1; i < nns; i++)
                        if (neuron[i] > outMaxVal)
                        {
                            outMaxPos = i;
                            outMaxVal = neuron[i];
                        }
                    if (targetNum + nns - output == outMaxPos) correct++;

                   //+----------- 4. Loss / Error with Softmax and Cross Entropy -----------+                    
                    for (int n = nns - output; n != nns; n++)
                        scale += (float)Math.Exp(neuron[n] - outMaxVal);
                    for (int n = nns - output, m = 0; n != nns; m++, n++)
                        neuron[n] = (float)Math.Exp(neuron[n] - outMaxVal) / scale;
                    ce2 = (ce -= (float)Math.Log(neuron[outMaxPos])) / x;

                   //+----------- 5. Backpropagation ---------------------------------------+    
                    target[targetNum] = 1.0f;
                    for (int i = dnn, j = nns - 1, ls = output, wd = wnn - 1, wg = wd, us = nns - output - 1, gs = nns - inputs - 1;
                        i != 0; i--, wd -= u[i + 1] * u[i + 0], us -= u[i], gs -= u[i + 1])
                        for (int k = 0; k != u[i]; k++, j--)
                        {
                            float gra = 0;
                           //--- first check if output or hidden, calc delta for both connected weights
                            if (i == dnn)
                                gra = target[--ls] - neuron[j];
                            else if(neuron[j] > 0) 
                                for (int n = gs + u[i + 1]; n > gs; n--, wg--)
                                    gra += weight[wg] * gradient[n];
                            else wg -= u[i + 1]; 
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
                        Console.WriteLine("runs: " + x + " accuracy: " + (correct * 100.0f / x));
                } //--- runs end

                Console.WriteLine("\nneurons: " + nns + " weights: " + wnn + " batch: " + miniBatch);
                Console.WriteLine("accuracy: " + (correct * 100.0 / (runs * 1.0f)) + " cross entropy: " + ce2);
                Console.WriteLine("correct: " + (correct) + " incorrect: " + (runs - correct));
                Console.ReadLine();

                MNISTimages.Close(); MNISTlabels.Close();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
            }
        } // Main
    } // Program
} // ns
