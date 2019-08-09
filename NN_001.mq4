#property strict
#property indicator_chart_window     

void OnStart()
{           
     //--- create neural network 
      int u[]               = { 784, 25, 25, 25, 10 };
      float learningRate    = 0.0067f;
      float bounceResRate   = 50.0f;
      float weightInitRange = 0.35f;
      int runs              = 10000;
      int miniBatch         = 8;
      int networkInfoCheck  = 10;
      
      int dnn = ArraySize(u) - 1, nns = 0, wnn = 0, inputs = u[0], output = u[dnn];
      int  correct = 0, incorrect = 0, correct2 = 0; 
      float ce = 0, ce2 = 0, acc = 0;  
            
      for (int n = 0; n < dnn + 1; n++)nns += u[n];
      for (int n = 1; n < dnn + 1; n++)wnn += u[n-1] * u[n]; 
      
      float neuron[]={};   ArrayResize(neuron,nns);
      float weight[]={};   ArrayResize(weight,wnn);
      float delta[]={};    ArrayResize(delta,wnn);
      float gradient[]={}; ArrayResize(gradient,nns-inputs);
      float target[]={};   ArrayResize(target,u[dnn]);

     //--- testdata "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"
      int MNISTImage = FileOpen("train-images.idx3-ubyte", FILE_READ|FILE_BIN),
          MNISTLabel = FileOpen("train-labels.idx1-ubyte", FILE_READ|FILE_BIN);         
               
      FileSeek(MNISTImage, 16, SEEK_SET); 
      FileSeek(MNISTLabel, 8, SEEK_SET);     
      
      for (int n = 0, p = 314; n < wnn; n++)
       weight[n] = (float)((p = p * 2718 % 2718281) / (2718281.0 * M_E * M_PI * weightInitRange));       
     
      for(int x = 1; x < runs+1; x++)
      { 
      
      //+----------- 1. MNIST as Inputs ---------------------------------------+      
       for(int j = 0; j != 784; j++)
        neuron[j] = FileReadInteger(MNISTImage,CHAR_VALUE) / 255.0f;     
       int targetNum = FileReadInteger(MNISTLabel,CHAR_VALUE); 
      
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
       int outMaxPos = ArrayMaximum(neuron, output, nns - output);
       if(targetNum + nns - output == outMaxPos) correct++;
      
      //+----------- 4. Loss / Error ------------------------------------------+   
       float outMaxVal = neuron[outMaxPos], scale = 0; 
       for(int n = nns - output; n != nns; n++)
        scale += (float)exp(neuron[n] - outMaxVal);
       for(int n = nns - output, m = 0; n != nns; m++, n++)
        neuron[n] = (float)exp( neuron[n] - outMaxVal) / scale;    
       ce2 = (ce -= (float)MathLog(neuron[outMaxPos])) / x;
      
      //+----------- 5. Backpropagation ---------------------------------------+    
       target[targetNum] = 1.0f;
       for (int i = dnn, j = nns - 1, ls = output, wd = wnn - 1, wg = wd, us = nns - ls - 1, gs = nns - inputs - 1;
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
         else 
          wg -= u[i + 1];
         gradient[j - inputs] = gra;           
         for (int n = us, w = wd - k; n > us - u[i - 1]; w -= u[i], n--)
          delta[w] += gra * neuron[n];         
        }
       target[targetNum] = 0;

      //+----------- 6. update Weights ----------------------------------------+         
       if((x%miniBatch==0)||(x==runs-1))     
        for(int m=0;m<wnn;m++)
        {              
         if(delta[m] * delta[m] > bounceResRate)continue;
         weight[m] += delta[m] * learningRate; 
         delta[m] *= 0.67f;
        }        
       
       if (x % (runs / networkInfoCheck) == 0)
        Print("runs: ", x," accuracy: ",DoubleToStr((correct * 100.0f / x),2));
      }// --- x end
      
      FileClose(MNISTImage); FileClose(MNISTLabel);
      
      Print("runs ",runs," batch ",miniBatch," LR: ",learningRate);
      Print("correct: ",correct," incorrect: ",runs-correct,"  ce: ",DoubleToStr(ce2,4)," accuracy: ",DoubleToStr((correct * 100.0f / runs),3));  
}
