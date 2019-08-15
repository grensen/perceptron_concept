function main()
{
         console.log("\nrunning\n");       
       
	 let u                 = [784, 25, 25, 25, 10];
         let learningRate      = 0.0067;
         let bounceResRate     = 50.0;
         let weightInitRange   = 0.35;
         let runs              = 1000;
         let miniBatch         = 8;
         let networkInfoCheck  = 10; 
	 
         let dnn = u.length - 1, nns = 0, wnn = 0, inputs = u[0], output = u[dnn], correct = 0;
         let ce = 0, ce2 = 0;

         for (let n = 0; n < dnn + 1; n++) nns += u[n]; // num of neurons
         for (let n = 1; n < dnn + 1; n++) wnn += u[n - 1] * u[n]; // num of weights

         let neuron   = [nns];
         let gradient = [nns - inputs];
         let weight   = [wnn];
         let delta    = [wnn];
         let target   = [output];

	 let FS = require('fs');
	 let MNISTimage = FS.readFileSync("train-images.idx3-ubyte", "binary");
	 let MNISTlabel = FS.readFileSync("train-labels.idx1-ubyte", "binary");
         
         // load input data...
         for (let n = 0, p = 314; n < wnn; n++)
             weight[n] = ((p = p * 2718 % 2718281) / (2718281.0 * Math.E * Math.PI * weightInitRange));

        //--- start training
         for (let x = 1; x < runs + 1; x++)
         {

           //+----------- 1. MNIST as Inputs --------------------------------------+      
           // feed input data...
	    for (let n = 0; n < inputs; n++)
            {            
             neuron[n] = neuron.push(MNISTimage [((x-1)*784) + n + 16]) / 255;
            }   
            let tmp = [];
  	    tmp.push(MNISTlabel [(x-1) + 8]);
            let targetNum = tmp;
            
           //+----------- 2. Feed Forward -----------------------------------------+            
            for (let i = 0, j = inputs, t = 0, w = 0; i < dnn; i++, t += u[i - 1], w += u[i] * u[i - 1])
                for (let k = 0; k < u[i + 1]; k++, j++){
                    let net = gradient[j - inputs] = 0;
                    for (let n = t, m = w + k; n < t + u[i]; n++, m += u[i + 1])
                        net += neuron[n] * weight[m];
                    neuron[j] = i == dnn - 1 ? net : net > 0 ? net : 0;
                }//--- k ends    

           //+------------ 3. NN prediction ---------------------------------------+
            let outMaxPos = nns - output;
            let outMaxVal = neuron[nns - output], scale = 0;
            for (let i = nns - output + 1; i < nns; i++)
                if (neuron[i] > outMaxVal){
                    outMaxPos = i; outMaxVal = neuron[i];
                }
            if (targetNum + nns - output == outMaxPos) correct++;

           //+----------- 4. Loss / Error with Softmax and Cross Entropy ----------+                    
            for (let n = nns - output; n != nns; n++)
                scale += Math.exp(neuron[n] - outMaxVal);
            for (let n = nns - output, m = 0; n != nns; m++, n++)
                neuron[n] = Math.exp(neuron[n] - outMaxVal) / scale;               
            ce2 = (ce -= Math.log(neuron[outMaxPos])) / x;   
           
           //+----------- 5. Backpropagation --------------------------------------+    
            target[targetNum] = 1.0;
            for (let i = dnn, j = nns - 1, ls = output, wd = wnn - 1, ws = wd, us = nns - output - 1, gs = nns - inputs - 1;
            i != 0; i--, wd -= u[i + 1] * u[i + 0], us -= u[i], gs -= u[i + 1])
                for (let k = 0; k != u[i]; k++, j--){
                    let gra = 0;
                   //--- first check if output or hidden, calc delta for both
                    if (i == dnn)
                        gra = target[--ls] - neuron[j];
                    else if(neuron[j] > 0)
                        for (let n = gs + u[i + 1]; n > gs; n--, ws--)
                            gra += weight[ws] * gradient[n];
                    else ws -= u[i + 1];
                    for (let n = us, w = wd - k; n > us - u[i - 1]; w -= u[i], n--)
                        delta[w] += gra * neuron[n];
                    gradient[j - inputs] = gra;
                }
            target[targetNum] = 0;

           //+----------- 6. update Weights ---------------------------------------+         
            if ((x % miniBatch == 0) || (x == runs - 1)){
                for (let m = 0; m < wnn; m++){
                   //--- bounce restriction
                    if (delta[m] * delta[m] > bounceResRate) continue;
                   //--- update weights
                    weight[m] += learningRate * delta[m];
                    delta[m] *= 0.67;
                }
            } //--- batch end
		 
            if (x % (runs / networkInfoCheck) == 0)
            	console.log("runs: " + x + " accuracy: " + (correct * 100.0 / x));

         } // x end
	
         console.log("\nneurons: " + nns + " weights: " + wnn + " batch: " + miniBatch);
         console.log("accuracy: " + (correct * 100.0 / (runs * 1.0)) + " cross entropy: " + ce2);
         console.log("correct: "+(correct) + " incorrect: " + (runs - correct));
       
}

main();
