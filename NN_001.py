import numpy as np
import math

np.seterr(over='ignore')

def main():

    print("\nrunning...\n");

   #--- create neural network
    u = np.array([784, 10], dtype=np.int)
    learningRate = 0.0067
    bounceResRate = 50.0
    weightInitRange = 0.35
    runs = 1000
    miniBatch = 16
    networkInfoCheck = 10

    dnn = len(u) - 1; output = u[dnn]; inputs = u[0]
    nns = wnn = correct = 0; ce = ce2 = 0.0;

    for i in range(dnn + 1): nns += u[i]
    for i in range(1, dnn + 1): wnn += u[i] * u[i - 1]

    neuron   = np.zeros(shape=nns, dtype=np.float32)
    gradient = np.zeros(shape=nns - inputs, dtype=np.float32)
    weight   = np.zeros(shape=wnn, dtype=np.float32)
    delta    = np.zeros(shape=wnn, dtype=np.float32)
    target   = np.zeros(shape=output, dtype=np.float32)

    MNISTimages = open("C:\\mnist\\train-images.idx3-ubyte", "rb")
    MNISTlabels = open("C:\\mnist\\train-labels.idx1-ubyte", "rb")

    MNISTimages.read(16); MNISTlabels.read(8);

    ov = np.array([314], dtype=np.int)
    for i in range(wnn):
        ov[0] = math.fmod((ov[0] * 2718), 2718281)
        weight[i] = ov[0] / (np.e * np.pi * weightInitRange * 2718281.0)

    for x in range(1, runs):

        # 1. take MNIST as Inputs
        for i in range(inputs):
            neuron[i] = ord(MNISTimages.read(1)) / 255.0
        targetNum = ord(MNISTlabels.read(1))

        # 2. feed forward
        j = inputs; t = w = 0;
        for i in range(0, dnn, 1):
            for k in range(0, u[i + 1], 1):
                net = gradient[j - inputs] = 0
                m = w + k
                for n in range(t, t + u[i], 1):
                    net += neuron[n] * weight[m]
                    m += u[i + 1]
                neuron[j] = net if i == dnn - 1 or net > 0 else 0
                j += 1
            t += u[i]; w += u[i] * u[i + 1];

        # 3. NN prediction
        outMaxPos = nns - output; outMaxVal = neuron[outMaxPos];
        for i in range(nns - output + 1, nns, 1):
            if neuron[i] > outMaxVal:
                outMaxPos = i; outMaxVal = neuron[i];

        if targetNum + nns - output == outMaxPos: correct += 1

        # 4. Loss / Error
        scale = 0.0
        for i in range(nns - output, nns, 1):
            scale += np.exp(neuron[i] - outMaxVal)
        for i in range(nns - output, nns, 1):
            neuron[i] = np.exp(neuron[i] - outMaxVal) / scale
        ce -= np.log(neuron[outMaxPos])
        ce2 = ce / x

        # 5. backpropagation
        target[targetNum] = 1.0

        j = nns - 1; ls = output - 1; wg = wd = wnn - 1
        us = nns - output - 1; gs = nns - inputs - 1
        for i in range(dnn, 0, -1):
            for k in range(u[i]):
                gra = 0.0
                nj = neuron[j]
                if i == dnn:
                    gra = target[ls] - nj
                    ls -= 1
                elif nj > 0:
                    for n in range(gs + u[i + 1], gs, -1):
                        gra += weight[wg] * gradient[n]
                        wg -= u[i + 1]
                else:
                    wg -= u[i + 1]
                w = wd - k
                for n in range(us, us - u[i - 1], -1):
                    delta[w] += gra * neuron[n]
                    w -= u[i]
                gradient[j - inputs] = gra
                j -= 1
            wd -= u[i - 1] * u[i]; us -= u[i - 1]; gs -= u[i];
        target[targetNum] = 0.0

        # 6. update weights
        if x % miniBatch == 0 or x == runs - 1:
            for m in range(0, wnn, 1):
                if delta[m] * delta[m] < bounceResRate:
                    weight[m] += learningRate * delta[m]
                    delta[m] *= 0.67

        if (x % (runs / networkInfoCheck) == 0):
            print("runs: ", x, " accuracy: ", (correct * 100.0 / x))

    print("\nneurons: ", nns, " weights: ", wnn, " batch: ", miniBatch)
    print("accuracy: ", correct * 100.0 / (runs * 1.0), " cross entropy: ", ce2)
    print("correct: ", correct, " incorrect: ", (runs - correct))

    MNISTimages.close(); MNISTlabels.close();

if __name__ == "__main__":
    main()
