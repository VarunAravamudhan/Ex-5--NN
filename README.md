<H3>ENTER YOUR NAME Varun A</H3>
<H3>ENTER YOUR REGISTER NO. 212224240178 </H3>
<H3>EX. NO.5</H3>
<H3>DATE: 04-09-26 </H3>
<H1 ALIGN =CENTER>Implementation of XOR  using RBF</H1>
<H3>Aim:</H3>
To implement a XOR gate classification using Radial Basis Function  Neural Network.

<H3>Theory:</H3>
<P>Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows XOR truth table </P>

<P>XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below </P>




<P>The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.
A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.
A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.
</P>





<H3>ALGORITHM:</H3>
Step 1: Initialize the input  vector for you bit binary data<Br>
Step 2: Initialize the centers for two hidden neurons in hidden layer<Br>
Step 3: Define the non- linear function for the hidden neurons using Gaussian RBF<br>
Step 4: Initialize the weights for the hidden neuron <br>
Step 5 : Determine the output  function as 
                 Y=W1*φ1 +W1 *φ2 <br>
Step 6: Test the network for accuracy<br>
Step 7: Plot the Input space and Hidden space of RBF NN for XOR classification.

<H3>PROGRAM:</H3>

```py
import numpy as np
import matplotlib.pyplot as plt

# XOR data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([0,1,1,0])

# RBF centers
mu1 = np.array([0,1])
mu2 = np.array([1,0])

# Gaussian RBF
def rbf(x, mu):
    return np.exp(-np.sum((x-mu)**2))

# Transform inputs
A = np.array([[rbf(x,mu1), rbf(x,mu2), 1] for x in X])

# Find weights
W = np.linalg.pinv(A) @ Y

# Prediction
pred = np.round(A @ W)

print("Weights:", W)
for x, y in zip(X, pred):
    print("Input:", x, "Predicted:", int(y))

# Graphs
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(X[Y==0,0], X[Y==0,1], label="Class 0")
plt.scatter(X[Y==1,0], X[Y==1,1], label="Class 1")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("XOR Input")
plt.legend()

plt.subplot(1,2,2)
plt.scatter(A[Y==0,0], A[Y==0,1], label="Class 0")
plt.scatter(A[Y==1,0], A[Y==1,1], label="Class 1")
plt.xlabel("RBF1")
plt.ylabel("RBF2")
plt.title("RBF Transformed")
plt.legend()

plt.show()
```

<H3>OUTPUT:</H3>

<img width="725" height="510" alt="image" src="https://github.com/user-attachments/assets/d9c15a24-2ab6-4973-b595-d167a735ebc4" />


<H3>Result:</H3>
Thus , a Radial Basis Function Neural Network is implemented to classify XOR data.








