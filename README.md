![image](https://github.com/Swapnil-Safkat/All-You-Need-to-Know-About-Activation-Function/assets/84597539/fc1d80fa-e36b-48f1-9c0e-d39269055c3a)# All You Need to Know About Activation Function

## **Introduction**

### **Defination of Activation Function**

An **activation function** in an artificial neural network determines the output of a node (neuron) based on its individual inputs and their weights. 

![image](https://github.com/Swapnil-Safkat/All-You-Need-to-Know-About-Activation-Function/assets/84597539/7c296114-fdcf-49ba-a0ee-71610e29b7c2)


### **Purpose of Activation Function**
- Activation function decides whether a neuron should be activated or not.
- Activation function allows to captures hidden non-linear pattern in the data.
- Activation function allows to drow complex relation between inputs and outputs.

### **Properties of Ideal Activation Function**
1. **Non-linear:** Non-linear activation functions can capture non-linearity in the data.
2. **Differentiable:**  We use gradient decent to update weights and biases, so activation function should be differentiable.
3. **Computationally inexpensive:** Activations functions should be simple, easy and fast so that the computational time is minimal.
4. **Zero centered(Normalized):** It helps to normalize data coming through each node. Ex.tanh.
5. **Non-saturating:** If the activation function is saturation then vanishing gradient problem might occur, Ex: sigmoid, tanh.

### **Activation functions and their derivatives**
![image](https://github.com/Swapnil-Safkat/All-You-Need-to-Know-About-Activation-Function/assets/84597539/67dd7fa7-91e7-46c5-be26-000cf88d2ed5)

### **Types of Activation Functions**

#### *1. Linear Activation Function*

##### **Formula** - f(x) = x

##### **Advantages**
  - Most simple
  - Compitationally fast
  - 
##### **Disadvantages**
  - Unable to capture non-linearty in data
  - Poor performance

##### **Output**
This is a general function which will take activation function and input values, and plot input vs output of activation function.

```
import numpy as np
import matplotlib.pyplot as plt
def plot(function,x,name):
    y = [function(i) for i in x]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(name+" activation function")
    fig.set_figwidth(12)
    fig.set_figheight(4)
    # Plot data in the first subplot
    axs[0].plot(x, x)
    axs[0].set_title('Actual')
    axs[0].grid()
    axs[0].legend()
    # Plot data in the second subplot
    axs[1].plot(x, y)
    axs[1].set_title('After Activation function')
    axs[1].grid()
    axs[1].legend()
    plt.show()

```

**Effect of Linear Activation Function***

```
def linear(x):
    return x
x = np.linspace(-10, 10, 400)
plot(linear,x,'linear')
```

![image](https://github.com/Swapnil-Safkat/All-You-Need-to-Know-About-Activation-Function/assets/84597539/09ef941f-9295-40be-b3b4-d2ddb3f9c6bb)


#### *2. Step Activation Function*

##### **Formula** - f(x) = x

##### **Advantages**
  - Simple
  - Compitationally fast
  - 
##### **Disadvantages**
  - Saturating
  - Poor performance

##### **Output**
**Effect of Step Activation Function***

```
def step(x):
    return 10 if x >= 0 else -10
x = np.linspace(-10, 10, 400)
plot(step,x,'step')
```

![image](https://github.com/Swapnil-Safkat/All-You-Need-to-Know-About-Activation-Function/assets/84597539/e7c28aeb-032d-448f-b7f7-6b6894fead97)


#### *3. Sigmoid Activation Function*

The sigmoid function, denoted as σ(x) or sig(x), is a special form of the logistic function.
It maps any real-valued input x to an output in the range (0, 1)

##### **Formula** - f(x) = 1/(1 + e^{-x})
 
##### **Advantages**
  - Simple
  - Output can be used as probability
  - Non-lineary
  - Differentiable
     
##### **Disadvantages**
  - Saturating
  - Non-zero centered
  - Computationally expensive
  - Might cause vanishing gradient problem

##### **Output**
**Effect of Sigmoid Activation Function***

```
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
x = np.linspace(-10, 10, 400)
plot(sigmoid,x,'sigmoid')
```

![image](https://github.com/Swapnil-Safkat/All-You-Need-to-Know-About-Activation-Function/assets/84597539/376fb365-9a7e-41af-8ee5-be96ad4f1780)

#### *4. Tanh Activation Function*

The tanh function is a scaled and shifted version of the hyperbolic tangent function.
It maps any real value as input to an output in the range [-1, 1].

##### **Formula** - f(x) = (e^{x} - e^{-x}) / (e^{x} + e^{-x})
 
##### **Advantages**
  - Non-lineary
  - Differentiable
  - Zero-centered
  - Allows fast training
     
##### **Disadvantages**
  - Saturating
  - Computationally expensive
  - Might cause vanishing gradient problem

##### **Output**
**Effect of Tanh Activation Function***

```
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
x = np.linspace(-10, 10, 400)
plot(tanh,x,'tanh')
```

![image](https://github.com/Swapnil-Safkat/All-You-Need-to-Know-About-Activation-Function/assets/84597539/06516ed9-6dcb-4999-bdd1-7cafa2f929ba)


### All Content Credit:
1. https://github.com/BytesOfIntelligences
2. https://www.youtube.com/@campusx-official
3. https://github.com/alsani-ipe/Understanding-Activation-functions-in-Neural-Networks/tree/main
