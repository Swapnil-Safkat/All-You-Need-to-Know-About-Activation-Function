# All You Need to Know About Activation Function

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

#### **1. Linear Activation Function*

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
import matplotlib.pyplot as plt
def plot(function,x,name):
    y = function(x)
    plt.figure(figsize=(8, 6))
    plt.plot(x, x, label="actual", color="blue",linewidth=4)
    plt.plot(x, y, label="after activation function", color="red",linewidth=2)
    plt.title(name+"activation function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend()
    plt.show()
```


