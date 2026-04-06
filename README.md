# -Lab-1.2-Perceptron-Activations-Backprop-from-scratch-
This README covers the fundamental mechanics of neural networks by implementing the core components—**Perceptrons**, **Activation Functions**, and **Backpropagation**—from scratch using only `NumPy`.

---

This lab moves "under the hood" of deep learning frameworks to implement the mathematical engines that drive model learning. It covers the transition from simple linear classifiers to non-linear Multi-Layer Perceptrons (MLPs).

## 🧠 Project Sections

### 1. The Classic Perceptron
The script begins with the **Perceptron Learning Rule**. It uses a synthetic dataset (blobs) to find a linear decision boundary.
* **Mechanism:** If the model misclassifies a point, it nudges the weight vector $w$ toward the correct label.
* **Decision Boundary:** The code calculates the line where $w \cdot X = 0$ to visualize how the classes are separated.

### 2. Non-Linear Activation Functions
To solve complex problems, neurons use activation functions to introduce non-linearity. This lab implements and plots:
* **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$ (Squashes values between 0 and 1).
* **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$ (Thresholds negative values to zero).
* **Tanh:** $\tanh(z)$ (Squashes values between -1 and 1).

The script also includes their **derivatives**, which are essential for the backpropagation process.

### 3. Backpropagation from Scratch (The XOR Problem)
The final section implements a 2-layer neural network to solve the **XOR problem**. XOR is a classic challenge because it is not "linearly separable"—a single line cannot separate the classes.

#### The Architecture:
* **Input Layer:** 2 neurons (the coordinates).
* **Hidden Layer:** 2 neurons with `tanh` activation.
* **Output Layer:** 1 neuron with `sigmoid` activation.

#### The Training Loop:
1.  **Forward Pass:** Compute predictions by flowing data through weights and activations.
2.  **Loss Calculation:** Mean Squared Error (MSE).
3.  **Backward Pass (Backprop):** * Calculate the gradient of the loss with respect to each weight using the **Chain Rule**.
    * Propagate the error from the output layer back to the hidden layer.
4.  **Weight Update:** Adjust weights using Gradient Descent: $W = W - \eta \cdot \nabla W$.

---

## 📊 Key Results

* **Linear Separation:** The initial Perceptron successfully separates simple clusters.
* **XOR Solution:** After 5,000 epochs of manual backpropagation, the loss decreases significantly, and the model correctly predicts the XOR logic gate:
    * `[0,0] → 0`
    * `[0,1] → 1`
    * `[1,0] → 1`
    * `[1,1] → 0`

---

## 🛠️ Technical Stack
* **NumPy:** Used for all matrix operations and gradient calculations.
* **Matplotlib:** Used to visualize the decision boundaries and activation curves.
* **Scikit-Learn:** Used only for generating the initial synthetic "blob" data.

> **Note:** This lab demonstrates that deep learning is essentially a sequence of matrix multiplications and calculus (the chain rule) applied iteratively.
