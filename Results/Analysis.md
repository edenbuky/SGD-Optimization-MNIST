# Results & Analysis 📊

## **1.Analysis for Hinge Loss Optimization**
Working with the MNIST data set. We aim to optimize the Hinge Loss with L2-regularization using Stochastic Gradient Descent (SGD). The regularized loss function is defined as:
𝓛(𝐰,𝐱,𝑦) = 𝐶∙𝐦𝐚𝐱\{𝟢,𝟷-𝑦⟨𝐱,𝐰⟩\}+𝟢.5∥𝐰∥²
We initialize 𝐰₁ = 𝟢, and at each iteration 𝑡 = 𝟷,... we sample 𝑖 uniformly, and if 𝑦ᵢ𝐰ₜ∙𝐱ᵢ < 𝟷, we update: 𝐰ₜ₊₁ = (𝟷-𝜂ₜ)𝐰ₜ + 𝜂ₜ𝐶𝑦ᵢ𝐱ᵢ.
And  𝐰ₜ₊₁ =(𝟷-𝜂ₜ)𝐰ₜ otherwise, where 𝜂ₜ = 𝜂₀/𝑡 and 𝜂₀ is a constant learning rate.

## **(a)Learning Rate Optimization**
### **Problem Statement**
Train the classifier on the training set. Use cross validation on the validation set to find the best 𝜂₀, assuming '𝑇 = 𝟷𝟢𝟢𝟢' and 𝐶 = 𝟷. For each possible 𝜂₀ 
(for example, you can search on the log scale 𝜂₀= 𝟷𝟢⁻⁵,𝟷𝟢⁻⁴,...,𝟷𝟢⁴,𝟷𝟢⁵ and increase resolution if needed), assess the performance of 𝜂₀ by averaging the accuracy on the validation set across 10 runs.
Plot to average accuracy on the validation set as a function of 𝜂₀.
### **Solution**
I conducted a run with learning rates 
𝜂₀∈\{𝟷𝟢⁻⁵,𝟷𝟢⁻⁴,...,𝟷𝟢⁴,𝟷𝟢⁵\} 
The best result was obtained for 𝜂₀ = 𝟷, with an **accuracy of 97.6%**.
<img width="425" alt="Screenshot 2025-02-11 at 16 28 33" src="https://github.com/user-attachments/assets/40264713-0aa9-48c6-8d9d-0a7bfaa88011" />

To refine the learning rate, we conducted a second run in the range 𝜂₀∈\{𝟢,𝟢.𝟷,𝟢.𝟸...,𝟷.𝟿,𝟸} 
The best result was obtained for 𝜂₀ = 𝟢.𝟽 with an accuracy of 98.16%.
<img width="447" alt="Screenshot 2025-02-11 at 16 28 22" src="https://github.com/user-attachments/assets/a0e51d38-76de-442c-8e38-e145a48179c2" />

---

## **(b) Cross-Validation for Regularization Parameter 𝐶**
### **Problem Statement**
Now, cross-validate on the validation set to find the best 𝐶 given the best 𝜂₀ found in section (a). For each possible 
𝐶 (again, you can search on the log scale as in section (a)), average the accuracy on the validation set across 10 runs. Plot the average accuracy on the validation set as a function of 𝐶.

### **Solution**
As shown in the initial search over the range 𝐶∈\{𝟷𝟢⁻⁴,...,𝟷𝟢⁴\}, the result suggests that 𝐶 = 𝟢.𝟢𝟢𝟢𝟷
yields the best **accuracy of 98.54%**.
In a refined search over the range 𝐶∈\{𝟢,𝟢.𝟢𝟻,...𝟢.𝟻\}, 𝐶 = 𝟢.𝟸 achieves an accuracy of 98.29%,
which is lower than the performance at 𝐶 = 𝟢.𝟢𝟢𝟢𝟷. Thus, we set the final regularization parameter to  𝐶 = 𝟢.𝟢𝟢𝟢𝟷.
<img width="592" alt="Screenshot 2025-02-11 at 16 28 48" src="https://github.com/user-attachments/assets/68ed3214-bd90-4e36-a85a-3579cec870e6" />

---
## **(c) Visualizing the Weight Vector 𝐰**
### **Problem Statement**
Using the best 𝐶, 𝜂₀ you found, train the classifier, but for 𝑇 = 𝟸𝟢𝟢𝟢𝟢.
Show the resulting 𝐰 as an image, e.g. using the following 𝚖𝚊𝚝𝚙𝚕𝚘𝚝𝚕𝚒𝚋.𝚙𝚢𝚙𝚕𝚘𝚝 function: 𝚒𝚖𝚜𝚑𝚘𝚠(𝚛𝚎𝚜𝚑𝚊𝚙𝚎(𝚒𝚖𝚊𝚐𝚎,(𝟸𝟾,𝟸𝟾)), 𝚒𝚗𝚝𝚎𝚛𝚙𝚘𝚕𝚊𝚝𝚒𝚘𝚗='𝚗𝚎𝚊𝚛𝚎𝚜𝚝').
Give an intuitive interpretation of the image you obtain.
### **Solution**
<img width="371" alt="Screenshot 2025-02-11 at 16 29 04" src="https://github.com/user-attachments/assets/d9205797-6113-4b30-b06b-0bf2b1b03379" />
In this image, each pixel corresponds proportionally to the associated weight value.
◦ **Bright regions** indicate **positive weights**, where higher pixel values increase the likelihood of the classifier predicting the positive class label.
◦ **Dark regions** indicate **negative weights**, where higher pixel values decrease the likelihood of a positive prediction.
The dark surrounding areas suggest that high pixel values in these regions are less likely to be associated with the positive class, indicating the presence of the digit "0."
In contrast, bright areas in the dark regions likely indicate that it is not a "0."
Meanwhile, the digit "8" appears to occupy the center of the image with positive values, showing where high pixel values are likely to predict "8" accurately.

---

## **(d) Best Accuracy **
What is the accuracy of the best classifier on the test set?
### **Solution:**
The classifier's best accuracy with 𝐶 = 𝟢.𝟢𝟢𝟢𝟷, 𝜂₀ = 𝟢.𝟽 and  𝑇 = 𝟸𝟢𝟢𝟢𝟢 is **99.28%**

---



