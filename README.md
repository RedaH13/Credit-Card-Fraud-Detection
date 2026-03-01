# Credit Card Fraud Detection using Ensemble Machine Learning 💳🕵️‍♂️

**Author:** Hadarbach Med Reda

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using various Machine Learning algorithms. The primary challenge in fraud detection is the massive **class imbalance**—fraudulent transactions represent a tiny fraction of everyday purchases. 

This notebook demonstrates an end-to-end data science pipeline, from data exploration and preprocessing to handling imbalanced classes using **SMOTE**, and finally training and evaluating powerful ensemble models.

## ⚠️ The Challenge: Imbalanced Data
In this dataset, frauds account for only **0.172%** of all transactions (492 frauds out of 284,807 transactions). 
If a model simply guesses "Not Fraud" every time, it achieves 99.8% accuracy but misses all the actual fraud. Therefore, traditional accuracy is a misleading metric here.

## 🛠️ Methodology & Solution
1. **Data Resampling (SMOTE):** To prevent the models from ignoring the minority class, the **Synthetic Minority Over-sampling Technique (SMOTE)** was applied to the training set. This generated synthetic fraud examples to perfectly balance the classes.
2. **Model Training:** Several algorithms were trained and compared, moving from basic classifiers to advanced ensemble techniques:
   * K-Nearest Neighbors (KNN)
   * Random Forest
   * SVM
3. **Evaluation Strategy:** Models were evaluated based on their **Confusion Matrix**, prioritizing:
   * **Recall (Sensitivity):** Maximizing the detection of actual fraud (minimizing False Negatives).
   * **Precision:** Ensuring legitimate customers aren't unnecessarily blocked (minimizing False Positives).
   * **F1-Score:** Finding the optimal balance between Precision and Recall.

## 💻 Technologies Used
* **Python 3**
* **Pandas & NumPy:** Data manipulation and analysis.
* **Scikit-Learn:** Machine Learning algorithms, metrics, and preprocessing.
* **Imbalanced-Learn (imblearn):** Implementation of SMOTE.
* **Matplotlib & Seaborn:** Data visualization and Confusion Matrix plotting.

## 🚀 How to Run
1. Clone this repository to your local machine.
2. Install the dataset (.csv file from Dataset Link file)
3. Ensure you have the required libraries installed:
   ```bash
   pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
