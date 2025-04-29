# Iris Dataset Classification - ML Models Comparison

This project explores and compares various machine learning classification algorithms implemented from scratch on the Iris dataset, including:

- Decision Tree Classifier (ID3 Algorithm)
- Decision Tree Classifier (CART Algorithm)
- Random Forest
- Naive Bayes

## Dataset

The Iris dataset consists of 150 samples with 4 features:
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm

Target classes:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Preprocessing

- Species labels were encoded as numerical values.
- Feature normalization using Min-Max scaling (for Naive Bayes).
- Dataset split into:
  - Training Set: 70%
  - Validation Set: 15%
  - Test Set: 15%

## Models and Results

### ID3 Decision Tree
- Built using entropy and information gain.
- Post-pruning based on validation set.
- **Training Accuracy**: 90.57%
- **Testing Accuracy**: 86.36%
  ![image](https://github.com/user-attachments/assets/00645ab2-5e93-414c-be53-e9b10de634ea)


### CART Decision Tree
- Built using Gini impurity.
- Tree depth limited for overfitting control.
- **Training Accuracy**: 98.11%
- **Testing Accuracy**: 95.45%
  ![image](https://github.com/user-attachments/assets/b1861a5b-d230-4408-a0ab-81373b36863a)


### Random Forest
- 30 CART-based decision trees, each trained on bootstrapped samples.
- Max tree depth: 3
- **Training Accuracy**: 96.23%
- **Testing Accuracy**: 100%
  ![image](https://github.com/user-attachments/assets/294ce8bd-5876-483b-9c00-f49a5598b8e7)


### Naive Bayes
- Gaussian Naive Bayes with Min-Max normalization.
- **Test Accuracy**: 90%
![image](https://github.com/user-attachments/assets/6540b341-ac8a-460d-a1c8-88741086674e)
![image](https://github.com/user-attachments/assets/033d1b15-85b1-4af8-a22e-811b112867f5)


## Conclusion

All models demonstrated solid performance, with Random Forest achieving perfect test accuracy. ID3 and Naive Bayes offer simplicity and interpretability, while CART and Random Forest provide higher accuracy and robustness.
![image](https://github.com/user-attachments/assets/ae2ff2d3-bf04-477a-ae3d-1180c8bac248)



## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib (optional for visualization)

## Usage

1. Clone this repository.
2. Run each classifier script (e.g., `id3_classifier.py`, `cart_classifier.py`, `random_forest.py`, `naive_bayes.py`) to see training and evaluation results.

