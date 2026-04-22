# Diabetes-Prediction-Model-using-Pima-Indians-Database

This repository contains a Google Colab Notebook implementing a diabetes prediction model using the Pima Indians Diabetes Dataset.

The primary goal of this project was to learn **Pytorch** and explore how different preprocessing steps, model architectures, and hyperparameters affect model performance.


<u>Model Improvement Notes:</u>

### 1. Data Imputation

- **Change:** Median → Mean for handling missing values
- **Result:** Accuracy improved from ~70% → ~77%

---

### 2. Model Architecture

- **More complex model (~70% accuracy):**

```
nn.Sequential(
nn.Linear(8,16),
nn.ReLU(),
nn.Linear(16,16),
nn.ReLU(),
nn.Linear(16,8),
nn.ReLU(),
nn.Linear(8,1)
)
```

- **Simpler model (~75% accuracy):**

```
nn.Sequential(
nn.Linear(8,16),
nn.ReLU(),
nn.Linear(16,16),
nn.ReLU(),
nn.Linear(16,1)
)
```

Simpler architecture performed better, likely due to reduced overfitting on a small dataset.

---

### 3. Hyperparameter Tuning

### Epochs

- **1000 epochs** → ~65% (overfitting)
- **200 epochs** → ~74%
- **100 epochs** → ~79% (best performance)

Increasing epochs beyond a certain point led to overfitting and reduced test accuracy.


### Learning Rate

- **0.001** → ~70%
- **0.01** → ~74% (best)
- **0.1** → ~67% (unstable training)

---

## Key Takeaways

- Proper **data preprocessing** significantly impacts performance
- Simpler models can outperform deeper ones on smaller datasets
- **Overfitting** is a major concern — tuning epochs is critical
- Hyperparameters like learning rate strongly influence convergence
