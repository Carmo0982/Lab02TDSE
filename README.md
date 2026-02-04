# Heart Disease Risk Prediction: Logistic Regression

Implements logistic regression from scratch for heart disease prediction using the Kaggle Heart Disease Dataset (270 patients; features: Age 29-77, Cholesterol 112-564 mg/dL, BP 94-200 mm Hg, Max HR 71-202 bpm, ST depression 0-6.2, Vessels 0-3; ~55% disease presence rate). The project covers exploratory data analysis, model training with gradient descent, decision boundary visualization, L2 regularization tuning, and Amazon SageMaker deployment for real-time inference.

## Getting Started

These instructions will help you run the heart disease prediction notebook on your local machine and optionally deploy the model to Amazon SageMaker for production use.

### Prerequisites

You need Python 3.8+ and the following libraries:

```bash
Python >= 3.8
NumPy >= 1.21
Pandas >= 1.3
Matplotlib >= 3.4
Jupyter Notebook or JupyterLab
```

For SageMaker deployment (optional):

```bash
AWS Account with SageMaker access
boto3
sagemaker SDK
```

### Installing

Step 1: Clone or download this repository

```bash
git clone https://github.com/your-username/Lab02TDSE.git
cd Lab02TDSE
```

Step 2: Install Python dependencies

```bash
pip install numpy pandas matplotlib jupyter
```

Step 3: Download the dataset from Kaggle

```bash
# Visit https://www.kaggle.com/datasets/neurocipher/heartdisease
# Download Heart_Disease_Prediction.csv and place it in the project folder
```

Step 4: Launch Jupyter Notebook

```bash
jupyter notebook heart_disease_lr_analysis.ipynb
```

Step 5: Run all cells sequentially to reproduce the analysis

```
Cell 1-10: Data loading and EDA
Cell 11-25: Logistic regression implementation
Cell 26-38: Decision boundary visualization
Cell 39-49: L2 regularization
Cell 50-60: SageMaker deployment (optional)
```

End result: You should see cost convergence plots, decision boundaries for 3 feature pairs, and model performance metrics (~80% test accuracy).

## Running the Tests

The notebook includes built-in validation tests to ensure correct implementation.

### Model Convergence Tests

Verify that gradient descent converges properly by checking the cost function plot:

```python
# After training, verify cost decreases monotonically
plt.plot(J_history)
plt.title("Cost should decrease monotonically")
# Expected: Smooth decreasing curve from ~0.69 to ~0.45
```

### Gradient Verification Tests

Test gradient computation at zero weights to verify implementation:

```python
# At w=0, b=0 with balanced classes:
# Expected cost: -ln(0.5) ≈ 0.693
# Expected dJ/db: close to 0 for balanced classes
w_test = np.zeros(6)
b_test = 0.0
J = compute_cost(w_test, b_test, X_train_norm, y_train)
print(f"Cost at zero: {J:.4f}")  # Should be ~0.693
```

### Prediction Accuracy Tests

Validate model predictions on known cases:

```python
# High-risk patient (expect probability > 0.5)
high_risk = [65, 160, 350, 120, 3.5, 2]  # Age, BP, Chol, HR, ST, Vessels
# Low-risk patient (expect probability < 0.5)
low_risk = [35, 110, 180, 180, 0, 0]
```

## Deployment

### Amazon SageMaker Deployment

To deploy the model on AWS SageMaker:

**Step 1:** Export the trained model

```python
import json
model_data = {
    'w': w_trained.tolist(),
    'b': float(b_trained),
    'mean': mu.tolist(),
    'std': sigma.tolist()
}
with open('model/model.json', 'w') as f:
    json.dump(model_data, f)
```

**Step 2:** Create the inference script (`inference.py`)

```python
def model_fn(model_dir):
    # Load model from JSON

def predict_fn(input_data, model):
    # Normalize input, compute sigmoid, return prediction
```

**Step 3:** Package and deploy

```python
import sagemaker
from sagemaker.sklearn import SKLearnModel

# Upload to S3
model_uri = session.upload_data('model.tar.gz')

# Deploy endpoint
predictor = SKLearnModel(
    model_data=model_uri,
    entry_point='inference.py'
).deploy(instance_type='ml.t2.medium')
```

**Step 4:** Test the endpoint

```python
response = predictor.predict([60, 130, 300, 150, 2.0, 1])
# Output: {"probability": 0.78, "prediction": 1, "label": "Presence"}
```

**⚠️ Important:** Delete the endpoint after testing to avoid charges:

```python
predictor.delete_endpoint()
```

## Built With

- [NumPy](https://numpy.org/) - Numerical computing and matrix operations
- [Pandas](https://pandas.pydata.org/) - Data loading and manipulation
- [Matplotlib](https://matplotlib.org/) - Visualization and plotting
- [Jupyter Notebook](https://jupyter.org/) - Interactive development environment
- [Amazon SageMaker](https://aws.amazon.com/sagemaker/) - Cloud deployment platform

## Dataset Description

**Source:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/neurocipher/heartdisease)

| Attribute    | Value                     |
| ------------ | ------------------------- |
| Samples      | 270 patient records       |
| Features     | 14 clinical attributes    |
| Target       | Binary (Presence/Absence) |
| Disease Rate | ~55% presence             |

### Selected Features

| Feature           | Range         | Clinical Significance                        |
| ----------------- | ------------- | -------------------------------------------- |
| Age               | 29-77 years   | Cardiovascular risk increases with age       |
| BP                | 94-200 mm Hg  | Hypertension damages arteries                |
| Cholesterol       | 112-564 mg/dL | Leads to atherosclerosis                     |
| Max HR            | 71-202 bpm    | Lower max HR indicates poor cardiac function |
| ST depression     | 0-6.2         | ECG abnormality indicating ischemia          |
| Number of vessels | 0-3           | Direct measure of arterial blockage          |

## Model Results

### Performance Metrics

| Metric    | Train | Test |
| --------- | ----- | ---- |
| Accuracy  | 83%   | 80%  |
| Precision | 85%   | 82%  |
| Recall    | 86%   | 84%  |
| F1 Score  | 85%   | 83%  |

### Regularization Tuning

| λ        | Test Accuracy | ‖w‖      | Observation  |
| -------- | ------------- | -------- | ------------ |
| 0        | 0.802         | 2.15     | Baseline     |
| **0.01** | **0.815**     | **2.08** | **Optimal**  |
| 1        | 0.728         | 0.98     | Underfitting |

**Optimal λ = 0.01** improves test accuracy by ~1.5%.

## Contributing

This is an academic project for TDSE (Eight Semester).

## Versioning

- **v1.0** - Initial implementation with Steps 1-5 complete

## Authors

- **Santiago Carmona**

## License

This project is for educational purposes as part of university coursework.

## Acknowledgments

- UCI Machine Learning Repository for the original Heart Disease dataset
- Kaggle for hosting the dataset
- AWS for SageMaker free tier resources
- Course instructors for the logistic regression theory notebooks
