# Physics-Informed Reactor Yield Prediction

A physics-informed machine learning approach for predicting the overall yield of Product B in a non-isothermal continuous-flow reactor.

## Overview

This project was developed for the **ML Hackathon: The Predictive Modeling Optimization Challenge**, where the objective was to predict the overall yield of Product B under unseen reactor operating conditions.

The reactor follows a series-parallel reaction network:

**A → B → C**

where B is the desired product and C is the undesired side-product.

The challenge is highly nonlinear because reactor yield depends on operating conditions such as flow rate, residence time, inlet temperature, and jacket temperature. With only **150 labeled training samples**, purely data-driven models can struggle to learn these physical relationships reliably.

To address this, this project combines:

* A **physics-based kinetic model**
* Physics-inspired feature engineering
* Machine learning models for residual correction
* Cross-validation and hyperparameter tuning
* A final **hybrid physics + ML model**

---

## Problem Statement

The goal of the hackathon was to build a predictive surrogate model capable of estimating reactor performance without repeatedly solving computationally expensive physics-based simulations.

The available data contains five operating variables:

| Feature                | Description                                  |
| ---------------------- | -------------------------------------------- |
| `flow_rate_L_min`      | Volumetric flow rate of the reactant mixture |
| `concentration_mol_L`  | Inlet concentration of Reactant A            |
| `inlet_temperature_K`  | Feed inlet temperature                       |
| `length_m`             | Reactor length                               |
| `jacket_temperature_K` | External heating jacket temperature          |

### Target

`overall_yield` — final percentage yield of Product B at the reactor exit.

The official problem statement provided **150 labeled training samples** and **50 unseen test samples**.

---

## Approach

### 1. Exploratory Data Analysis

The training data was analyzed to understand:

* Feature distributions
* Missing values
* Yield distribution
* High- and low-yield operating regions
* Temperature and residence-time effects

The dataset contained no missing values.

The training set contained **37 zero-yield observations** and **13 observations with yield above 95%**, highlighting the highly nonlinear behavior of the reactor.

---

### 2. Physics-Based Modeling

Instead of treating the problem as a completely black-box regression task, the reactor chemistry was explicitly modeled.

The model assumes the series-parallel reaction:

**A → B → C**

and uses temperature-dependent reaction kinetics based on Arrhenius relationships.

A residence-time term was introduced as:

```text
tau = reactor_length / flow_rate
```

The physics model estimates reactor yield using five fitted parameters:

* `ln(A1)`
* `Ea1/R`
* `ln(A2)`
* `Ea2/R`
* Thermal time constant

The parameters were fitted using global optimization with Differential Evolution.

The fitted model achieved an in-sample RMSE of approximately **14.64**.

An important physical insight was that:

```text
Ea2/R > Ea1/R
```

indicating that the side reaction consuming Product B is more temperature-sensitive than the desired reaction.

---

## 3. Physics-Informed Feature Engineering

Additional features were created to capture the physical behavior of the reactor:

* Residence time (`tau`)
* Average temperature (`avg_T`)
* Temperature difference (`delta_T`)
* Log residence time (`log_tau`)
* Inverse jacket temperature (`inv_jacketT`)
* Inverse inlet temperature (`inv_inletT`)

These features allow the ML models to work with physically meaningful representations rather than only the raw process variables.

---

## 4. Model Comparison

Several approaches were evaluated using **5-fold × 5-repeat cross-validation**, resulting in 25 validation splits.

The physics model was re-fitted inside every training fold to prevent data leakage.

Models compared:

* Physics-only model
* Ridge Regression
* Random Forest
* Gradient Boosting
* XGBoost
* Hybrid Physics + Gradient Boosting
* Hybrid Physics + XGBoost
* Hybrid ensemble

### Cross-Validation Results

| Model                    | Mean CV RMSE |
| ------------------------ | -----------: |
| Hybrid XGBoost           |       13.452 |
| Hybrid Ensemble          |       13.531 |
| Hybrid Gradient Boosting |       13.821 |
| Physics-only             |       15.381 |
| XGBoost                  |       20.074 |
| Gradient Boosting        |       20.119 |
| Random Forest            |       21.630 |
| Ridge Regression         |       29.482 |

The results demonstrate that incorporating the reactor physics substantially improves generalization compared with purely black-box ML models.

---

## 5. Hybrid Modeling

The final modeling strategy predicts the reactor yield in two stages:

### Stage 1 — Physics Prediction

The kinetic model produces an initial physically grounded prediction:

```text
Physics Prediction
```

### Stage 2 — ML Residual Correction

The ML models learn the difference between the observed yield and the physics-model prediction:

```text
Residual = Actual Yield - Physics Prediction
```

The final prediction is then:

```text
Final Prediction =
Physics Prediction + ML Residual Correction
```

This allows the physics model to capture the dominant nonlinear chemical behavior while machine learning captures deviations that the simplified physical model cannot explain.

---

## 6. Hyperparameter Tuning

The hybrid Gradient Boosting and XGBoost residual models were further tuned using cross-validation.

### Best Gradient Boosting Configuration

```text
n_estimators = 200
max_depth = 4
learning_rate = 0.05
subsample = 0.8
```

Cross-validated RMSE:

```text
13.384
```

### Best XGBoost Configuration

```text
n_estimators = 300
max_depth = 4
learning_rate = 0.03
subsample = 0.8
colsample_bytree = 0.8
```

Cross-validated RMSE:

```text
13.491
```

The tuned hybrid Gradient Boosting model achieved the lowest cross-validated RMSE among the tuned configurations.

---

## 7. Final Prediction

The final tuned hybrid model was trained using the complete training dataset.

Predictions were generated for the **50 unseen test conditions** and constrained to the physically meaningful range:

```text
0 ≤ overall_yield ≤ 100
```

The final submission contained:

* 50 predictions
* One column: `overall_yield`
* Predictions rounded to three decimal places

This matches the hackathon submission requirements.

---

## Key Physical Insight

One of the most important observations from the modeling process was the **yield-volcano behavior**.

Product B is formed through the desired reaction but can subsequently be consumed by the side reaction.

Therefore, increasing temperature does not simply increase yield.

At sufficiently high temperatures, both reactions become fast, and the side reaction increasingly consumes Product B. This creates an optimal operating region where the desired reaction is favored while excessive conversion of B to C is avoided.

Similarly, residence time creates a nonlinear trade-off:

* Too little residence time → insufficient formation of B
* Optimal residence time → maximum B yield
* Excessive residence time → increased conversion of B to C

This physical structure is difficult for a purely black-box model to learn reliably from only 150 samples.

---

## Why a Physics-Informed Model?

A purely data-driven approach can fit the training data well but may struggle to generalize when the dataset is small.

The hybrid approach provides:

**Physics → dominant nonlinear behavior**

**Machine Learning → residual corrections**

This improves interpretability while reducing the burden on ML models to rediscover known chemical relationships from limited data.

The hackathon evaluation explicitly emphasized process insight, feature engineering, robustness, and scalability in addition to predictive accuracy.

---

## Repository Structure

```text
physics-informed-reactor-yield-prediction/
│
├── The_Insight_Pair.ipynb
├── data/
│   ├── train_dataset.csv
│   └── test_dataset.csv
│
├── predictions/
│   └── submission.csv
│
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* SciPy
* Scikit-learn
* XGBoost
* Jupyter Notebook

---

## Key Takeaways

* Incorporated **reaction kinetics and physical constraints** into ML modeling.
* Engineered physically meaningful features such as residence time and temperature relationships.
* Compared multiple ML and physics-based approaches using repeated cross-validation.
* Used **residual learning** to combine mechanistic modeling with machine learning.
* Performed hyperparameter tuning for the hybrid models.
* Achieved a best tuned hybrid CV RMSE of **13.384**.
* Generated predictions for all **50 unseen reactor conditions**.

---

## Hackathon

**ML Hackathon: The Predictive Modeling Optimization Challenge**

The competition focused on building a data-driven surrogate model for fast reactor performance prediction. The official evaluation used RMSE on 50 hidden test cases, while the final judging also considered process understanding, innovation, robustness, and scalability.

---

## Author

**The Insight Pair**

Built as part of the ML Hackathon.
