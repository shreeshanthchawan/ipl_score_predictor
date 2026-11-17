# IPL Next-Score Prediction — ES335

## 1. Introduction

The objective of this project is to investigate whether a batter's recent batting performance can be used to estimate their score in their next batting appearance.

The main research question is:

**Can a batter's five most recent batting-appearance scores be used to estimate the runs scored in their next batting appearance?**

To investigate this, I developed a regression-based prediction system using IPL ball-by-ball data. The project involves data processing, feature creation, model training, evaluation and an interactive website.

## 2. Website Overview

I developed an interactive website where users can experiment with the prediction models.

Users can either select a batter to automatically load their five most recent recorded scores or enter five scores manually. The automatically populated scores can also be edited.

After entering the scores, users can select either Linear Regression or Random Forest and obtain an estimated score for the next batting appearance.

The five inputs are arranged chronologically, from the oldest appearance to the most recent one.

The prediction target is the next recorded batting appearance, not necessarily the immediately following IPL fixture. The model does not predict whether a player will be selected or get an opportunity to bat.

## 3. Repository Structure

| File/Folder | Description |
|---|---|
| `train_models.py` | Data processing, feature creation, model training and evaluation. |
| `matches_updated_ipl_upto_2025.csv` | Supporting match-level dataset. |
| `outputs/` | Processed datasets, test predictions, evaluation metrics and saved models. |
| `site/` | Website files used for local execution and GitHub Pages deployment. |
| `requirements.txt` | Required Python libraries. |
| `AI_USE.md` | Details of AI assistance during development. |
| `REFERENCES.md` | Dataset and software references. |

## 4. Data Processing and Feature Creation

The ball-by-ball data is aggregated to calculate the total runs scored by each batter in each match.

The appearances of each batter are then arranged chronologically.

Five previous batting-appearance scores are used as input features, while the score in the current appearance is used as the target variable.

The first five appearances of each batter cannot be used to create a training example because five previous scores are required.

## 5. Model Development

I implemented two regression models:

**Linear Regression**

This model estimates the relationship between the five previous scores and the score in the next batting appearance.

**Random Forest Regression**

This model combines predictions from multiple decision trees. The implementation uses 150 trees, a maximum depth of 8, a minimum leaf size of 5 and random seed 42.

A Mean Baseline model was also used for comparison. It provides a reference point to evaluate whether the trained models perform better than simply predicting the average target value.

The dataset was divided chronologically. The earliest 80% of unique match dates were used for training, while the latest 20% were reserved for testing.

No hyperparameter search was performed.

## 6. Model Evaluation

The models were evaluated using three metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual runs.
- **Root Mean Squared Error (RMSE):** Gives larger prediction errors more weight.
- **R² Score:** Measures how well the model performs compared with a constant-mean prediction.

The test results are:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean Baseline | 17.911 | 23.315 | -0.0062 |
| Linear Regression | 17.003 | 22.257 | 0.0831 |
| Random Forest | 16.940 | 22.293 | 0.0800 |

Random Forest achieved the lowest MAE, while Linear Regression had slightly better RMSE and R².

Both regression models performed better than the mean baseline on the reported metrics. However, the difference between the two models is small, and neither demonstrates particularly strong predictive performance.

This suggests that the five most recent scores contain some useful information, but they are not sufficient to explain most of the variation in a batter's next score.

## 7. Running the Project Locally

Python 3.10 or later is required.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python train_models.py
```

The script processes the input data and generates the processed datasets, evaluation results and model exports.

## 8. Running the Website

The website is static and does not require a separate Python prediction server.

From the project root, run:

```bash
python -m http.server 8000 --directory site
```

Open the following address in a browser:

`http://localhost:8000`

The website loads batter information from `data.json` and model parameters from `models.json`. Predictions are calculated directly in the browser using the exported model parameters.

## 9. GitHub Pages Deployment

The website can be hosted using GitHub Pages.

The deployment settings are:

- Source: Deploy from a branch
- Branch: `main`
- Folder: `/site`

After saving these settings and waiting for deployment, the website can be accessed through its GitHub Pages URL.

## 10. Limitations

The current models use only five previous batting-appearance scores.

They do not consider batting position, venue, opposition, pitch conditions, toss, confirmed playing XI, injuries or other match-specific factors.

A player's next batting appearance may not occur in the immediately following IPL fixture if the player does not bat in an intervening match.

The prediction should therefore be treated as an estimate based on historical scores rather than a guaranteed result.

## 11. AI Tool Usage

I used ChatGPT as a supporting tool during different stages of development.

The assistance included discussing the data-processing approach, getting suggestions for certain parts of the Python implementation, debugging coding issues and developing parts of the website interface. AI assistance was also used to organise and prepare the project documentation, including this README.

The project implementation, modelling approach and evaluation results were reviewed during development. The training script was executed on the dataset to generate the reported results.

AI tools were used to assist with selected development tasks rather than generate the entire project independently.
