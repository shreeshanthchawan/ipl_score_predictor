# IPL Next-Score Prediction — ES335

**Project question:** Can a batter's five most recent batting-appearance scores be used to estimate the runs scored in the next batting appearance?

## What the website does
The visitor can select a batter to load their five latest recorded scores, edit those values, or enter five scores manually. The website then runs either the Linear Regression or Random Forest model in the browser and displays an estimated next score. Both models use exactly five inputs, ordered from the oldest of the five appearances to the most recent.

The prediction target is the runs scored in the next recorded batting appearance. The data cannot predict whether a player will be selected or whether the player will get an opportunity to bat.

## Repository contents
- `train_models.py` — aggregation, sequence creation, chronological split, training and evaluation.
- `deliveries_updated_ipl_upto_2025.csv` — supplied raw ball-by-ball data.
- `matches_updated_ipl_upto_2025.csv` — supplied match-level file, retained as supporting source data.
- `outputs/` — processed tables, test predictions, metrics, saved models and browser-model exports.
- `site/` — static website files for local use and GitHub Pages.
- `AI_USE.md` — AI-use disclosure.
- `REFERENCES.md` — dataset attribution placeholder and software references.

## Rebuild the models
Use Python 3.10 or later.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages and run:
```bash
pip install -r requirements.txt
python train_models.py
```

The script reads `deliveries_updated_ipl_upto_2025.csv` from the project root. It creates one row per batter per match by summing `batsman_runs`. It then creates five lagged score columns for each batter. The five previous appearances are the input features; the current appearance's runs are the target. The first five appearances for each batter cannot form a supervised example.

The latest 20% of unique match dates are reserved for testing. Earlier dates are used for training. The Random Forest uses 150 trees, maximum depth 8, minimum leaf size 5 and random seed 42. No hyperparameter search was performed.

## Run the website locally
The site is static and runs without a Python prediction server. From the project root:

```bash
python -m http.server 8000 --directory site
```

Open `http://localhost:8000`.

The website loads `site/data.json` and `site/models.json`. The model parameters are exported from the trained scikit-learn models, so the predictions run in the browser rather than displaying fixed example numbers.

## Publish with GitHub Pages
1. Upload the project files to a GitHub repository.
2. Open **Settings → Pages**.
3. Select **Deploy from a branch**, choose `main`, and select `/site`.
4. Save and wait for deployment.
5. Open the published URL and test both model options and manual score entry.

The CSV may exceed GitHub's browser upload limit; use Git or GitHub Desktop if needed. The repository must be accessible to the instructor.

## Evaluation metrics
MAE is the average absolute difference between predicted and actual runs. RMSE gives larger errors more weight. R² compares the model with a constant-mean prediction on the test set.

## Limitations
The model only receives five previous scores. It does not use batting position, venue, opposition, pitch, toss, confirmed playing XI, injuries or match context. Scores from a player's next batting appearance are not necessarily from the immediately following IPL fixture if the player did not bat in an intervening match. The output is an estimate, not a guaranteed score.

## Before submission
Add the exact original dataset source URL and license/attribution terms to `REFERENCES.md`. Verify the website after publishing and include the live website URL and source repository URL in the submission.
