# Results summary

Supervised examples: 14,831. Training: 11,901. Test: 2,930. Test dates: 2023-04-05 to 2025-06-03.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | 17.911 | 23.315 | -0.0062 |
| Linear Regression | 17.003 | 22.257 | 0.0831 |
| Random Forest | 16.940 | 22.293 | 0.0800 |

The Random Forest has the lowest MAE in this test, while Linear Regression has a slightly lower RMSE and higher R². The difference is small. The results do not show that one model is clearly better on every metric. The mean baseline is weaker than both fitted models. The experiment is limited to five recent scores and does not include match context.
