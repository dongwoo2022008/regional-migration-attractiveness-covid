## Appendix Table A3. Hyperparameter Search Spaces and Optimal Values

| Algorithm | Parameter | Type | Range | Scale | Best Value |
|-----------|-----------|------|-------|-------|------------|
| DT | `max_depth` | int | [3.0, 15.0] | linear | 5 |
|  | `min_samples_split` | int | [2.0, 30.0] | linear | 11 |
|  | `min_samples_leaf` | int | [1.0, 20.0] | linear | 13 |
| RF | `n_estimators` | int | [200.0, 800.0] | linear | 350 |
|  | `max_depth` | int | [3.0, 20.0] | linear | 20 |
|  | `min_samples_leaf` | int | [1.0, 20.0] | linear | 1 |
|  | `max_features` | float | [0.4, 1.0] | linear | 0.572592 |
| GB | `n_estimators` | int | [200.0, 600.0] | linear | 587 |
|  | `learning_rate` | float | [0.01, 0.3] | log | 0.0146427 |
|  | `max_depth` | int | [2.0, 6.0] | linear | 6 |
|  | `subsample` | float | [0.6, 1.0] | linear | 0.983985 |
| XGB | `n_estimators` | int | [200.0, 1000.0] | linear | 658 |
|  | `learning_rate` | float | [0.01, 0.3] | log | 0.0109944 |
|  | `max_depth` | int | [3.0, 12.0] | linear | 11 |
|  | `subsample` | float | [0.6, 1.0] | linear | 0.655344 |
|  | `colsample_bytree` | float | [0.6, 1.0] | linear | 0.648433 |
|  | `reg_lambda` | float | [0.0, 5.0] | linear | 4.09454 |
| LGBM | `n_estimators` | int | [200.0, 1000.0] | linear | 988 |
|  | `learning_rate` | float | [0.01, 0.3] | log | 0.055855 |
|  | `num_leaves` | int | [15.0, 127.0] | linear | 68 |
|  | `feature_fraction` | float | [0.6, 1.0] | linear | 0.706963 |
|  | `bagging_fraction` | float | [0.6, 1.0] | linear | 0.855563 |
|  | `bagging_freq` | int | [1.0, 10.0] | linear | 1 |
|  | `reg_lambda` | float | [0.0, 5.0] | linear | 3.66747 |
| CatBoost | `iterations` | int | [200.0, 800.0] | linear | 677 |
|  | `learning_rate` | float | [0.01, 0.3] | log | 0.108849 |
|  | `depth` | int | [4.0, 10.0] | linear | 8 |
|  | `l2_leaf_reg` | float | [1.0, 10.0] | linear | 2.15247 |

*Note.* All algorithms used Optuna TPE sampler (seed = 42, N = 80 trials per algorithm).
LR (Ridge, α = 1.0) was not tuned and served as the baseline.
The objective function was the mean expanding-window validation R² across three folds
(train 2017–2019/val 2020; train 2017–2020/val 2021; train 2017–2021/val 2022).
Best values are from `outputs_reference/table_A3_best_hyperparams.json`.
