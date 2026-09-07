# Generated benchmark results

Generated UTC: 2026-09-07T19:07:04.198265+00:00 · seed: 42 · Python: 3.13.5 · NumPy: 2.5.3 · scikit-learn: 1.9.0

Regenerate with `python -m benchmarks.run_all`. Full environment, settings, preprocessing, and convergence data are recorded in `results.json`. Relative difference is |ours − reference| / |reference|; — means a zero reference. Runtimes are single measurements with one BLAS thread and are not speed claims. Prediction RMSE and projector distance compare implementations directly and have zero as their ideal reference; reference ARI has one as its ideal.

| Algorithm | Dataset | Metric | From scratch | scikit-learn | Absolute difference | Relative difference |
| :-- | :-- | :-- | --: | --: | --: | --: |
| LinearRegression(gd) | synthetic_regression | test_mse | 93.242 | 93.242 | 2.09727e-08 | 2.24927e-10 |
| LinearRegression(gd) | synthetic_regression | test_r2 | 0.993494 | 0.993494 | 1.4635e-12 | 1.47308e-12 |
| LinearRegression(gd) | synthetic_regression | prediction_rmse_vs_reference | 5.03236e-08 | 0 | 5.03236e-08 | — |
| LinearRegression(gd) | synthetic_regression | fit_seconds | 0.001312 | 0.000956 | 0.000356 | 0.372385 |
| LinearRegression(normal) | synthetic_regression | test_mse | 93.242 | 93.242 | 4.26326e-14 | 4.57225e-16 |
| LinearRegression(normal) | synthetic_regression | test_r2 | 0.993494 | 0.993494 | 0 | 0 |
| LinearRegression(normal) | synthetic_regression | prediction_rmse_vs_reference | 7.84225e-14 | 0 | 7.84225e-14 | — |
| LinearRegression(normal) | synthetic_regression | fit_seconds | 0.0002778 | 0.000956 | 0.0006782 | 0.709414 |
| LinearRegression(gd) | diabetes | test_mse | 2848.31 | 2848.31 | 1.96955e-06 | 6.91481e-10 |
| LinearRegression(gd) | diabetes | test_r2 | 0.484906 | 0.484906 | 3.56178e-10 | 7.3453e-10 |
| LinearRegression(gd) | diabetes | prediction_rmse_vs_reference | 4.09992e-07 | 0 | 4.09992e-07 | — |
| LinearRegression(gd) | diabetes | fit_seconds | 0.445963 | 0.000778 | 0.445185 | 572.218 |
| LinearRegression(normal) | diabetes | test_mse | 2848.31 | 2848.31 | 4.54747e-13 | 1.59655e-16 |
| LinearRegression(normal) | diabetes | test_r2 | 0.484906 | 0.484906 | 1.11022e-16 | 2.28956e-16 |
| LinearRegression(normal) | diabetes | prediction_rmse_vs_reference | 5.23271e-14 | 0 | 5.23271e-14 | — |
| LinearRegression(normal) | diabetes | fit_seconds | 0.0002145 | 0.000778 | 0.0005635 | 0.724293 |
| LogisticRegression | breast_cancer | test_accuracy | 0.986014 | 0.986014 | 0 | 0 |
| LogisticRegression | breast_cancer | test_f1 | 0.988889 | 0.988889 | 0 | 0 |
| LogisticRegression | breast_cancer | test_log_loss | 0.0854248 | 0.0854248 | 6.29016e-10 | 7.36339e-09 |
| LogisticRegression | breast_cancer | probability_max_abs_difference | 1.55925e-07 | 0 | 1.55925e-07 | — |
| LogisticRegression | breast_cancer | fit_seconds | 0.416903 | 0.0067829 | 0.41012 | 60.4638 |
| KMeans | iris | inertia | 139.82 | 139.82 | 8.52651e-14 | 6.09819e-16 |
| KMeans | iris | ari_vs_known_labels | 0.620135 | 0.620135 | 0 | 0 |
| KMeans | iris | ari_vs_reference | 1 | 1 | 0 | 0 |
| KMeans | iris | iterations | 5 | 4 | 1 | 0.25 |
| KMeans | iris | fit_seconds | 0.0093088 | 0.028908 | 0.0195992 | 0.677985 |
| KMeans | wine | inertia | 1277.93 | 1277.93 | 9.09495e-13 | 7.11695e-16 |
| KMeans | wine | ari_vs_known_labels | 0.897495 | 0.897495 | 0 | 0 |
| KMeans | wine | ari_vs_reference | 1 | 1 | 0 | 0 |
| KMeans | wine | iterations | 2 | 7 | 5 | 0.714286 |
| KMeans | wine | fit_seconds | 0.0092674 | 0.0082427 | 0.0010247 | 0.124316 |
| PCA | iris | explained_variance_ratio_sum | 0.958132 | 0.958132 | 0 | 0 |
| PCA | iris | reconstruction_mse | 0.0418679 | 0.0418679 | 6.93889e-18 | 1.65733e-16 |
| PCA | iris | projector_frobenius_distance | 8.18672e-16 | 0 | 8.18672e-16 | — |
| PCA | iris | fit_seconds | 0.0001739 | 0.0004874 | 0.0003135 | 0.643209 |
| PCA | wine | explained_variance_ratio_sum | 0.801623 | 0.801623 | 1.11022e-16 | 1.38497e-16 |
| PCA | wine | reconstruction_mse | 0.198377 | 0.198377 | 5.55112e-17 | 2.79826e-16 |
| PCA | wine | projector_frobenius_distance | 4.45037e-15 | 0 | 4.45037e-15 | — |
| PCA | wine | fit_seconds | 0.0002113 | 0.0004706 | 0.0002593 | 0.550999 |
| PCA | digits | explained_variance_ratio_sum | 0.894303 | 0.894303 | 2.22045e-16 | 2.48288e-16 |
| PCA | digits | reconstruction_mse | 0.00775101 | 0.00775101 | 8.67362e-19 | 1.11903e-16 |
| PCA | digits | projector_frobenius_distance | 2.6972e-14 | 0 | 2.6972e-14 | — |
| PCA | digits | fit_seconds | 0.0059953 | 0.0068029 | 0.0008076 | 0.118714 |
