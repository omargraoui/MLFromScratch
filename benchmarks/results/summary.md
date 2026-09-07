| Algorithm | Dataset | Metric | From scratch | scikit-learn | Absolute difference | Relative difference |
| :-- | :-- | :-- | --: | --: | --: | --: |
| LinearRegression(gd) | synthetic_regression | test_mse | 93.242 | 93.242 | 2.09727e-08 | 2.24927e-10 |
| LinearRegression(gd) | diabetes | test_mse | 2848.31 | 2848.31 | 1.96955e-06 | 6.91481e-10 |
| LogisticRegression | breast_cancer | test_accuracy | 0.986014 | 0.986014 | 0 | 0 |
| KMeans | iris | inertia | 139.82 | 139.82 | 8.52651e-14 | 6.09819e-16 |
| KMeans | wine | inertia | 1277.93 | 1277.93 | 9.09495e-13 | 7.11695e-16 |
| PCA | iris | reconstruction_mse | 0.0418679 | 0.0418679 | 6.93889e-18 | 1.65733e-16 |
| PCA | wine | reconstruction_mse | 0.198377 | 0.198377 | 5.55112e-17 | 2.79826e-16 |
| PCA | digits | reconstruction_mse | 0.00775101 | 0.00775101 | 8.67362e-19 | 1.11903e-16 |
