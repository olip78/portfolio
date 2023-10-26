# Error Analysis Module

The Error Analysis Module provides a comprehensive set of tools for post-analysis of machine learning models. It includes the following features:

- **Statistical tests**: Normality (Shapiro–Wilk test),  Unbiasedness (One sample T-test), Homoscedasticity (Bartlett's, Levene's, Fligner-Killeen's tests).

- **Diagnostic plots**: Residuals vs. Fitted, Normal Q-Q Plot, Scale-Location Plot.

- **Residual fairness check**: Gini-index fairness, Best model selection for a given fairness.

- **Adversarial validation**: Adversarial validation for model error analysis. The function returns the ROC-AUC (on the full sample) and the features importance of the auxiliary model.

- **Residuals and pseudo-residuals calculation**: Analysis objects derived from different metrics such as MSE, MAE, MAPE, logloss, quantile_loss.