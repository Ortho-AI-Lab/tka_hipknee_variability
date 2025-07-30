from shared_vars import (
    project_dir,
    preprocessed_dataset,
    hip_features,
    knee_features,
    abridged_names,
    hip_features_with_spaces_to_underscores,
    knee_features_with_spaces_to_underscores,
)
import tablemage as tm


tm.options.print_options.mute()


regression_analyzer = tm.Analyzer(df=preprocessed_dataset, test_size=0.3)
regression_analyzer.impute(
    numeric_strategy="median", categorical_strategy="most_frequent"
)

confounders = [
    "Age",
    "Weight",
    "Sex",
]

confounders_with_spaces_to_underscores = [
    feature.replace(" ", "_") for feature in confounders
]


for hip_feature in hip_features_with_spaces_to_underscores:
    ols_result = regression_analyzer.ols(
        target=hip_feature,
        predictors=knee_features_with_spaces_to_underscores
        + confounders_with_spaces_to_underscores,
    )
    print(ols_result.metrics("test"))
    test_r2 = ols_result.metrics("test").loc["r2", "OLS Linear Model"]
    train_r2 = ols_result.metrics("train").loc["r2", "OLS Linear Model"]
    print(f"R2 for {hip_feature} (test): {test_r2:.4f}")
    print(f"R2 for {hip_feature} (train): {train_r2:.4f}")
    coefs_df = ols_result.coefs(format="coef|se|pval")
    coefs_df["Estimate"] = coefs_df["Estimate"].round(2)
    coefs_df["Std. Error"] = coefs_df["Std. Error"].round(2)
    coefs_df["Estimate (SE) [p-value]"] = (
        coefs_df["Estimate"].astype(str)
        + " ("
        + coefs_df["Std. Error"].astype(str)
        + ") ["
        + coefs_df["p-value"].map(lambda x: f"{x:.3f}")
        + "]"
    )
    coefs_df = coefs_df.sort_index(
        key=lambda x: x.str.startswith(("const", "Sex::M", "Age", "Weight")),
        ascending=False,
    )
    coefs_df = coefs_df[["Estimate (SE) [p-value]"]]
    to_save = project_dir / "results" / "tables" / "ols_results" / "coefficients"
    to_save.mkdir(parents=True, exist_ok=True)
    coefs_df.to_excel(
        to_save / f"regress_hip_on_knee-{hip_feature}-ols_coefs.xlsx",
    )

    regression_result = regression_analyzer.regress(
        models=[
            tm.ml.LinearR(type="ols", name="Ordinary Least Squares"),
            tm.ml.LinearR(type="l2", name="Ridge Regression"),
            tm.ml.LinearR(type="l1", name="Lasso Regression"),
            tm.ml.LinearR(type="elasticnet", name="Elastic Net"),
            tm.ml.TreesR(type="random_forest", name="Random Forest"),
        ],
        target=hip_feature,
        predictors=knee_features_with_spaces_to_underscores
        + confounders_with_spaces_to_underscores,
    )
    to_save = project_dir / "results" / "tables" / "regression_results"
    to_save.mkdir(parents=True, exist_ok=True)
    regression_result.metrics("test").to_excel(
        to_save / f"regress_hip_on_knee-{hip_feature}-test_metrics.xlsx",
    )
