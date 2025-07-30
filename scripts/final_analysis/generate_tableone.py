import tableone

from shared_vars import (
    project_dir,
    preprocessed_dataset,
    hip_features,
    knee_features,
    abridged_names,
)


preprocessed_dataset_renamed = preprocessed_dataset.rename(
    columns={
        feature: abridged_names.get(feature, feature)
        for feature in preprocessed_dataset.columns
    }
)

knee_features_renamed = [
    abridged_names.get(feature, feature) for feature in knee_features
]
preprocessed_dataset_renamed.columns = preprocessed_dataset_renamed.columns.astype(str)
preprocessed_dataset_renamed.columns = preprocessed_dataset_renamed.columns.str.strip()

expected_columns = [
    "Femoral Version",
    "Femoral Neck-Shaft Angle",
    "Femoral Neck Angle",
    "Age",
    "Femoral Torsion",
    "Varus vs Valgus",
    "Acetabular Version",
    "Sex",
    "Weight",
    "Femoral Head Width",
]
missing = [
    col for col in expected_columns if col not in preprocessed_dataset_renamed.columns
]
print("Missing columns:", missing)

table_overall = tableone.TableOne(
    data=preprocessed_dataset_renamed,
    columns=["Weight", "Age", "Varus vs Valgus", "Sex"]
    + hip_features
    + knee_features_renamed,
    categorical=["Varus vs Valgus", "Sex"],
    continuous=["Weight", "Age"] + hip_features + knee_features_renamed,
    decimals=2,
)

table_overall.to_excel(
    project_dir / "results" / "tables" / "tableone_no_stratification.xlsx",
)


table_sex = tableone.TableOne(
    data=preprocessed_dataset_renamed,
    columns=["Weight", "Age", "Varus vs Valgus", "Sex"]
    + hip_features
    + knee_features_renamed,
    categorical=[
        "Varus vs Valgus",
    ],
    continuous=["Weight", "Age"] + hip_features + knee_features_renamed,
    decimals=2,
    groupby="Sex",
    overall=True,
    htest=True,
)

table_sex.to_excel(
    project_dir / "results" / "tables" / "tableone_sex.xlsx",
)


table_varus = tableone.TableOne(
    data=preprocessed_dataset_renamed,
    columns=["Weight", "Age", "Varus vs Valgus", "Sex"]
    + hip_features
    + knee_features_renamed,
    categorical=[
        "Sex",
    ],
    continuous=["Weight", "Age"] + hip_features + knee_features_renamed,
    decimals=2,
    groupby="Varus vs Valgus",
    overall=True,
    htest=True,
)

table_varus.to_excel(
    project_dir / "results" / "tables" / "tableone_varus.xlsx",
)
