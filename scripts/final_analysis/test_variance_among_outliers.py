from pathlib import Path
import pandas as pd
from scipy.stats import iqr, median_abs_deviation, mannwhitneyu
import numpy as np
from sklearn.decomposition import PCA
from shared_vars import hip_features, knee_features, axial_hip, axial_knee

curr_dir = Path(__file__).resolve().parent
project_dir = curr_dir.parent.parent

df = pd.read_csv(project_dir / "local_data" / "TKA-Hip-Rotation-Data.csv")
df = df.drop(
    columns=[
        "Measurements Recorded",
        "Height",
        "Subchondral Line Measurements",
        "Average Hounsfield Factor",
        "Medial Hounsfield Factor",
        "Lateral Hounsfield Factor",
    ]
)
df = df.map(lambda x: str(x) if isinstance(x, bool) else x)

# this is obtained from Woo et al (2024); we revisit their detected anatomical outliers
outlier_threshold_ids = pd.read_pickle(
    project_dir / "local_data" / "outlier_threshold_ids.pkl"
)
nine_through_27 = list(range(9, 28))
indices = []
for elem in nine_through_27:
    if elem in outlier_threshold_ids:
        indices.extend(outlier_threshold_ids[elem])
indices = list(set(indices))
df["is_outlier"] = df["Patient ID"].isin(indices)


# step 1: median imputation
def median_impute(df: pd.DataFrame, features):
    for feature in features:
        median_value = df[feature].median()
        df[feature] = df[feature].fillna(median_value)
    return df


df = median_impute(df, hip_features + knee_features)


# step 2: compute the 15th/85th percentile per metric
# this follows Woo et al (2024) in how they obtained the outlier threshold IDs


# -- uncomment for all hip and knee metrics --
all_metrics = hip_features + knee_features
metrics_name = "all_hip_and_knee"

# -- uncomment for axial hip and knee metrics --
# all_metrics = axial_hip + axial_knee
# hip_features = axial_hip
# knee_features = axial_knee
# metrics_name = "axial_hip_and_knee"


q15 = df[all_metrics].quantile(0.15)
q85 = df[all_metrics].quantile(0.85)
flags = (df[all_metrics] < q15) | (df[all_metrics] > q85)


# step 3: compute dispersion statistics, then test for feature-level differences
outliers_only_df = df[df["is_outlier"]].copy()


def compute_dispersions(features):
    variances = []
    iqrs = []
    mads = []
    for feat in features:
        vals = outliers_only_df[feat].dropna().values
        variances.append(vals.var(ddof=1))
        iqrs.append(iqr(vals))
        mads.append(median_abs_deviation(vals, scale="normal"))
    return pd.DataFrame(
        {"feature": features, "variance": variances, "IQR": iqrs, "MAD": mads}
    )


hip_disp = compute_dispersions(hip_features)
knee_disp = compute_dispersions(knee_features)


disp_df = pd.concat(
    [hip_disp.assign(region="Hip"), knee_disp.assign(region="Knee")], ignore_index=True
)
(project_dir / "results" / "variance_analysis").mkdir(parents=True, exist_ok=True)
disp_df.to_csv(
    project_dir
    / "results"
    / "variance_analysis"
    / f"outlier_dispersion-{metrics_name}.csv"
)

results = []
for metric in ["variance", "IQR", "MAD"]:
    stat, pval = mannwhitneyu(
        hip_disp[metric], knee_disp[metric], alternative="greater"
    )
    results.append({"metric": metric, "statistic": stat, "p_value": pval})

results_df = pd.DataFrame(results)
results_df.to_csv(
    project_dir
    / "results"
    / "variance_analysis"
    / f"mannwhitneyu_dispersion-{metrics_name}.csv",
    index=False,
)


# step 4: PCA variance comparison
n_out = len(outliers_only_df)

hip_mat = outliers_only_df[hip_features].values
knee_mat = outliers_only_df[knee_features].values

hip_centered = hip_mat - np.nanmean(hip_mat, axis=0)
knee_centered = knee_mat - np.nanmean(knee_mat, axis=0)

# fit PCA and extract variance metrics —
pca_hip = PCA().fit(hip_centered)
pca_knee = PCA().fit(knee_centered)

total_var_hip = np.sum(pca_hip.explained_variance_)
total_var_knee = np.sum(pca_knee.explained_variance_)
pc1_ratio_hip = pca_hip.explained_variance_ratio_[0]
pc1_ratio_knee = pca_knee.explained_variance_ratio_[0]

print("Hip total var:", f"{total_var_hip:.3f}")
print("Knee total var:", f"{total_var_knee:.3f}")
print("Hip/Knee var ratio:", f"{(total_var_hip/total_var_knee):.3f}")
print("Hip PC1 % var:", f"{pc1_ratio_hip:.2%}")
print("Knee PC1 % var:", f"{pc1_ratio_knee:.2%}")

# permutation test (shuffle feature‐to‐region labels)
all_feats = hip_features + knee_features
n_hip = len(hip_features)

n_perm = 1000
deltas = np.zeros(n_perm)
rng = np.random.default_rng(0)

for i in range(n_perm):
    # shuffle features, split back into two groups of size n_hip
    perm = rng.permutation(all_feats)
    h_feats = perm[:n_hip]
    k_feats = perm[n_hip:]

    H = outliers_only_df[h_feats].values - outliers_only_df[h_feats].values.mean(axis=0)
    K = outliers_only_df[k_feats].values - outliers_only_df[k_feats].values.mean(axis=0)

    ph = PCA().fit(H).explained_variance_.sum()
    pk = PCA().fit(K).explained_variance_.sum()
    deltas[i] = ph - pk

obs_delta = total_var_hip - total_var_knee
p_val = np.mean(deltas >= obs_delta)
ci_lower, ci_upper = np.percentile(deltas, [2.5, 97.5])

# Save permutation test results to a .txt file
with open(
    project_dir
    / "results"
    / "variance_analysis"
    / f"pca_permutation_test-{metrics_name}.txt",
    "w",
) as f:
    f.write(f"Observed Δ = {obs_delta:.3f}\n")
    f.write(f"{n_perm}-perm 95% CI for Δ: [{ci_lower:.3f}, {ci_upper:.3f}]\n")
    f.write(f"One‐tailed p (Hip > Knee) = {p_val:.3f}\n")
