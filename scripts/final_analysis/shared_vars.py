from pathlib import Path
import pandas as pd


curr_dir = Path(__file__).parent.resolve()
project_dir = curr_dir.parent.parent


preprocessed_dataset = pd.read_csv(
    project_dir / "local_data" / "TKA-Hip-Rotation-Data.csv", index_col=0
)

hip_features = [
    "Acetabular Version",
    "Femoral Version",
    "Femoral Neck Angle",
    "Femoral Neck-Shaft Angle",
    "Femoral Torsion",
]

knee_features = [
    "Hip-Knee-Ankle Axis Angle",
    "Joint Line Convergence Angle",
    "Lateral Distal Femoral Angle",
    "Lateral Posterior Tibial Slope Angle",
    "Medial Posterior Tibial Slope Angle",
    "Posterior Condylar Axis Angle",
    "Sulcus Angle",
    "Tibiofemoral Angle",
    "Transepicondylar Axis Horizontal Angle",
    "Transepicondylar Axis-Posterior Condylar Axis Angle",
]

abridged_names = {
    "Hip-Knee-Ankle Axis Angle": "HKAA",
    "Joint Line Convergence Angle": "JLCA",
    "Lateral Distal Femoral Angle": "LDFA",
    "Posterior Condylar Axis Angle": "PCA",
    "Lateral Posterior Tibial Slope Angle": "Lateral PTS",
    "Medial Posterior Tibial Slope Angle": "Medial PTS",
    "Transepicondylar Axis Horizontal Angle": "TEA",
    "Transepicondylar Axis-Posterior Condylar Axis Angle": "TEA-PCA",
    "Tibiofemoral Angle": "TFA",
}

hip_features_with_spaces_to_underscores = [
    feature.replace(" ", "_") for feature in hip_features
]
knee_features_with_spaces_to_underscores = [
    feature.replace(" ", "_") for feature in knee_features
]


axial_hip = [
    "Acetabular Version",
    "Femoral Version",
    "Femoral Neck Angle",
    "Femoral Torsion",
]
axial_knee = [
    "Posterior Condylar Axis Angle",
    "Sulcus Angle",
    "Transepicondylar Axis Horizontal Angle",
    "Transepicondylar Axis-Posterior Condylar Axis Angle",
]
