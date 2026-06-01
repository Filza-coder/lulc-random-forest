"""
Land Use Land Cover (LULC) Classification with Random Forest
=============================================================
Multi-class land cover mapping from Sentinel-2 multispectral imagery using
Random Forest — one of the most widely-used approaches in remote sensing ML.

Classes: Water | Urban/Built-up | Vegetation | Bare Soil | Agriculture | Forest
Data:    Synthetic spectral signatures (real workflow mirrors Google Earth Engine export)

Author: Fatima Filza Hassan | github.com/Filza-coder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SENTINEL-2 BAND DEFINITIONS
# ─────────────────────────────────────────────
S2_BANDS = ["B2_Blue", "B3_Green", "B4_Red", "B5_RE1", "B6_RE2",
            "B7_RE3", "B8_NIR", "B8A_NIR2", "B11_SWIR1", "B12_SWIR2"]

# Spectral signatures per class (mean reflectance × 10000, realistic S2 values)
CLASS_SIGNATURES = {
    "Water":       [500,  600,  400,  300,  200,  150,  100,   80,   50,   30],
    "Urban":       [1800, 1900, 2000, 2100, 2200, 2100, 2300, 2250, 2400, 2200],
    "Vegetation":  [600,  900, 700,  2200, 3000, 3200, 4000, 4100, 2000, 1200],
    "Bare Soil":   [2000, 2200, 2400, 2500, 2600, 2550, 2700, 2650, 2800, 2700],
    "Agriculture": [900,  1200, 1000, 2800, 3500, 3700, 4500, 4600, 2200, 1400],
    "Forest":      [400,  700,  500,  2000, 2800, 3000, 4200, 4300, 1800, 1000],
}
COLORS = {
    "Water": "#1e88e5", "Urban": "#e53935", "Vegetation": "#43a047",
    "Bare Soil": "#fb8c00", "Agriculture": "#8bc34a", "Forest": "#1b5e20"
}

# ─────────────────────────────────────────────
# 1.  GENERATE SYNTHETIC TRAINING SAMPLES
# ─────────────────────────────────────────────
np.random.seed(42)
SAMPLES_PER_CLASS = 500

rows = []
for cls, sig in CLASS_SIGNATURES.items():
    noise = np.random.normal(0, 150, (SAMPLES_PER_CLASS, len(S2_BANDS)))
    vals  = np.array(sig) + noise
    vals  = np.clip(vals, 0, 10000)
    df_c  = pd.DataFrame(vals, columns=S2_BANDS)
    df_c["class"] = cls
    rows.append(df_c)

df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=42)

# Derived indices
df["NDVI"]  = (df["B8_NIR"] - df["B4_Red"])   / (df["B8_NIR"] + df["B4_Red"]   + 1e-6)
df["NDWI"]  = (df["B3_Green"] - df["B8_NIR"]) / (df["B3_Green"] + df["B8_NIR"] + 1e-6)
df["NDBI"]  = (df["B11_SWIR1"] - df["B8_NIR"])/ (df["B11_SWIR1"] + df["B8_NIR"]+ 1e-6)
df["EVI"]   = 2.5 * (df["B8_NIR"] - df["B4_Red"]) / (df["B8_NIR"] + 6*df["B4_Red"] - 7.5*df["B2_Blue"] + 10000)

FEATURES = S2_BANDS + ["NDVI", "NDWI", "NDBI", "EVI"]
X = df[FEATURES]
y = df["class"]

print(f"Dataset: {len(df)} samples × {len(FEATURES)} features")
print("Class distribution:\n", y.value_counts().to_string())

# ─────────────────────────────────────────────
# 2.  TRAIN / TEST SPLIT  +  RF CLASSIFIER
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_leaf=2,
    n_jobs=-1, random_state=42, oob_score=True
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy : {acc:.4f}")
print(f"OOB Score        : {rf.oob_score_:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# 3.  VISUALISATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("LULC Classification — Random Forest on Sentinel-2 Data",
             fontsize=14, fontweight="bold")

# A) Spectral signatures
ax = axes[0, 0]
band_labels = ["B2\nBlue", "B3\nGrn", "B4\nRed", "B5\nRE1", "B6\nRE2",
               "B7\nRE3", "B8\nNIR", "B8A", "B11\nSWIR1", "B12\nSWIR2"]
for cls, sig in CLASS_SIGNATURES.items():
    ax.plot(range(len(S2_BANDS)), sig, label=cls, color=COLORS[cls], linewidth=2, marker="o", markersize=4)
ax.set_xticks(range(len(S2_BANDS))); ax.set_xticklabels(band_labels, fontsize=8)
ax.set_title("Sentinel-2 Spectral Signatures by Class"); ax.set_ylabel("Reflectance (×10⁴)")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

# B) Confusion matrix
ax = axes[0, 1]
cm = confusion_matrix(y_test, y_pred, labels=list(CLASS_SIGNATURES.keys()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(CLASS_SIGNATURES.keys()))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix  (OA={acc:.3f})")
ax.tick_params(axis="x", rotation=30)

# C) Feature importance
ax = axes[1, 0]
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
colors_bar = ["#1b5e20" if "ND" in f or "EVI" in f or "BI" in f else "#42a5f5" for f in imp.index]
ax.barh(imp.index, imp.values, color=colors_bar, edgecolor="white")
ax.set_title("Feature Importance"); ax.set_xlabel("Importance Score")
ax.axvline(imp.values.mean(), color="red", linestyle="--", alpha=0.6, label="Mean")
ax.legend()

# D) Simulated classified image
ax = axes[1, 1]
np.random.seed(99)
grid_size = 80
class_list = list(CLASS_SIGNATURES.keys())
weights = [0.10, 0.20, 0.25, 0.15, 0.20, 0.10]
class_grid_labels = np.random.choice(class_list, size=(grid_size, grid_size), p=weights)
# Apply some spatial smoothing for realism
from scipy.ndimage import uniform_filter
class_ids = np.array([[class_list.index(c) for c in row] for row in class_grid_labels])
class_ids_smooth = np.round(uniform_filter(class_ids.astype(float), size=4)).astype(int)
color_array = np.array([[mcolors.to_rgb(COLORS[class_list[v]]) for v in row] for row in class_ids_smooth])
ax.imshow(color_array, aspect="auto")
ax.set_title("Simulated LULC Classification Output")
ax.set_xlabel("Pixel Column (East →)"); ax.set_ylabel("Pixel Row (North ↓)")
patches = [mpatches.Patch(color=COLORS[c], label=c) for c in class_list]
ax.legend(handles=patches, loc="lower right", fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig("lulc_classification_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: lulc_classification_results.png")

# ─────────────────────────────────────────────
# 4.  CROSS-VALIDATION
# ─────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 5.  ACCURACY ASSESSMENT TABLE  (per-class)
# ─────────────────────────────────────────────
report = classification_report(y_test, y_pred, output_dict=True)
acc_df = pd.DataFrame(report).T.round(3)
print("\nPer-class Accuracy Assessment:")
print(acc_df[["precision", "recall", "f1-score", "support"]].head(6).to_string())
acc_df.to_csv("accuracy_assessment.csv")
print("Saved: accuracy_assessment.csv")
print("\n✅ LULC Classification complete.")
