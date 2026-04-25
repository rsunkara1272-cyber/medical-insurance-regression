"""
Regression – Medical Insurance Charges
=======================================
Dataset : Medical Cost Personal Dataset (public)
          Auto-downloaded from GitHub mirror.
Model   : Gradient Boosting Regressor (with baseline Linear Regression)
Plots   : Predicted vs Actual, Residuals, Feature Importance, Smoker Impact
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import logging, os, urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT_DIR = "outputs/regression"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

STYLE = {
    "bg":      "#0F172A",
    "panel":   "#1E293B",
    "accent1": "#38BDF8",
    "accent2": "#F97316",
    "accent3": "#4ADE80",
    "text":    "#F1F5F9",
    "subtext": "#94A3B8",
}

def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  STYLE["bg"],
        "axes.facecolor":    STYLE["panel"],
        "axes.edgecolor":    STYLE["subtext"],
        "axes.labelcolor":   STYLE["text"],
        "xtick.color":       STYLE["subtext"],
        "ytick.color":       STYLE["subtext"],
        "text.color":        STYLE["text"],
        "grid.color":        "#334155",
        "grid.alpha":        0.5,
        "font.family":       "DejaVu Sans",
    })

# ── Data ──────────────────────────────────────

def load_data() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/medical_cost.csv"
    local = "insurance.csv"
    try:
        urllib.request.urlretrieve(url, local)
        df = pd.read_csv(local)
        log.info("Downloaded insurance dataset  (%d rows)", len(df))
    except Exception:
        log.info("Network unavailable – generating synthetic insurance data")
        df = _synthetic_insurance(1338)
    return df

def _synthetic_insurance(n=1338, seed=42):
    rng = np.random.default_rng(seed)
    age    = rng.integers(18, 65, n)
    sex    = rng.choice(["male","female"], n)
    bmi    = rng.normal(30.7, 6.1, n).clip(15, 55).round(2)
    children = rng.integers(0, 5, n)
    smoker = rng.choice(["yes","no"], n, p=[0.2, 0.8])
    region = rng.choice(["northeast","northwest","southeast","southwest"], n)
    charges = (
        age * 250
        + bmi * 150
        + children * 500
        + (smoker == "yes") * rng.normal(23000, 3000, n)
        + rng.normal(3000, 1500, n)
    ).clip(1000).round(2)
    return pd.DataFrame(dict(age=age, sex=sex, bmi=bmi, children=children,
                              smoker=smoker, region=region, charges=charges))

def preprocess(df: pd.DataFrame):
    df = df.copy()
    le = LabelEncoder()
    df["sex"]    = le.fit_transform(df["sex"])
    df["smoker"] = le.fit_transform(df["smoker"])
    df = pd.get_dummies(df, columns=["region"], drop_first=True)
    X = df.drop("charges", axis=1)
    y = df["charges"]
    return X, y

# ── Models ────────────────────────────────────

def train_models(X_train, y_train, X_test, y_test):
    results = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["Linear Regression"] = {"model": lr, "preds": y_pred_lr}

    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                    max_depth=4, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    results["Gradient Boosting"] = {"model": gb, "preds": y_pred_gb}

    for name, r in results.items():
        preds = r["preds"]
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        r.update({"mae": mae, "rmse": rmse, "r2": r2})
        log.info("%-22s  R²=%.4f  MAE=$%.0f  RMSE=$%.0f", name, r2, mae, rmse)

    return results

# ── Plots ─────────────────────────────────────

def plot_all(df_raw, X_test, y_test, results, feature_names):
    set_dark_style()
    gb = results["Gradient Boosting"]
    lr = results["Linear Regression"]
    y_pred = gb["preds"]

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(STYLE["bg"])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── 1. Predicted vs Actual ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, y_pred, alpha=0.5, s=18, color=STYLE["accent1"], edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, "--", color=STYLE["accent2"], lw=1.5, label="Perfect fit")
    ax1.set(title="Predicted vs Actual Charges", xlabel="Actual ($)", ylabel="Predicted ($)")
    ax1.legend(fontsize=8)
    ax1.text(0.05, 0.88, f"R² = {gb['r2']:.4f}", transform=ax1.transAxes,
             color=STYLE["accent3"], fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # ── 2. Residuals ──
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_test.values - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, s=18, color=STYLE["accent2"], edgecolors="none")
    ax2.axhline(0, color=STYLE["accent1"], lw=1.5, linestyle="--")
    ax2.set(title="Residuals vs Predicted", xlabel="Predicted ($)", ylabel="Residual ($)")
    ax2.grid(True, alpha=0.3)

    # ── 3. Feature Importance ──
    ax3 = fig.add_subplot(gs[1, 0])
    importances = gb["model"].feature_importances_
    idx = np.argsort(importances)
    colors = [STYLE["accent1"] if i != idx[-1] else STYLE["accent2"] for i in idx]
    ax3.barh(np.array(feature_names)[idx], importances[idx], color=colors, edgecolor="none")
    ax3.set(title="Feature Importance (Gradient Boosting)", xlabel="Importance")
    ax3.grid(True, alpha=0.3, axis="x")

    # ── 4. Smoker vs Non-smoker charges ──
    ax4 = fig.add_subplot(gs[1, 1])
    smoker_charges    = df_raw[df_raw["smoker"] == "yes"]["charges"]
    nonsmoker_charges = df_raw[df_raw["smoker"] == "no"]["charges"]
    ax4.hist(nonsmoker_charges, bins=40, alpha=0.7, color=STYLE["accent3"],
             label=f"Non-smoker  μ=${nonsmoker_charges.mean():,.0f}", edgecolor="none")
    ax4.hist(smoker_charges, bins=40, alpha=0.7, color=STYLE["accent2"],
             label=f"Smoker  μ=${smoker_charges.mean():,.0f}", edgecolor="none")
    ax4.set(title="Insurance Charges: Smoker vs Non-Smoker", xlabel="Charges ($)", ylabel="Count")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Title banner ──
    fig.text(0.5, 0.97,
             f"Medical Insurance Charges  |  Gradient Boosting  R²={gb['r2']:.4f}  MAE=${gb['mae']:,.0f}",
             ha="center", va="top", fontsize=13, fontweight="bold", color=STYLE["text"])

    path = f"{OUTPUT_DIR}/regression_linkedin.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    log.info("Saved → %s", path)

# ── Main ──────────────────────────────────────

def main():
    log.info("═"*55)
    log.info("  Regression – Medical Insurance Charges")
    log.info("═"*55)

    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    results = train_models(X_train, y_train, X_test, y_test)
    plot_all(df, X_test, y_test, results, X.columns.tolist())

    gb = results["Gradient Boosting"]
    print(f"\n{'═'*45}")
    print(f"  GRADIENT BOOSTING RESULTS")
    print(f"{'═'*45}")
    print(f"  R²   : {gb['r2']:.4f}")
    print(f"  MAE  : ${gb['mae']:,.0f}")
    print(f"  RMSE : ${gb['rmse']:,.0f}")
    print(f"\n  Smoker avg charge    : ${df[df['smoker']=='yes']['charges'].mean():,.0f}")
    print(f"  Non-smoker avg charge: ${df[df['smoker']=='no']['charges'].mean():,.0f}")
    print(f"  → Smokers pay {df[df['smoker']=='yes']['charges'].mean() / df[df['smoker']=='no']['charges'].mean():.1f}x more")
    log.info("Done ✓")

if __name__ == "__main__":
    main()
