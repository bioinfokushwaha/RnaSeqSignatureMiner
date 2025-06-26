import os
import random
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, roc_auc_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

random.seed(0)
np.random.seed(0)

# Create output directory
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)
os.chmod(output_dir, 0o775)

# Load and preprocess data
df = pd.read_excel("normalised_values.xlsx")
df = df.fillna(0)
df_T = df.transpose(copy=True)
header_row = df_T.loc["Geneid"]
df_T.columns = header_row
df_T = df_T.drop("Geneid")
df_transposed = df_T.reset_index()
df_transposed.rename(columns={'index': 'Geneid'}, inplace=True)

# Load sample info
df_S = pd.read_excel("Sampleinfo.xlsx")
print(df.isna().sum())
df_S = df_S.replace({"Treatment": 1, "Control": 0})

# Merge expression and sample info
data_feature = pd.merge(df_transposed, df_S, right_on="Sample", left_on="Geneid", how="inner")

feature = data_feature.drop(columns=["Geneid", "condition", "Sample"])
sample = data_feature.iloc[:, -1]
columns_to_keep = [col for col in data_feature.columns if col not in ["Geneid", "condition", "Sample"]]

# Scale data
scaler = RobustScaler()
feature_scaled = pd.DataFrame(scaler.fit_transform(feature), index=feature.index, columns=columns_to_keep)

# Feature selection using Lasso
lasso = Lasso(alpha=0.001, max_iter=200)
lasso.fit(feature_scaled, sample)
y_pred = lasso.predict(feature_scaled)

r2 = r2_score(sample, y_pred)
rmse = np.sqrt(mean_squared_error(sample, y_pred))
print("R-squared:", r2)
print("RMSE:", rmse)

indexes = np.asarray(np.where(lasso.coef_ != 0)).ravel()
print("Selected features (Lasso):", indexes.shape)

# Extract selected features
cols = feature_scaled.iloc[:, indexes]
cl_list = list(cols.columns)
class_data = feature_scaled[cl_list]
class_data["Sample"] = sample

# Save selected features
pd.DataFrame(cl_list).to_csv(os.path.join(output_dir, "selected_features.txt"), index=False, header=False)

# Train/test split
class_train_data, class_test_data = train_test_split(class_data, test_size=0.3, random_state=42)
Xc_train = class_train_data.drop(columns=["Sample"])
yc_train = class_train_data['Sample']
Xc_test = class_test_data.drop(columns=['Sample'])
yc_test = class_test_data['Sample']

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
models = [
    ("Ada_boost", AdaBoostClassifier(random_state=0)),
    ("Rand_For", RandomForestClassifier(random_state=0)),
    ("Grad_Boost", GradientBoostingClassifier(random_state=0)),
    ("Xtra_Tree", ExtraTreesClassifier(random_state=0)),
    ("NB", GaussianNB()),
    ("KNC", KNeighborsClassifier()),
    ("SVM_Poly", SVC(kernel="poly", probability=True, random_state=0, C=11)),
    ("SVM_Rad", SVC(kernel="rbf", probability=True, random_state=0, C=2)),
    ("Dec_Tree", DecisionTreeClassifier(random_state=0))
]

fprs, tprs, aucs, model_names = [], [], [], []
results_summary = []

for name, model in models:
    mean_fpr = np.linspace(0, 1, 100)
    tpr_list, auc_list = [], []
    fold_accuracy, fold_f1 = [], []

    for train_index, val_index in kf.split(Xc_train, yc_train):
        X_fold_train, X_val = Xc_train.iloc[train_index], Xc_train.iloc[val_index]
        y_fold_train, y_val = yc_train.iloc[train_index], yc_train.iloc[val_index]

        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_list.append(interp_tpr)
        auc_list.append(auc(fpr, tpr))

        fold_accuracy.append(accuracy_score(y_val, y_pred))
        fold_f1.append(f1_score(y_val, y_pred))

    avg_accuracy = np.mean(fold_accuracy)
    avg_f1 = np.mean(fold_f1)
    avg_auc = np.mean(auc_list)

    results_summary.append((name, avg_accuracy, avg_f1, avg_auc))

    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0
    fprs.append(mean_fpr)
    tprs.append(mean_tpr)
    aucs.append(avg_auc)
    model_names.append(name)

# Save metrics
results_df = pd.DataFrame(results_summary, columns=["Models", "Accuracy", "F1-Score", "AUC"])
results_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
print(results_df)

# Plot ROC curves
def plot_roc_curve(fprs, tprs, aucs, model_names):
    plt.figure(figsize=(11, 9))
    for fpr, tpr, auc_val, model_name in zip(fprs, tprs, aucs, model_names):
        plt.plot(fpr, tpr, label=f'{model_name} (AUC= {auc_val:.2f})')
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(linestyle="dotted")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=14, frameon=False)
    plt.tight_layout()
    for fmt in ["png", "tiff", "svg", "jpeg"]:
        plt.savefig(os.path.join(output_dir, f"auc_curve.{fmt}"), format=fmt, bbox_inches="tight")

plot_roc_curve(fprs, tprs, aucs, model_names)

# LDA Plot
X_lda = class_data.drop(columns="Sample")
y_lda = class_data['Sample']
scaler_lda = StandardScaler()
X_lda_scaled = scaler_lda.fit_transform(X_lda)
lda = LinearDiscriminantAnalysis()
lda_components = lda.fit_transform(X_lda_scaled, y_lda)

lda_df = pd.DataFrame(data=lda_components, columns=['LD'])
lda_df["Sample"] = y_lda
control = lda_df[lda_df["Sample"] == 0]
infection = lda_df[lda_df["Sample"] == 1]

plt.figure(figsize=(8, 6))
plt.scatter(control['LD'], np.zeros_like(control['LD']), c="blue", label="Control", marker='o', edgecolors='k', alpha=0.6)
plt.scatter(infection['LD'], np.zeros_like(infection['LD']), c="red", label="Mastitis", marker='o', edgecolors='k', alpha=0.6)
plt.title('LDA - Control vs Mastitis')
plt.xlabel('LDA Components')
plt.yticks([])
plt.legend()
for fmt in ["png", "tiff", "svg", "jpeg"]:
    plt.savefig(os.path.join(output_dir, f"lda_plot.{fmt}"), format=fmt, bbox_inches="tight")

