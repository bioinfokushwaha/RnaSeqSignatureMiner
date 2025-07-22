import os
import random
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, f1_score, roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Set random seed and suppress warnings
random.seed(0)
np.random.seed(0)
warnings.filterwarnings("ignore")

# Create output directory
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)
os.chmod(output_dir, 0o777)

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

# Select features
indexes = np.asarray(np.where(lasso.coef_ != 0)).ravel()
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

# Define classifiers
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

# Evaluate models on test set
test_results = []
fprs, tprs, aucs, model_names = [], [], [], []

for name, model in models:
    model.fit(Xc_train, yc_train)
    y_test_pred = model.predict(Xc_test)
    y_test_prob = model.predict_proba(Xc_test)[:, 1]

    # Test set metrics
    acc = accuracy_score(yc_test, y_test_pred)
    f1 = f1_score(yc_test, y_test_pred)
    roc_auc = roc_auc_score(yc_test, y_test_prob)

    print(f"[Test] {name}: Accuracy={acc:.3f}, F1={f1:.3f}, AUC={roc_auc:.3f}")
    test_results.append((name, acc, f1, roc_auc))

    # ROC Curve
    fpr, tpr, _ = roc_curve(yc_test, y_test_prob)
    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(roc_auc)
    model_names.append(name)

# Save test results
df_results = pd.DataFrame(test_results, columns=["Models", "Test_Accuracy", "Test_F1", "Test_AUC"])
df_results.to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)

# Plot ROC curves
def plot_roc_curve(fprs, tprs, aucs, model_names):
    plt.figure(figsize=(11, 9))
    for fpr, tpr, auc_val, model_name in zip(fprs, tprs, aucs, model_names):
        plt.plot(fpr, tpr, label=f'{model_name} (AUC= {auc_val:.2f})')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set Only)")
    plt.grid(linestyle="dotted")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=12)
    plt.tight_layout()
    for fmt in ["png", "tiff", "svg", "jpeg"]:
        plt.savefig(os.path.join(output_dir, f"test_roc_curve.{fmt}"), format=fmt, bbox_inches="tight")

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
treatment = lda_df[lda_df["Sample"] == 1]

plt.figure(figsize=(8, 6))
plt.scatter(control['LD'], np.zeros_like(control['LD']), c="blue", label="Control", edgecolors='k', alpha=0.6)
plt.scatter(treatment['LD'], np.zeros_like(treatment['LD']), c="red", label="Treatment", edgecolors='k', alpha=0.6)
plt.title('LDA - Control vs Treatment')
plt.xlabel('LDA Component')
plt.yticks([])
plt.legend()
for fmt in ["png", "tiff", "svg", "jpeg"]:
    plt.savefig(os.path.join(output_dir, f"lda_plot.{fmt}"), format=fmt, bbox_inches="tight")
