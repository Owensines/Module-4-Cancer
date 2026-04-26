# Filters data
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score



meta = pd.read_csv(r'data\TRAINING_SET_GSE62944_metadata.csv')
expr = pd.read_csv(r'data\TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0) 



# Filter for lung cancer types LUAD and LUSC
meta = meta[meta['cancer_type'].isin(['LUAD', 'LUSC'])]
expr = expr.T # transpose to have samples as rows



my_genes = [
    'CD274', 'CTLA4', 'LAG3', 'HLA-A', 'B2M', 'STAT3', 
    'TGFB1', 'MYC', 'EGFR', 'PIK3CA', 'BRAF', 'CTNNB1',
    'PTEN', 'TP53', 'STK11', 'RB1', 'SMAD4', 'APC', 'ATM'
]

X_data = expr[my_genes]
df = X_data.join(meta.set_index('sample'), how='inner')

labels = ['cancer_type', 'ajcc_pathologic_tumor_stage']
df_clean = df[my_genes + labels]

# model
# defines x and y
X = df_clean[my_genes]
y = df_clean['cancer_type']

# logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# in sample error (training dataset)
y_train_pred = model.predict(X)
train_acc = accuracy_score(y, y_train_pred)


print(f"In-sample Accuracy (Training): {train_acc:.2f}")

# saves model
import joblib
joblib.dump(model, 'lung_cancer_model.pkl')