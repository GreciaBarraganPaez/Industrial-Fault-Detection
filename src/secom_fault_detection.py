from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

# ========== CONFIGURACIÓN ==========
SECOM_PATH = Path(__file__).resolve().parents[1] / 'data' / 'secom.data'
LABELS_PATH = Path(__file__).resolve().parents[1] / 'data' / 'secom_labels.data'
SELECTED_FEATURES = 40  # Según línea base del dataset
RANDOM_STATE = 42
# ===================================

# Configuración visual
plt.style.use('ggplot')
sns.set(rc={'figure.figsize':(12,6)})

# 1. Carga de datos
def load_secom():
    try:
        # Cargar características
        X = pd.read_csv(SECOM_PATH,
                       sep=r'\s+',
                       header=None,
                       na_values='NaN',
                       engine='python')

        # Eliminar columnas con >90% NaN
        X = X.loc[:, X.isnull().mean() < 0.9]

        # Cargar etiquetas
        y = pd.read_csv(LABELS_PATH,
                       sep=r'\s+',
                       header=None,
                       usecols=[0]).squeeze().replace(-1,0)

        return X, y

    except Exception as e:
        print(f"Error loading SECOM data: {str(e)}")
        print(f"Expected files: {SECOM_PATH} and {LABELS_PATH}")
        return None, None

# 2. Pipeline de procesamiento
def build_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        SelectKBest(score_func=f_classif, k=SELECTED_FEATURES),
        SMOTE(k_neighbors=3, random_state=RANDOM_STATE),
        RandomForestClassifier(n_estimators=150,
                              class_weight='balanced',
                              random_state=RANDOM_STATE,
                              n_jobs=-1)
    )

# 3. Análisis de resultados
def analyze_results(y_true, y_pred, y_proba):
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fallo'],
                yticklabels=['Normal', 'Fallo'])
    plt.title('Matriz de Confusión')
    plt.show()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1], [0,1], linestyle='--', color='navy')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()

# 4. Ejecución principal
if __name__ == "__main__":
    # Carga de datos
    print("Cargando dataset SECOM...")
    X, y = load_secom()
    if X is None or y is None:
        exit()

    # Estadísticas iniciales
    print(f"\nMuestras: {X.shape[0]}, Características originales: {X.shape[1]}")
    print(f"Distribución de clases:\n{y.value_counts().to_frame('count')}")

    # Preprocesamiento y modelado
    pipeline = build_pipeline()

    # Validación cruzada
    print("\nEvaluando modelo...")
    scores = cross_val_score(pipeline, X, y,
                            cv=10,
                            scoring='balanced_accuracy',
                            n_jobs=-1)
    print(f"\nPrecisión balanceada (10-fold CV): {scores.mean():.1%} (±{scores.std():.2})")

    # Entrenamiento final
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=0.3,
                                                       stratify=y,
                                                       random_state=RANDOM_STATE)
    pipeline.fit(X_train, y_train)

    # Predicciones
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]

    # Resultados
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fallo']))

    # Visualización
    analyze_results(y_test, y_pred, y_proba)

    # Importancia de características
    selected_features = pipeline.named_steps['selectkbest'].get_support()
    # Preserve the original SECOM column IDs for the selected model inputs.
    selected_column_ids = np.asarray(X.columns)[selected_features]
    importances = pipeline.named_steps['randomforestclassifier'].feature_importances_
    top_features = np.argsort(importances)[-20:]

    plt.figure(figsize=(10,8))
    plt.barh([f"Feature {selected_column_ids[i] + 1}" for i in top_features],
             importances[top_features])
    plt.title('Top 20 Características más Importantes')
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.show()
