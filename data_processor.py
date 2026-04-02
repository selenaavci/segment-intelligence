import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(uploaded_file):
    """CSV veya Excel dosyasini yukler."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Desteklenen formatlar: CSV, XLSX, XLS")
    return df


def analyze_columns(df):
    """Kolon tiplerini otomatik siniflandirir."""
    analysis = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "id_or_useless": [],
    }

    for col in df.columns:
        # ID benzeri kolonlar
        if col.lower() in ("id", "index", "row", "unnamed: 0") or col.lower().endswith("_id"):
            analysis["id_or_useless"].append(col)
            continue

        # Tarih alanlari
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            analysis["datetime"].append(col)
            continue

        # Tarih parse denemesi
        if df[col].dtype == object:
            try:
                pd.to_datetime(df[col].dropna().head(20))
                analysis["datetime"].append(col)
                continue
            except (ValueError, TypeError):
                pass

        # Sayisal kolonlar
        if pd.api.types.is_numeric_dtype(df[col]):
            nunique = df[col].nunique()
            # Cok fazla unique deger + ardisik pattern = muhtemelen ID
            if nunique == len(df) and df[col].is_monotonic_increasing:
                analysis["id_or_useless"].append(col)
            else:
                analysis["numeric"].append(col)
            continue

        # Kategorik kolonlar
        if df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col]):
            nunique = df[col].nunique()
            # Cok fazla unique = muhtemelen free-text veya ID
            if nunique > 50 or nunique == len(df):
                analysis["id_or_useless"].append(col)
            else:
                analysis["categorical"].append(col)
            continue

        analysis["id_or_useless"].append(col)

    return analysis


def recommend_features(analysis):
    """Clustering icin uygun kolonlari onerir."""
    recommended = analysis["numeric"] + analysis["categorical"]
    excluded = analysis["id_or_useless"] + analysis["datetime"]
    return recommended, excluded


def preprocess_data(df, selected_features, apply_pca=False, pca_components=2):
    """Secilen feature'lari preprocess eder."""
    work_df = df[selected_features].copy()

    # Missing value handling
    for col in work_df.columns:
        if pd.api.types.is_numeric_dtype(work_df[col]):
            work_df[col] = work_df[col].fillna(work_df[col].median())
        else:
            work_df[col] = work_df[col].fillna(work_df[col].mode().iloc[0] if not work_df[col].mode().empty else "unknown")

    # Encoding categorical features
    label_encoders = {}
    categorical_cols = work_df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        work_df[col] = le.fit_transform(work_df[col].astype(str))
        label_encoders[col] = le

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(work_df)
    scaled_df = pd.DataFrame(scaled_data, columns=work_df.columns, index=work_df.index)

    # PCA
    pca_model = None
    if apply_pca and scaled_data.shape[1] > pca_components:
        from sklearn.decomposition import PCA
        pca_model = PCA(n_components=pca_components)
        pca_data = pca_model.fit_transform(scaled_data)
        pca_df = pd.DataFrame(
            pca_data,
            columns=[f"PC{i+1}" for i in range(pca_components)],
            index=work_df.index,
        )
        return pca_df, scaled_df, scaler, label_encoders, pca_model

    return scaled_df, scaled_df, scaler, label_encoders, pca_model
