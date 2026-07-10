import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def load_data(path="products.csv"):
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def prepare_features(df):
    # Infer target as a common name or as the last column
    possible_targets = [c for c in ["category", "label", "target"] if c in df.columns]
    if possible_targets:
        target_col = possible_targets[0]
    else:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical columns
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include=[object, "category"]).columns:
        X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))

    # Encode target if needed
    if y.dtype == object or str(y.dtype).startswith("category"):
        y = LabelEncoder().fit_transform(y.astype(str))

    return X_enc, y


def train_and_evaluate(path="products.csv", test_size=0.2, n_neighbors=5):
    df = load_data(path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    return model, acc


if __name__ == "__main__":
    # Ejecutar con el archivo 'products.csv' en el mismo directorio o pasar ruta
    train_and_evaluate("products.csv")
