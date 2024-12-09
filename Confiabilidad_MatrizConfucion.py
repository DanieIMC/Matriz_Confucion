import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Cargar el dataset
file_path = r"C:\Users\s2dan\OneDrive\Documentos\WorkSpace\Proyect_AI\ObesityDataSet_raw_and_data_sinthetic.csv"
data = pd.read_csv(file_path)

# Codificar las variables categóricas
label_encoder = LabelEncoder()

# Detectar las columnas categóricas y aplicarlas
categorical_columns = data.select_dtypes(include=['object']).columns

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Dividir el dataset en características (X) y la etiqueta (y)
X = data.drop('NObeyesdad', axis=1)  # Características
y = data['NObeyesdad']  # Etiqueta

# Dividir en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el clasificador Random Forest
rf_classifier = RandomForestClassifier(random_state=42)

# Entrenar el clasificador
rf_classifier.fit(X_train, y_train)

# Realizar predicciones
y_pred = rf_classifier.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Mostrar los resultados
print(f"Confiabilidad: {accuracy * 100:.2f}%")
print("\nMatriz de Confusión:")
print(conf_matrix)
print("\nReporte de Clasificación:")
print(class_report)
