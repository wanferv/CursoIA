### **Ejemplo Completo: Analizador de Sentimientos en Reseñas de Productos**

Este ejemplo sigue la guía unificada para implementar un analizador de sentimientos utilizando procesamiento de lenguaje natural (PLN) y modelos supervisados. El objetivo es clasificar reseñas de productos como positivas o negativas.

---

#### **1. Explora los Datos**

- **Cargar los datos:**  
  Usaremos un dataset de reseñas de productos disponible en Kaggle. Si no tienes acceso, puedes utilizar un conjunto simulado.

```python
# Importar librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
data = pd.read_csv(url)

# Mostrar las primeras filas
print(data.head())
```

- **Entender los datos:**  
  El dataset incluye columnas como "text" (reseña) y "label" (0: negativa, 1: positiva).

```python
print(data.info())
print(data['label'].value_counts())  # Distribución de etiquetas
```

- **Visualizar datos:**  
  Gráfico de barras para entender la distribución de reseñas.

```python
sns.countplot(data['label'])
plt.title('Distribución de Sentimientos')
plt.show()
```

---

#### **2. Preprocesa los Datos**

- **Limpieza del texto:**  
  Elimina signos de puntuación, convierte el texto a minúsculas y realiza tokenización.

```python
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Función de preprocesamiento
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Eliminar puntuación y minúsculas
    tokens = word_tokenize(text)  # Tokenización
    tokens = [word for word in tokens if word not in stop_words]  # Quitar stopwords
    return ' '.join(tokens)

# Aplicar preprocesamiento
data['clean_text'] = data['text'].apply(preprocess_text)
```

- **Dividir los datos:**  
  Separar en conjuntos de entrenamiento y prueba.

```python
from sklearn.model_selection import train_test_split

X = data['clean_text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- **Vectorización:**  
  Convierte el texto en representaciones numéricas utilizando TF-IDF.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()
```

---

#### **3. Crea un Modelo**

- **Seleccionar un modelo:**  
  Utilizaremos un clasificador Naive Bayes para este proyecto.

```python
from sklearn.naive_bayes import MultinomialNB

# Entrenar el modelo
model = MultinomialNB()
model.fit(X_train_vec, y_train)
```

---

#### **4. Evalúa el Modelo**

- **Realizar predicciones:**

```python
y_pred = model.predict(X_test_vec)
```

- **Métricas de evaluación:**  
  Usaremos precisión, recall y F1-score para medir el rendimiento.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Calcular precisión
print("Precisión:", accuracy_score(y_test, y_pred))

# Mostrar el informe de clasificación
print("\nInforme de clasificación:\n", classification_report(y_test, y_pred))

# Matriz de confusión
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Verdad')
plt.show()
```

---

#### **5. Ideas para Mejoras**

1. **Aumenta la complejidad:** Prueba con modelos más avanzados como Random Forest o redes neuronales.
2. **Explora embeddings de palabras:** Usa Word2Vec o GloVe para mejorar las representaciones textuales.
3. **Incrementa el dataset:** Agrega datos de otras fuentes para obtener un modelo más robusto.

---

#### **Resultados Esperados**

1. **Precisión del modelo:** Un clasificador básico como Naive Bayes debería alcanzar una precisión entre el 70% y 80%.
2. **Matriz de confusión:** Identifica dónde el modelo comete errores y por qué.

Este proyecto puede servir como base para otros casos de PLN, como el análisis de emociones o la clasificación temática de texto.
