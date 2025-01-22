### **Aplicación de Modelos de Regresión Lineal y Regresión Logística en los Proyectos**

Los modelos de **regresión lineal** y **regresión logística** son herramientas poderosas para predecir valores continuos y clasificar datos, respectivamente.

---

### **1. Analizador de Sentimientos en Reseñas de Productos**

**Modelo adecuado: Regresión Logística**

- **Contexto de aplicación:**  
  La regresión logística es ideal para este proyecto, ya que el problema es de clasificación (positivo o negativo). Se entrenará el modelo con características extraídas del texto, como conteos de palabras clave (TF-IDF o Bag of Words).

**Pasos clave de implementación:**

1. **Preparación de datos:**
   - Limpieza del texto (eliminación de stopwords, tokenización).
   - Transformación del texto en características numéricas con TF-IDF.
2. **Entrenamiento:**

   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import train_test_split

   vectorizer = TfidfVectorizer(max_features=5000)
   X = vectorizer.fit_transform(data['clean_text']).toarray()
   y = data['label']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   model = LogisticRegression()
   model.fit(X_train, y_train)
   ```

3. **Evaluación:**
   - Medir precisión y matriz de confusión.
   ```python
   from sklearn.metrics import accuracy_score, confusion_matrix
   y_pred = model.predict(X_test)
   print("Precisión:", accuracy_score(y_test, y_pred))
   print(confusion_matrix(y_test, y_pred))
   ```

---

### **2. Predicción del Precio de Casas**

**Modelo adecuado: Regresión Lineal**

- **Contexto de aplicación:**  
  La regresión lineal es perfecta para este problema, ya que el objetivo es predecir un valor numérico continuo (precio de la casa) basado en características como el tamaño, la ubicación y el número de habitaciones.

**Pasos clave de implementación:**

1. **Preparación de datos:**
   - Manejo de valores nulos y escalado de características.
2. **Entrenamiento:**

   ```python
   from sklearn.linear_model import LinearRegression
   from sklearn.model_selection import train_test_split

   X = data[['tamaño', 'habitaciones', 'ubicación']]
   y = data['precio']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   model = LinearRegression()
   model.fit(X_train, y_train)
   ```

3. **Evaluación:**
   - Calcular error cuadrático medio (RMSE).
   ```python
   from sklearn.metrics import mean_squared_error
   y_pred = model.predict(X_test)
   print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
   ```

---

### **3. Clasificación de Manuscritos Digitalizados**

**Modelo adecuado: Regresión Logística**

- **Contexto de aplicación:**  
  La regresión logística es adecuada para la clasificación de dígitos manuscritos (0-9), ya que se trata de un problema de clasificación multinomial.

**Pasos clave de implementación:**

1. **Preparación de datos:**
   - Normalización de imágenes y división en entrenamiento y prueba.
2. **Entrenamiento:**

   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.datasets import load_digits
   from sklearn.model_selection import train_test_split

   digits = load_digits()
   X = digits.data
   y = digits.target

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   model = LogisticRegression(max_iter=1000)
   model.fit(X_train, y_train)
   ```

3. **Evaluación:**
   - Medir la precisión de la clasificación.
   ```python
   from sklearn.metrics import accuracy_score
   y_pred = model.predict(X_test)
   print("Precisión:", accuracy_score(y_test, y_pred))
   ```

---

### **4. Predicción de Accidentes de Tránsito**

**Modelo adecuado: Regresión Logística**

- **Contexto de aplicación:**  
  La regresión logística es útil para predecir la probabilidad de que un accidente sea grave o leve (clasificación binaria).

**Pasos clave de implementación:**

1. **Preparación de datos:**
   - Conversión de variables categóricas en variables numéricas.
2. **Entrenamiento:**

   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import train_test_split

   X = data[['hora', 'clima', 'ubicación']]
   y = data['gravedad']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   model = LogisticRegression()
   model.fit(X_train, y_train)
   ```

3. **Evaluación:**
   - Medir precisión, matriz de confusión y curva ROC.
   ```python
   from sklearn.metrics import classification_report
   print(classification_report(y_test, model.predict(X_test)))
   ```

---

### **5. Resumen Automático de Artículos**

**Modelo adecuado: Regresión Lineal (con ajuste personalizado)**

- **Contexto de aplicación:**  
  Se puede aplicar regresión lineal para predecir la relevancia de las oraciones en un texto basándose en características como la longitud y la frecuencia de palabras clave.

**Pasos clave de implementación:**

1. **Preparación de datos:**
   - Extraer características de texto (frecuencia de palabras, longitud de oraciones).
2. **Entrenamiento:**

   ```python
   from sklearn.linear_model import LinearRegression

   X = data[['longitud_oracion', 'frecuencia_palabras']]
   y = data['relevancia']

   model = LinearRegression()
   model.fit(X, y)
   ```

3. **Evaluación:**
   - Comparar con resúmenes creados manualmente.

---

### **6. Predicción de Demanda de Ventas**

**Modelo adecuado: Regresión Lineal**

- **Contexto de aplicación:**  
  La regresión lineal es perfecta para predecir ventas futuras a partir de datos históricos, considerando factores como tendencias estacionales.

**Pasos clave de implementación:**

1. **Preparación de datos:**
   - Conversión de fechas en variables numéricas, escalado de datos.
2. **Entrenamiento:**

   ```python
   from sklearn.linear_model import LinearRegression

   X = data[['mes', 'promociones', 'precio']]
   y = data['ventas']

   model = LinearRegression()
   model.fit(X, y)
   ```

3. **Evaluación:**
   - Evaluar precisión con RMSE y graficar predicciones vs valores reales.
   ```python
   import matplotlib.pyplot as plt
   plt.plot(y_test.values, label="Real")
   plt.plot(model.predict(X_test), label="Predicho")
   plt.legend()
   plt.show()
   ```

---
