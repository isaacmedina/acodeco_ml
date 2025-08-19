# Predicción y alerta de variación de precios en la Canasta Básica (Ciudad de Panamá)

## Descripción
Este proyecto desarrolla un sistema de **aprendizaje automático** para predecir el precio mensual de productos de la **Canasta Básica Familiar de Alimentos (CBA)** en supermercados de la Ciudad de Panamá.  
El sistema también genera **alertas automáticas de incrementos significativos (≥ X%, por defecto 5%)** para apoyar en la vigilancia de precios y la toma de decisiones.

---

## Objetivos

Construir un sistema de *machine learning* que prediga el precio del próximo mes por producto y supermercado, y que emita alertas de incrementos significativos.

- Unificar y limpiar datos históricos de precios (ACODECO, 2020–2022 o rango disponible).
- Diseñar variables predictoras temporales y categóricas a nivel producto–supermercado.
- Entrenar y comparar modelos de regresión y clasificación frente a *baselines* simples.
- Evaluar con cortes temporales estrictos y métricas adecuadas.
- Documentar resultados, limitaciones y recomendaciones.

---

## Alcance
- **Cobertura:** productos de la CBA en supermercados de la Ciudad de Panamá.  
- **Granularidad:** (producto, presentación, supermercado, mes).  
- **Salidas:**
  - **Regresión:** precio del próximo mes.  
  - **Clasificación:** alerta de incremento ≥ X% (default X=5%).  

---

## Datos y fuentes
- **Fuente:** ACODECO – publicaciones mensuales de precios de la CBA.  
- **Formato esperado:** CSV/XLS/XLSX (un archivo por mes).  
- **Campos mínimos:** producto, presentación, supermercado, precio, fecha.  
- **Llave canónica:** (producto, presentación, supermercado, cadena, fecha_mes).  
- **Tratamiento:** limpieza de nulos, estandarización de cadenas, detección/corrección de outliers, conversión de precios a numérico.

---


## Instalación y uso

### Requisitos
- **Python** 3.10+  
- Librerías mínimas:  
  ```bash
  pip install -r requirements.txt
  ```
  - pandas  
  - scikit-learn  
  - numpy  
  - matplotlib  

### Ejecución
1. Descargar datasets de ACODECO y colocarlos en `data/raw/`.  
2. Ejecutar **Notebook 1** para limpieza y unificación.  
3. Ejecutar **Notebook 2** para entrenamiento de modelos.  
4. Ejecutar **Notebook 3** para evaluación y explicabilidad.  
