# Redes Neuronales Profundas - Trabajo Práctico Integrador
## Where's Wally? 

Este repositorio contiene el proyecto final del **Grupo 8**. El objetivo es entrenar y desplegar una red neuronal con PyTorch capaz de resolver la detección de Wally sobre imágenes complejas.

### Integrantes
* García Nieto, Roberto - 47576
* Moyano, Gonzalo Damian - 47600
* Fernandez Rossi, Matías - 49483

### Estructura del Repositorio 
De acuerdo con las pautas presentadas, el proyecto se divide en las siguientes secciones:
* **`data/`**: README de datos, scripts y archivos CSV de particionados livianos. Las imágenes locales están excluidas por su tamaño
* **`dev/`**: Notebooks de desarrollo experimental (`.ipynb`) y almacenamiento de los pesos finales del modelo (`modelo.pth`)
* **`prod/`**: Código fuente de producción para la interfaz web (Streamlit) e inferencia en producción.

---
###URL de la aplicación
https://whereswaldo.streamlit.app

---
### 🎮 Instrucciones de Uso

La aplicación cuenta con una barra lateral de navegación que divide la experiencia en dos grandes modalidades:

#### 1. Modo: Jugar contra la IA
* **Selección de Dificultad:** Elegí entre los niveles disponibles (*Fácil, Medio, Difícil o Extra*). El sistema cargará el póster correspondiente y establecerá la configuración de la lupa de exploración en alta definición.
* **Marcado de Objetivos:** Seleccioná el personaje que vas a buscar en la paleta de herramientas de la botonera lateral. Hacé clic y arrastrá el mouse sobre el lienzo (**Canvas interactivo**) para dibujar un recuadro sobre la zona donde creés que se encuentra.
* **Evaluación IoU:** Ajustá el slider de *Tolerancia IoU* a nivel matemático para definir la exigencia del emparejamiento.
* **Verificación:** Presioná **"Verificar recuadros marcados"**. La app cruzará espacialmente tus coordenadas y las de la IA contra las posiciones reales del dataset (*Ground Truth*). El sistema arrojará un veredicto dinámico (*Ganaste, Ganó la IA o Empate*) detallando el feedback cualitativo y permitiéndote descargar un plano anotado comparativo.

#### 2. Modo: Resolver con IA (Escáner Neuronal)
* **Carga de Archivo:** Subí cualquier póster propio desde el gestor de archivos (*Dropzone*) en formato `.jpg`, `.png` o `.webp`.
* **Inferencia Automatizada:** Hacé clic en **"Ejecutar Escáner IA"**. La red neuronal procesará el plano general mediante segmentación por parches superpuestos.
* **Resultados:** La interfaz devolverá un contador exacto detallando cuántos objetivos de cada clase fueron localizados con éxito y desplegará la imagen final reescalada con sus respectivas bounding boxes de confianza listas para descargar en alta calidad.
