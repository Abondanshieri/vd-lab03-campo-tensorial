# Laboratorio 03 · Visualización de campo tensorial

De una matriz por punto a autovalores, FA y glifos

**Visualización de Datos** · Escuela Profesional de Ingeniería de Sistemas · Universidad Nacional del Altiplano – Puno
Estudiante: Franco · 25 de noviembre de 2026

Se construyó un campo de 15 × 15 tensores simétricos de 3 × 3 (`T.shape = (15, 15, 3, 3)`), se obtuvieron sus autovalores y autovectores, se calcularon los mapas escalares MD y FA y se dibujaron glifos elípticos. Tres retos exploraron el caso isotrópico, el efecto de reforzar la anisotropía y la superposición de glifos.

## Estructura

| Ruta | Contenido |
|---|---|
| `notebooks/LAB_03_Campo_Tensorial.ipynb` | Cuaderno ejecutado: bloques guiados, retos, respuestas y conclusiones |
| `src/LAB_03_Campo_Tensorial.py` | El mismo código en texto plano |
| `anexos/figuras/` | 4 figuras PNG extraídas del cuaderno |
| `anexos/salidas/` | Salidas de consola de cada bloque |
| `informe/` | Informe del laboratorio en `.docx` y `.pdf` |

## Cómo reproducirlo

```bash
pip install numpy matplotlib jupyter
jupyter notebook notebooks/LAB_03_Campo_Tensorial.ipynb
```

También puede abrirse en Google Colab (*Archivo → Abrir cuaderno → GitHub*, con la sesión de GitHub autorizada, ya que el repositorio es privado) y ejecutarse bloque por bloque con `Shift + Enter`.
Los datos son sintéticos: se generan en el propio código, por lo que no hay archivos de datos externos.
