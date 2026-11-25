"""Laboratorio 03 · Visualización de campo tensorial — código plano (exportado del cuaderno).
Ejecutar bloque por bloque: cada bloque reutiliza las variables de los anteriores."""

# ===== Entorno =====
import sys, platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110          # nitidez de las figuras exportadas
print("Python    :", platform.python_version())
print("NumPy     :", np.__version__)
print("Matplotlib:", matplotlib.__version__)
plt.show = lambda *a, **k: None  # ejecución sin ventanas


# ===== Bloque 1. Conocer una sola matriz antes de crear un campo =====
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Una matriz tensorial simétrica de 3 x 3
A = np.array([[3.0, 0.0, 0.0],
              [0.0, 1.5, 0.0],
              [0.0, 0.0, 0.5]])

print("Matriz A:")
print(A)
print("Dimensiones de A:", A.shape)
print("Componente X-X:", A[0, 0])


# ===== Bloque 2. Obtener los autovalores y autovectores =====
valores, vectores = np.linalg.eigh(A)

# Ordenamos de mayor a menor valor
valores = valores[::-1]
vectores = vectores[:, ::-1]

print("Autovalores (mayor a menor):", valores)
print("Autovectores (columnas):")
print(vectores)

# Comprobamos la definición para el primer par
print("¿A·e1 = lambda1·e1?",
      np.allclose(A @ vectores[:, 0], valores[0] * vectores[:, 0]))


# ===== Bloque 3. Comparar isotropía y anisotropía =====
# Tres ejemplos con autovalores positivos
casos = np.array([[2.0, 2.0, 2.0],
                  [3.0, 1.5, 0.5],
                  [4.0, 0.0, 0.0]])

for w in casos:
    md = np.mean(w)                     # Promedio de autovalores
    numerador = np.sum((w - md)**2)     # Diferencias respecto al promedio
    denominador = np.sum(w**2)         # Normalización
    fa = np.sqrt(1.5 * numerador / denominador)
    print("Autovalores:", w, "MD:", round(md, 3),
          "FA:", round(fa, 3))


# ===== Bloque 4. Construir una matriz 3 × 3 en cada punto =====
# Creamos 15 x 15 posiciones
x = np.linspace(-2, 2, 15)
y = np.linspace(-2, 2, 15)
X, Y = np.meshgrid(x, y)

# La dirección principal gira según la posición
angulo = np.arctan2(Y, X) + np.pi / 2
c = np.cos(angulo)
s = np.sin(angulo)
r = np.sqrt(X**2 + Y**2)

# Valores positivos para el tensor de ejemplo
mayor = 1 + 2 * np.exp(-((r - 1.2)**2) / 0.25)
medio = np.ones_like(X)
menor = 0.6 * np.ones_like(X)

# Reservamos una matriz 3 x 3 para cada posición
T = np.zeros((15, 15, 3, 3))
T[:, :, 0, 0] = mayor*c*c + medio*s*s
T[:, :, 0, 1] = (mayor - medio)*c*s
T[:, :, 1, 0] = T[:, :, 0, 1]
T[:, :, 1, 1] = mayor*s*s + medio*c*c
T[:, :, 2, 2] = menor

print("Número de matrices:", X.size)
print("Forma de T:", T.shape)
print("Matriz en una posición:")
print(np.round(T[7, 11], 2))


# ===== Bloque 5. Crear mapas escalares a partir del campo tensorial =====
# Obtenemos los tres autovalores en cada punto
w, v = np.linalg.eigh(T)
w = w[:, :, ::-1]
v = v[:, :, :, ::-1]

# MD: promedio de los tres autovalores
MD = np.mean(w, axis=2)

# FA: diferencia relativa de autovalores
numerador = np.sum((w - MD[:, :, None])**2, axis=2)
denominador = np.sum(w**2, axis=2)
FA = np.sqrt(1.5 * numerador / denominador)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
im1 = axes[0].imshow(MD, origin="lower", extent=[-2, 2, -2, 2],
                     cmap="viridis")
axes[0].set_title("Mapa de difusividad media (MD)")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(FA, origin="lower", extent=[-2, 2, -2, 2],
                     cmap="magma", vmin=0, vmax=1)
axes[1].set_title("Mapa de anisotropía (FA)")
fig.colorbar(im2, ax=axes[1])
plt.tight_layout()
plt.show()


# ===== Bloque 6. Dibujar glifos: una elipse por posición seleccionada =====
fig, ax = plt.subplots(figsize=(7, 7))

# Dibujamos una elipse cada dos puntos de la grilla
for fila in range(0, 15, 2):
    for columna in range(0, 15, 2):
        # Valores principales de esta matriz
        lam1 = w[fila, columna, 0]
        lam2 = w[fila, columna, 1]

        # Dirección principal proyectada al plano XY
        e1 = v[fila, columna, :, 0]
        ang = np.degrees(np.arctan2(e1[1], e1[0]))

        # Elipses que resumen forma y orientación
        elipse = Ellipse((X[fila, columna], Y[fila, columna]),
                         width=0.19*lam1, height=0.19*lam2,
                         angle=ang, facecolor="skyblue",
                         edgecolor="navy", alpha=0.85)
        ax.add_patch(elipse)

ax.set_xlim(-2.4, 2.4)
ax.set_ylim(-2.4, 2.4)
ax.set_aspect("equal")
ax.set_title("Glifos elípticos de un campo tensorial")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()


# ===== Reto 1. Caso isotrópico: [3, 1,5, 0,5] por [2, 2, 2] =====
def fa_de(lambdas):
    """FA y MD de un conjunto de tres autovalores (misma fórmula del bloque 3)."""
    lambdas = np.asarray(lambdas, dtype=float)
    md = lambdas.mean()
    return md, np.sqrt(1.5 * np.sum((lambdas - md)**2) / np.sum(lambdas**2))

original = [[2, 2, 2], [3, 1.5, 0.5], [4, 0, 0]]
nuevo    = [[2, 2, 2], [2, 2, 2],     [4, 0, 0]]

print(f"{'Autovalores':<18}{'MD':>8}{'FA antes':>10}{'FA ahora':>10}")
for a, b in zip(original, nuevo):
    print(f"{str(b):<18}{fa_de(b)[0]:>8.3f}{fa_de(a)[1]:>10.3f}{fa_de(b)[1]:>10.3f}")

# Un tensor isotrópico es un múltiplo de la identidad: toda dirección es autovector
I2 = 2 * np.eye(3)
print("\nAutovalores de 2·I:", np.linalg.eigvalsh(I2))
print("Con un vector cualquiera p:", np.allclose(I2 @ np.array([1.0, 2.0, 3.0]), 2 * np.array([1.0, 2.0, 3.0])))


# ===== Reto 2. Reforzar la anisotropía: el 2 de `mayor` por 3 =====
def campo_tensorial(amp):
    """Repite los bloques 4 y 5 con amplitud `amp` en el anillo; devuelve autovalores, autovectores, MD y FA."""
    angulo = np.arctan2(Y, X) + np.pi / 2
    cc, ss = np.cos(angulo), np.sin(angulo)
    rr = np.sqrt(X**2 + Y**2)
    mayor_ = 1 + amp * np.exp(-((rr - 1.2)**2) / 0.25)
    medio_, menor_ = np.ones_like(X), 0.6 * np.ones_like(X)
    Tt = np.zeros((15, 15, 3, 3))
    Tt[:, :, 0, 0] = mayor_*cc*cc + medio_*ss*ss
    Tt[:, :, 0, 1] = (mayor_ - medio_)*cc*ss
    Tt[:, :, 1, 0] = Tt[:, :, 0, 1]
    Tt[:, :, 1, 1] = mayor_*ss*ss + medio_*cc*cc
    Tt[:, :, 2, 2] = menor_
    ww, vv = np.linalg.eigh(Tt)
    ww, vv = ww[:, :, ::-1], vv[:, :, :, ::-1]
    md = ww.mean(axis=2)
    fa = np.sqrt(1.5 * np.sum((ww - md[:, :, None])**2, axis=2) / np.sum(ww**2, axis=2))
    return ww, vv, md, fa

def glifos(ax, ww, vv, paso=2):
    """Repite el bloque 6 sobre un eje dado."""
    for fila in range(0, 15, paso):
        for columna in range(0, 15, paso):
            e1 = vv[fila, columna, :, 0]
            ang = np.degrees(np.arctan2(e1[1], e1[0]))
            ax.add_patch(Ellipse((X[fila, columna], Y[fila, columna]),
                                 width=0.19*ww[fila, columna, 0], height=0.19*ww[fila, columna, 1],
                                 angle=ang, facecolor="skyblue", edgecolor="navy", alpha=0.85))
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal")

# Control: con amp = 2 se debe reproducir el campo del bloque 5
w2, v2, MD2, FA2 = campo_tensorial(2)
print("¿amp=2 reproduce el bloque 5?", np.allclose(MD2, MD) and np.allclose(FA2, FA))

fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
for fila, amp in enumerate([2, 3]):
    ww, vv, md, fa = campo_tensorial(amp)
    print(f"amp={amp}: MD {md.min():.2f}-{md.max():.2f} | FA {fa.min():.2f}-{fa.max():.2f} | "
          f"autovalor mayor máx. {ww[:, :, 0].max():.2f} | ancho máx. de elipse {0.19*ww[:, :, 0].max():.2f}")
    i1 = axes[fila, 0].imshow(md, origin="lower", extent=[-2, 2, -2, 2], cmap="viridis", vmin=0.8, vmax=1.9)
    axes[fila, 0].set_title(f"MD (amp = {amp})")
    fig.colorbar(i1, ax=axes[fila, 0])
    i2 = axes[fila, 1].imshow(fa, origin="lower", extent=[-2, 2, -2, 2], cmap="magma", vmin=0, vmax=1)
    axes[fila, 1].set_title(f"FA (amp = {amp})")
    fig.colorbar(i2, ax=axes[fila, 1])
    glifos(axes[fila, 2], ww, vv)
    axes[fila, 2].set_title(f"Glifos (amp = {amp})")
print("Separación entre elipses dibujadas (paso 2):", round(8 / 14, 3))
plt.tight_layout()
plt.show()


# ===== Reto 3. Paso del bucle de glifos de 2 a 1 =====
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, paso in zip(axes, [2, 1]):
    glifos(ax, w, v, paso=paso)                 # w, v: campo original del bloque 5
    ax.set_title(f"Paso = {paso}: {len(range(0, 15, paso))**2} glifos")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

for paso in (2, 1):
    separacion = paso * 4 / 14                   # distancia entre posiciones dibujadas
    anchos = 0.19 * w[::paso, ::paso, 0]         # ancho de cada elipse
    print(f"Paso {paso}: separación = {separacion:.3f} | ancho máx. = {anchos.max():.2f} | "
          f"elipses más anchas que la separación: {int((anchos > separacion).sum())} de {anchos.size}")
plt.tight_layout()
plt.show()
