# Ball Tree y KD-Tree

Este proyecto implementa y compara dos estructuras de datos ampliamente utilizadas para búsquedas de vecinos más cercanos en espacios multidimensionales: **KD-Tree** y **Ball Tree**.

## 📚 Descripción del Proyecto

Las estructuras KD-Tree y Ball Tree se utilizan para acelerar búsquedas de tipo *nearest neighbor*, especialmente en problemas de clasificación, clustering y reducción de dimensionalidad. Ambas permiten organizar puntos en espacios multidimensionales de manera eficiente, pero tienen diferentes comportamientos según la dimensionalidad del espacio.

Este proyecto:
- Implementa ambas estructuras desde cero.
- Evalúa su rendimiento con distintos tamaños y dimensiones de datasets.
- Analiza cómo cada estructura se comporta en espacios de baja y alta dimensionalidad.

## ⚙️ Instrucciones de Instalación y Ejecución

### Requisitos

- Python 3.10+
- Instalar dependencias:

```bash
pip install -r requirements.txt
```

### Ejecutar pruebas

```bash
pytest -v
```

### Generar datasets sintéticos

```bash
python tests/generate_points.py
```

### Ejecutar benchmarks y análisis de rendimiento

```bash
python -m tests.driver
python -m tests.dimension_analysis
```

## 📁 Estructura del Proyecto

```
ball-and-kd-tree/
├── ball_tree/               # Implementación del Ball Tree
│   ├── ball_tree.py
│   ├── ball_node.py
├── kd_tree/                 # Implementación del KD-Tree
│   ├── kd_tree.py
│   ├── kd_node.py
├── tests/                   # Scripts de prueba y benchmarking
│   ├── test_balltree.py
│   ├── test_kdtree.py
│   ├── generate_points.py
│   ├── dimension_analysis.py
│   ├── driver.py
│   └── datasets/            # Datasets generados
│       ├── dataset_2D.csv/.npy
│       ├── dataset_10D.csv/.npy
│       └── dataset_50D.csv/.npy
├── requirements.txt         # Dependencias del proyecto
├── main.py                  # Punto de entrada (opcional)
└── README.md
```

## 🧪 Documentación de la API Pública

### Clase KDTree

```python
class KDTree:
    KDTree(points: List[Tuple[float]])
        # Inicializa el árbol KD a partir de una lista de puntos.
        # Lanza ValueError si los puntos no tienen dimensiones consistentes.

    nearest_neighbor(target: Tuple[float]) -> Tuple[float, Tuple[float]]
        # Busca el vecino más cercano al punto objetivo.
        # Retorna una tupla (distancia, punto más cercano).
        # Si el árbol está vacío, retorna (inf, None).

    k_nearest_neighbors(target: Tuple[float], k: int) -> List[Tuple[float, Tuple[float]]]
        # Retorna una lista con los k vecinos más cercanos.
        # La lista contiene tuplas (distancia, punto) ordenadas por cercanía.

    insert(point: Tuple[float])
        # Inserta un nuevo punto en el árbol.
        # Puede reconstruir el árbol si se alcanza un umbral logarítmico de inserciones.
        # Lanza ValueError si las dimensiones del punto no coinciden.

    rebuild()
        # Reconstruye completamente el árbol con los puntos actuales.

    check_rebuild() -> bool
        # Verifica si se alcanzó el umbral para reconstrucción.
```

### Clase BallTree

```python
class BallTree:
    BallTree(data: List[List[float]], node_size: int = 10)
        # Construye un Ball Tree a partir de una lista de puntos multidimensionales.
        # node_size determina cuántos puntos caben en una hoja.

    nearest_neighbor(target: List[float]) -> Tuple[List[float], float]
        # Retorna el punto más cercano al objetivo y la distancia euclidiana.
        # Si el árbol está vacío, retorna (None, inf).

    k_nearest_neighbors(target: List[float], k: int) -> List[Tuple[float, List[float]]]
        # Retorna una lista con los k vecinos más cercanos.
        # Cada elemento es una tupla (distancia, punto), ordenados por cercanía.

    insert(point: List[float])
        # Inserta un nuevo punto en el árbol.
        # Si se supera el umbral de inserciones, el árbol se reconstruye.

    check_rebuild() -> bool
        # Verifica si debe reconstruirse el árbol según la cantidad de nuevas inserciones.

    rebuild()
        # Reconstruye completamente el árbol con los datos actuales.
```

## 📊 Resultados

Los resultados del análisis de rendimiento y dimensionalidad se encuentran en la carpeta `tests/results/`:
- `benchmark_kdtree_balltree_comparison.csv`
- `dimensionality_analysis.png`
