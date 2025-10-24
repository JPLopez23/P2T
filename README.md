# Proyecto 2 – Teoría de la Computación

Este proyecto implementa un analizador sintáctico parser basado en el algoritmo **CYK** para determinar si una oración pertenece a un lenguaje generado por una gramática libre de contexto CFG, convertida previamente a su **Forma Normal de Chomsky (CNF)**.


## Instalación

1. Clona el repositorio: 
2. corre el archivo P2.py
3. Prueba el programa

## Estructura

* **Grammar**: Representa la gramática libre de contexto.
* **CNFConverter**: Convierte la CFG a CNF.
* **CYKParser**: Implementa el algoritmo CYK para el análisis sintáctico.
* **ParseTreeNode**: Genera y muestra el árbol de derivación.

## Ejemplos

* **Entrada**: `she eats a cake`
* **Resultado**: "Frase aceptada" (si pertenece al lenguaje)
