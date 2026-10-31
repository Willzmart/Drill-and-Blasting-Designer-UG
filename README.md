# Diseño de voladura subterránea — avance, chimeneas, piques y producción

Web app en Streamlit que reúne los métodos de:

- **[C]** Carpio Chávez, A. (2025). *Diseño de perforación y voladura de labores de avance*. Cusco.
- **[LJ]** López Jimeno, C. et al. (1987). *Manual de perforación y voladura de rocas*. IGME, Madrid.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tipos de labor y métodos

| Labor | Métodos | Base |
|---|---|---|
| **Avance** | 12 cueles: cilíndrico de 4 secciones (versión Carpio y versión LJ con desviación), quemado simple, Sarrois, Sueco, Fagersta, espiral, Coromant, cráter, en V, en abanico, instantáneo piramidal. Tres métodos de carga (Carpio q₀, Carpio factores, LJ fondo + columna). Contorno con o sin voladura controlada. Modo labor estrecha | [C] cap. 3; [LJ] cap. 22 |
| **Chimenea** | Manual clásico (V/abanico), jaula Jora, plataforma Alimak, taladros largos con cuele de vacíos, cuele cráter, VCR, combinado (raise boring + destroza) | [LJ] cap. 23 §3 |
| **Pique** | Banqueo, espiral, sección completa (cuele en V, cónico, barrenos paralelos, barreno o chimenea central) | [LJ] cap. 23 §2 |
| **Producción** | Taladros largos LBH, taladros largos en vetas angostas, subniveles con abanicos, VCR, cámaras y pilares (banqueo vertical y frontal), corte y relleno (realce y breasting), cámaras de obra pública | [LJ] cap. 24 y cap. 20 (Langefors); [C] ec. 06 (Pearse con Kv) |

## Pestañas

- **Memoria de cálculo**: cada fase con su ecuación y la referencia al libro.
- **Fragmentación (P80)**:
  - Kuz-Ram y ruta ML (modelo propio + Evolución Diferencial). Avance, piques y producción optimizan el burden; chimeneas sólo verifican que el bloque mayor (P99) no atasque la sección.
  - Rangos de P80 de Carpio 6.1.7 como referencia.
- **Vista 2D**: frente, planta o sección del abanico.
- **Vista 3D y carguío**: taladros con retacado, cargas (incluye cargas seccionadas y esféricas), cebo y línea de iniciación; columna de carga por zona.
- **Secuenciamiento**:
  - Métodos: MILP (HiGHS), heurística o por filas (Du Pont).
  - Detonadores Exsanel MS/LP (Carpio, Tabla 05) con el tiempo mínimo entre taladros del arranque (Tabla 04).
  - Carmex con longitudes de mecha.
  - Daño al shotcrete (Carpio cap. V).
- **Resumen y exportación**: CSV y JSON.

## Validaciones con los ejemplos de los libros

- Avance: el Ejemplo N°06 de Carpio (baúl 4.5 × 4.0 m) reproduce c, q₀, cuadrantes y burdens de tajeo (diferencias por redondeo).
- Cuele de 4 secciones de LJ: aberturas efectivas A_h = √2·(X − E_p), como en su ejemplo F (0.10 / 0.28 / 0.70 m).
- Chimenea VCR con 114 mm: S = 1.20 m, X = 2.0 m, T = 1.37 m (Tabla 23.1: 1.2 / 2.1 / 1.5 m).
- Piques: consumo específico resultante cercano a la Fig. 23.6 de Wild.
- Daño al shotcrete: V crítica ≈ 335 mm/s y d ≈ 4.35 m (ejemplo de Carpio: 340 mm/s y 4.31 m).

## Supuestos y limitaciones

- Las geometrías de los cueles Fagersta, espiral y Coromant son aproximadas a partir de las figuras.
- Langefors usa ĉ = c + 0.75 (B ≥ 1.4 m) como lo imprime LJ; en la zona ambigua se adopta B = 1.4 m.
- Pearse con Kv entrega piedras de arranque; en producción se advierte y se recomienda comparar con Langefors.
- En abanicos, Kuz-Ram usa el espaciamiento de fondo y la carga media por sección.
- Las curvas de las Figs. 23.5 y 23.6 están digitalizadas.
- Constantes de vibración, combustión de mecha y costos son referenciales.
- El sustituto ML de demostración no es un modelo real.
