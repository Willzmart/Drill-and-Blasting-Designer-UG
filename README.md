# Diseño de voladura subterránea: avance, chimeneas, piques y producción (v2)

Web app en Streamlit que reúne los métodos de:

- **[C]** Carpio Chávez, A. (2025). *Diseño de perforación y voladura de labores de avance*. Cusco.
- **[LJ]** López Jimeno, C. et al. (1987). *Manual de perforación y voladura de rocas*. IGME, Madrid.
- **[K]** Konya, C. J. *Manual de voladura*, caps. 6–9: burden, espaciamiento, voladura controlada, túneles y tiros.
- **[TL]** Material de capacitación:
  - Curso *Diseño y control de perforación y voladura con taladros largos* (Llontop), sesiones 05–09 y plantillas Excel.
  - Curso *Cálculos para la perforación y voladura en minería subterránea* (Salas).
- **[EN]** Enaex (2016). *Guía del explosivista*.
- **Papers del curso:**
  - Onederra y Chitombo (2007), *Design methodology for underground ring blasting*.
  - Holmberg y Persson (1980).
  - *Comparative analysis of rock fragmentation models* (KCO / Swebrec).
  - *Reparametrización de la ecuación de Kuznetsov*.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Novedades de la versión 2

### Métodos nuevos

| Labor | Método | Ecuaciones |
|---|---|---|
| Avance | **Diseño de Konya** (cap. 9.3), elegible frente al de Holmberg | Ver "Diseño de Konya en avance" |
| Avance, piques | **Cuña quemada de Konya** (barreno vacío grande) | Ver "Cuña quemada de Konya" |
| Pique | **Anillos con barrenos verticales, Konya §9.2** | Ver "Anillos de Konya en piques" |
| Producción | **Criterio Konya** para taladros largos y bancos | Ver "Criterio Konya en producción" |
| Producción | **Ash** | B = Kb·De/12 (Kb 25 para ANFO, 30 para emulsión) [TL] |
| Producción | **Torbica y Lapcevic (2018)** | B = 0.17·PD·r_h/(K·σt), con K = (1−ν)/((1+ν)(1−2ν)) [TL] |
| Producción | **Pearse con PD de Cook** | PD = 423·10⁻⁶·ρe·VOD²/(1+0.8ρe); Std = 4.5·σt; Kv = 1.96 − 0.27·ln(RQD·JSF) [TL] |
| Producción | **Rustan** | B = 11.8·D^0.63 (Onederra y Chitombo 2007) |
| Abanicos | **Retacado escalonado Ts–Tl–Tm–Tl** | Ts = 20D, Tl = 125D, Tm = 50D; T ≤ 2/3·L (AEL / Onederra; curso [TL]) |

**Diseño de Konya en avance (cap. 9.3)**
- Burden: B = 0.012·(2ρe/ρr + 1.5)·De.
- Auxiliares: S = 1.1B, T = 0.5B. Piso: T = 0.2B.
- Recorte: S = 0.45–0.60 m, B = 1.3S, q = D²/12 g/m.
- Ajuste del perímetro (look-out): 0.1 + H·tan 2°.

**Cuña quemada de Konya (barreno vacío grande)**
- D_H = d_H·√N; B1 = 1.5·D_H.
- Cuadros siguientes: B_n = S_c(n−1); R_n = S_c(n−1)/2 + B_n.
- Se agregan cuadros mientras la carga pueda romper el bordo (ec. 6.2).

**Anillos de Konya en piques (§9.2)**
- N_R = R/B; S = B; N_H = 2πR_R/S.
- L = 2B; J = 0.3B; T = 0.5B.

**Criterio Konya en producción**
- Fórmulas: Konya-Myers del curso, B = 3.15·De·∛(ρe/ρr) (pies y pulgadas), o las ecuaciones 6.2 y 6.3.
- Correcciones Kd, Ks y Kr (Tablas 6.2–6.4).
- Espaciamiento por la relación de rigidez L/B (ecs. 7.1–7.4). T = 0.7B, J = 0.3B.

### Correcciones y datos agregados

- **Comparación de criterios de burden:** tabla con los 12 criterios en la memoria de producción, para el mismo diámetro y explosivo.
- **Control de desviación:** la memoria registra la desviación al fondo, α·L. Hay advertencia si pasa de 3 % en taladros largos (el curso acepta 2–3 %; en frentes, hasta 5 %).
- **Factor de atacado del cartucho:** las plantillas del curso usan 0.8 (longitud confinada / nominal). Por defecto queda en 1.0, para no cambiar las validaciones con Carpio.
- **Explosivos:**
  - Se agregan Examon P, Examon V y dinamitas semigelatina 45/65/80 % (datos del curso).
  - Cada explosivo tiene su energía en MJ/kg: ANFO 3.82 [EN]; para los demás, RWS·3.82, que coincide con las fichas de Emulnor dentro de ±2 %.
- **Detonadores:** se agrega la serie Fanel (Famesa) de periodo corto (tabla del curso).
- **Macizo:** σt, ν y Vp medida. VPP crítica = σt·Vp/E.
- **Fragmentación:**
  - Modelo **KCO (Swebrec)**, b = 2·ln2·ln(x_max/X50)·n, como alternativa a Rosin-Rammler.
  - Exponentes de Kuznetsov calibrables, para la reparametrización con datos de mina.
- **Daño al macizo:** ley de Devine, PPV = K·(D/√Q)^−α, con la carga por retardo del secuenciamiento. Da el radio en que se supera la VPP crítica y su cuádruplo (plantilla del curso).

### Pestaña Energía (después de la Vista 3D)

- **Energía por taladro:** Q·e (MJ), energía lineal (MJ/m), MJ/t y kg/t del volumen de influencia. Incluye gráfico de barras por zona.
- **Mapa 2D en un plano:**
  - Planos: paralelo al frente (avance), horizontal (planta, piques) o la sección del anillo. En abanicos, el plano por defecto es el del burden.
  - Variables: concentración 3D de explosivo de **Kleine** (kg/t, MJ/t o kg/m³) o **PPV de campo cercano de Holmberg–Persson** (envolvente por taladro).
  - Contornos de umbral: el factor de carga del diseño o los umbrales de daño y rotura.
- **Vista 3D:** cargas coloreadas por energía lineal e isosuperficie opcional de la zona que supera el umbral.
- **Producción:** se puede analizar una voladura (las filas o anillos por voladura) o toda la malla.

### Exportación a PDF (pestaña Resumen)

El PDF es A4 apaisado y contiene:

1. **Plano acotado de la malla.** Lleva contorno, taladros numerados y coloreados por zona, vacíos, cotas generales, cotas del centro del cuele, origen (0, 0), cuadrícula, escala aproximada y rótulo.
2. **Parámetros de entrada e indicadores**, y distribución por zona.
3. **Columna de carga típica por zona** (taco, carga, retacados) y el mapa de energía de la pestaña Energía.
4. **Cuadro de taladros.** Para cada taladro: N°, zona, x–y, boca (X₀, Y₀, Z₀), azimut, inclinación, longitud, taco, longitud de carga, explosivo, cartuchos, kg, B y S. Si ya se calculó la secuencia, agrega N° de detonador, tiempo y largo de mecha.

## Validaciones con los ejemplos del material

**Konya**
- Fórmula Konya-Myers (curso): 1.11 m (ANFO en 45 mm) y 1.02 m (emulsión en 38.1 mm), igual que la diapositiva de breasting.
- Ec. 6.2: 0.96 m (ejemplo 9.4) y 1.14 m (ejemplo 9.1).
- Ejemplo 9.3: con B = 1.075 m, la cuña tiene 4 cuadros (16 taladros), igual que el libro.

**Plantillas del curso**
- Torbica y Lapcevic: 1.115 m.
- Ash: 3.69 pies.
- Retacado de anillos: Ts = 1.28 m, Tl = 8.0 m y Tm = 3.2 m con 64 mm.

**Kleine:** la integración numérica reproduce la ec. (3) de Onederra y Chitombo (2007).

**Regresión:** los 12 cueles de la versión 1 en avance dan los mismos taladros, avance y kg.

## Supuestos y limitaciones

- **Konya, ec. 9.14:** la profundidad que asegura el vacío usa L = 0.15 + 34.1·Ø − 39.4·Ø². El ejemplo 9.3 indica más avance con 102 mm, así que solo se emite una advertencia y no se recorta la perforación.
- **Konya, ecuaciones 9.1–9.5 de anillos:** el PDF del manual no las trae legibles. Se usan las relaciones del texto: N_R = R/B, S = B y N_H = 2πR/S.
- **Pearse del curso:** se aplica la raíz cuadrada de las diapositivas. La plantilla Excel escribe "^1/2", que Excel evalúa como división entre 2.
- **Concentración de Kleine:** es un índice estático (no considera los retardos). Crece con el tamaño del patrón, por eso en producción conviene analizar una voladura o un anillo.
- **K y α de Holmberg–Persson:** valores referenciales (700 y 0.7; la plantilla usa 470 y 0.94). Calíbrelos con monitoreo.
- **Dinamitas:** la RWS es referencial. La energía de los explosivos se estima con RWS·3.82 MJ/kg. Verifique las fichas del fabricante.
- **Modelo ML:** el sustituto incluido es de demostración, no un modelo real.
