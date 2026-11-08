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

### Labor de perforación, pivote y carrusel (taladros largos)

Aplica a LBH, taladros largos en vetas angostas, abanicos y VCR. Se configura en la barra lateral, en el panel "Labor de perforación y equipo".

- **Sección de la labor:** se define igual que en el módulo de avance: baúl, medio punto, herradura, rectangular o trapezoidal, con secciones típicas o medidas propias.
- **Equipo:**
  - punto de pivote a una altura sobre el piso (1.3–1.5 m típico; hay advertencia fuera de ese rango);
  - rectángulo del carrusel (ancho, alto y desfase vertical respecto al pivote), que depende de cada equipo.
- **Abanicos:**
  - todos los taladros del anillo salen del pivote;
  - la boca de cada taladro está donde la recta pivote–fondo corta el contorno de la labor, y la longitud perforada se mide desde esa boca;
  - la vista 2D muestra la labor, el pivote, el carrusel y la trayectoria pivote–boca;
  - hay advertencia si el carrusel no cabe en la labor.
- **Taladros paralelos (LBH, vetas, VCR):**
  - se perforan desde galería(s) de perforación o desde una cámara de ancho completo; el número de galerías es automático o dato;
  - para cada taladro, el pivote se ubica sobre la prolongación de su eje, a la altura del pivote sobre el piso de la labor;
  - si el carrusel no cabe en la labor, el taladro queda "fuera del alcance" (en rojo y con advertencia);
  - hay una vista adicional en sección transversal en la pestaña Vista 2D y una lámina en el PDF.
- **Vista 3D:** las labores de perforación se dibujan extruidas a lo largo del tajeo.

### Chimenea slot con Simba (producción) y capacidad del equipo

- **Método nuevo en Producción: "Chimenea slot con Simba (taladros largos)".** Basado en las plantillas del curso de cálculos de perforación y voladura (sesiones 7, 8 y 16).
  - **Sección:** rectangular o circular (2 × 2 m por defecto), con longitud igual a la altura entre subniveles. La perforación puede ser descendente o ascendente.
  - **Cuele:** vacíos de 4" y cargados de 2.5", con las proporciones de Konya: B1 = 1.5·D_H, B_n = S_c(n−1) y taco por cuadro. Como alternativa, el cuele de López Jimeno §3.2-A.
  - **Contorno:** taladros cuadradores simétricos (8 en una sección de 2 × 2 m, como la plantilla).
  - **Voladura:** por tramos, integral, o integral con cargas espaciadas por tacos intermedios. Incluye booster por carga.
  - **Esponjamiento:** el volumen roto esponjado (30 %) se compara con el volumen disponible en la labor inferior.
  - **Ensanche opcional a cámara slot** (sesión 8), con el burden de Konya.
  - **KPI:** índice de perforación (t/m), barras por taladro, factor de carga y costos.
- **Capacidad del equipo:** Simba, por defecto 102 pies (≈ 31.1 m), con barras de 4–6 pies. Se aplica en LBH, vetas angostas, abanicos, VCR, chimenea slot y chimenea por taladros largos.
  - La memoria indica la capacidad y las barras del taladro más largo.
  - Hay advertencia si algún taladro la supera (subnivel intermedio o perforación desde ambos niveles).
  - LBH y VCR pasan a 30 m de altura por defecto para quedar dentro de esa capacidad.

### Chimeneas inclinadas (dip)
- Labor *Chimenea* (todos los métodos) y *Chimenea slot con Simba* (sola): deslizador "Inclinación de la chimenea —
  dip" (90° = vertical). La malla se diseña en la sección perpendicular al eje y los taladros son paralelos al eje.
- Memoria: longitud según el eje, altura vertical (L·sen δ), desplazamiento horizontal (L·cos δ), avance vertical por
  pega y alargamiento de la sección horizontal (1/sen δ).
- Avisos por método: manual y Alimak ≥ 45°, taladros largos / cráter / raise boring ≥ 55°, VCR ≥ 60°, jaula Jora
  ≥ 70° (cuelga del cable del barreno central), y < 50° el material volado se desliza con dificultad.
- Vista 3D girada con la inclinación real y cuadro de taladros del PDF con coordenadas, azimut e inclinación reales.
  En la chimenea slot dentro de producción, la inclinación la da el método (taladros paralelos inclinados).

### Apartado "Macizo rocoso": un dominio o por dominios
- **Un solo dominio litológico** (modo anterior): todo el diseño usa las mismas propiedades. Sirve para cuerpos
  masivos, redondeados o irregulares, o cuando no hay datos de las cajas. No se calcula sobrerotura por dominios.
- **Por dominios: mineral, caja techo y caja piso**: cada dominio tiene RMR, GSI, RQD, σci, ρ, Q, E, σt y Vp (con su
  VPP crítica). Cada método usa la información que le corresponde:
  - Avance, chimenea y pique: las propiedades del dominio donde se excava la labor (mineral, caja techo o caja piso).
  - Producción: burden y carga con el mineral; las cajas alimentan el expander de sobrerotura, dilución y
    estabilidad (taladros largos, corte y relleno, cámaras y pilares; en mantos, techo = caja techo y piso = caja piso).
  - Obras públicas (banqueo) usa el mineral.

### Macizo por dominios: caja techo, caja piso y techo (producción con taladros largos)
- El apartado *Macizo rocoso* es el **mineral**. En el expander *Macizo por dominios* se ingresan la caja techo, la
  caja piso y, en abanicos ascendentes, el techo / corona (σt, E, ρ, Q o Vp) con su VPP crítica σt·Vp/E.
- Con Holmberg–Persson (K, α del sitio) se calcula el radio de sobrerotura de cada taladro de borde en la caja
  (umbral = n × VPP crítica, n = 4 por defecto) y su distancia efectiva al contacto (− emboquille − desviación·s).
  ELOS de cada pared = promedio en la fila de borde; dilución = estéril / (mineral + estéril) por metro de corrida;
  ley diluida = ley · (1 − D).
- Taladros de borde: sin ajuste, factor manual o **automático** (ELOS objetivo): baja la carga lineal en taladros
  paralelos al contacto o recorta la carga del extremo que llega al contacto en abanicos.
- Se dibujan las cajas y las líneas de sobrerotura en la planta / anillo, en la sección y en el PDF (con tabla de
  dominios). En Energía (PPV) aparecen los umbrales de sobrerotura de cada caja. Avisos: carga dentro de la caja,
  ELOS > 0.5 m y riesgo de subrotura (mineral pegado).
- En tresbolillo ya no se coloca un taladro sobre el contacto (fila desplazada a menos de S/4 de la caja).
- **Corte y relleno (realce y breasting)**: también usa caja techo, caja piso y mineral. Las cajas diluyen; el techo
  de la tajada es mineral de la siguiente tajada, así que su daño no diluye pero se reporta como daño al techo de
  trabajo (desatado y sostenimiento). En realce el modo automático recorta la carga del extremo de los taladros que
  llega al techo; el piso es relleno y no se evalúa.
- **Cámaras y pilares (banqueo vertical y frontal)**: las paredes son los pilares (mineral) y se evalúan además el
  techo (a banco + galería piloto) y el piso, cada uno marcado como estéril (diluye) o mineral. Con la sobrerotura de
  los pilares se calcula el ancho efectivo w_ef = w − 2·ELOS y el FS del pilar por área tributaria
  (σp = γ·H·((w + Wc)/w_ef)²) con Hedley–Grant (K·w^0.5/h^0.75) y Lunder–Pakalnis (0.44·σci·(0.68 + 0.52·κ)),
  antes y después de la voladura; el techo se compara con la abertura máxima de Barton 2·ESR·Q^0.4 (cámara real
  Wc + 2·ELOS y diagonal en las intersecciones). Tabla de pilares en el PDF.
- **VCR sin chimenea slot**: su cara libre es el corte inferior (undercut).

### Vista 3D a escala real, galerías de cada nivel y nuevas láminas del PDF
- **Vista 3D (y 3D de energía) con la misma escala en X, Y y Z**: ya no se comprimen los ejes; el diseño se ve en
  sus proporciones reales.
- **Taladros largos: una sola labor de perforación por nivel** (sin galerías superpuestas). El tajeo se dibuja como
  prisma de su sección real (inclinada si los taladros paralelos lo están) y la chimenea slot con su inclinación.
- **Galería del otro nivel** (expander *Labor de perforación y equipo*): galería inferior de extracción / undercut
  bajo el tajeo en taladros paralelos y VCR; en abanicos, galería superior (ascendentes), inferior (descendentes) o
  ambas. Sección, ancho, alto y desplazamiento lateral configurables; aparece en 3D, en la sección y en el PDF.
- **PDF**: dos láminas nuevas después del plano general:
  1. *Estándar de malla de perforación* para replanteo en el frente: cotas **solo horizontales y verticales**
     desde dos ejes perpendiculares (eje de la labor y línea horizontal por el centro del cuele; en planta, la cara
     libre x = 0 y el eje del tajeo y = 0). Cada fila de taladros lleva una cadena horizontal desde el eje vertical;
     la altura de cada fila va en una cadena vertical al margen; los taladros alineados en columna (hastiales,
     abanicos) se acotan a lo largo de la columna. Se indican la distancia del eje a la pared y la altura de la
     línea de referencia sobre el piso, y un detalle ampliado del cuele. En tajeos muy largos se acota un tramo típico.
  2. *Distribución del accesorio de iniciación*: tiempo de salida de cada taladro (ms) y flechas de la secuencia.
     Si no se calculó la secuencia, se usa una por defecto (heurística; por filas en producción).

### Chimenea slot integrada, giro del carrusel y carguío de taladros largos
- **Chimenea slot en cada método de taladros largos** (LBH, vetas angostas, abanicos/subniveles; el VCR no la usa:
  su cara libre es el corte inferior): expander *Chimenea slot (cara libre inicial)*. La chimenea se diseña con el
  cuele Konya o LJ, se ensancha de caja a caja (cámara slot) y la primera fila/anillo de producción sale hacia ella
  (x = 0) con el burden y espaciamiento de producción. Se dibuja en planta, en 3D y en el PDF; su explosivo y metraje
  se reportan aparte y no entran en la carga por voladura de producción. En la secuencia sale primero.
- **Taladros paralelos inclinados**: inclinación 45–90° y sentido (±y) para LBH y vetas; perforación **pasante**
  (conecta las dos labores, tapón de fondo 0.3 m) o **ciega** (puente de roca configurable). La sección del tajeo
  se dibuja como paralelogramo y la chimenea slot toma la misma inclinación.
- **Carrusel que gira en dip**: el rectángulo del carrusel se gira con el eje de cada taladro (paralelos y
  abanicos) y se verifica que no choque con paredes ni techo de la labor; los taladros con choque se reportan.
  En abanicos se puede inclinar el plano del anillo (dip 60–90°).
- **Carga en toda la longitud**: en taladros largos la columna es continua desde el taco de boca hasta el fondo
  (con el iniciador en el fondo); el límite de carga es la longitud total del taladro. Las cargas espaciadas
  quedan como opción. El VCR mantiene cargas esféricas (L ≤ 6D).

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
