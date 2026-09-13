# V-Umlaut — Fundamentalmatrix-Schätzung

Eigenständige Implementierung und Weiterentwicklung des in einem CVPR-2026-Paper vorgeschlagenen
**heuristischen, minimalen, linearen Sieben-Punkt-Solvers** zur Schätzung der Fundamentalmatrix
(V-Umlaut-Konfiguration), eingebettet in eine selbst entwickelte **LO-RANSAC-Pipeline**.

Entstanden im Rahmen eines zweiwöchigen universitätsinternen Forschungspraktikums, seitdem
eigenständig weitergeführt.

## Motivation

Klassische 7- und 8-Punkt-Solver zur Fundamentalmatrix-Schätzung sind entweder nicht-linear oder nicht-minimal. Das V-Umlaut-Paper schlägt einen minimalen,
rein linearen Sieben-Punkt-Ansatz vor.

Beim Nachimplementieren zeigte sich beim Ground-Truth-Abgleich: **das Paper selbst enthielt
Formelfehler**. Durch eigenständige mathematische Herleitung wurden diese identifiziert und
korrigiert. Daraus entstand die Grundlage für eine eigene, leistungsfähigere **matrixbasierte
Lösungsmethode** mit nicht-minimalem Refitting.

## Methodik

Der Solver (`VUmlaut` in `models/v_umlaut_solver.py`) arbeitet in mehreren Schritten:

1. **Perspektivische Normalisierung** — die ersten vier Korrespondenzen jedes Bildes werden über
   eine Homographie (`H1`, `H2`) auf eine kanonische Basis abgebildet.
2. **Erste baryzentrische Umformung (Virtuelle Korrespondenzen)** — vor der perspektivischen Normalisierung werden `p6`/`q6`/`p7`/`q7`
   als baryzentrische 2D-Midpoints auf den Verbindungsgeraden `p1–p2` bzw. `p1–p3` konstruiert, 
   gewichtet mit `v_umlaut_sij`.
3. **Zweite baryzentrische Umformung** — nach der perspektivischen Transformation (`H1`, `H2`) werden die tranformierten 
   virtuellen Punkte in barycentrische Koordinaten überführt.
4. **Lösen des linearen Systems** auf zwei Wegen:
    - `symbolic_method` — symbolische Lösung via SymPy (`sp.linsolve`)
    - `matrix_method` — geschlossene 5×5-Matrixlösung via `numpy.linalg.lstsq`,
      erweiterbar auf beliebig viele zusätzliche Korrespondenzen (nicht-minimales Refitting)
5. **Denormalisierung**: `F = H2.T @ F_norm @ H1`

Der Parameter `v_umlaut_sij` wird nicht analytisch bestimmt: Während der lokalen Optimierung
in RANSAC wird er per Zufallssuche im Bereich `[0.4, 0.6]` mitoptimiert, und mit jedem getesteten
Wert erfolgt anschließend ein nicht-minimales Refitting auf den aktuellen Inliern (siehe unten).

## LO-RANSAC-Pipeline

`estimator/ransac.py` implementiert eine eigene RANSAC-Variante mit lokaler Optimierung:

- Sampling von 5-Punkt-Teilmengen aus den Korrespondenzen (`Features.__next__`)
- Inlier-Bewertung wahlweise über **Sampson-Distanz** oder **symmetrische Reprojektionsfehler**
  (`estimator/estimator_pipeline.py`, Klasse `Evaluator`)
- Bei neuem Bestwert: **lokale Optimierung** — mehrere zufällige `v_umlaut_sij`-Werte werden
  getestet und das Modell auf den aktuellen Inliern nicht-minimal neu gefittet
  (`non_minimal_refit`); nur Verbesserungen werden übernommen
- Abbruch nach einer festen Anzahl Iterationen ohne Verbesserung (`iters_after_best`). Eine Verbesserung
  über Erfolgswahrscheinlichkeit p (Confidence) ist in Planung

Zur Bewertung dient außerdem ein skalen-/vorzeichenkorrigierter **L2-Fehler** zwischen
geschätzter und Ground-Truth-F (`Evaluator.l2_loss`).

## Datenquellen

- **Feature-Matching** — SIFT + FLANN auf realen Bildpaaren (`feature_extraction/features.py`)
- **Synthetischer Szenengenerator** (`data/inn/synthetic/synthetic_data.py`) — erzeugt
  Kamerapaare, 3D-Weltpunkte und daraus projizierte Korrespondenzen inklusive exakter
  Ground-Truth-F, wahlweise mit Pixelrauschen und injizierten Ausreißern; unterstützt sowohl
  allgemeine Szenen als auch ein WAMI-Setup (lange Brennweite, hohe Kameraposition)

## Ergebnisse

Erste Tests unter kontrollierten Bedingungen gegen OpenCVs `findFundamentalMat` (RANSAC):

| Methode | L2-Fehler |
|---|---|
| OpenCV RANSAC | Baseline |
| V-Umlaut LO-RANSAC (eigene Pipeline) | gleiche Größenordnung |

Getestet sowohl auf synthetischen Szenen (bekannte Ground Truth) als auch auf realen Bildpaaren
(u. a. dem `squirrel`-Testdatensatz).

> Eine umfassendere Evaluierung inklusive Vergleich gegen weitere SOTA-Modelle über
> OpenCV RANSAC hinaus, ist in Planung.

## Projektstruktur

```
├── estimator/
│   ├── estimator_pipeline.py   # Estimator-Basisklasse + Evaluator (Sampson distance, Reprojektionsfehler, L2-Loss)
│   └── ransac.py                # LO-RANSAC-Pipeline
├── feature_extraction/
│   └── features.py              # SIFT+FLANN-Matching, synthetische Korrespondenzgenerierung
├── models/
│   ├── model.py                  # abstrakte Model-Basisklasse
│   └── v_umlaut_solver.py        # VUmlaut-Solver (symbolisch + matrixbasiert)
├── data/inn/
│   ├── real/squirrel_images_and_data/   # reale Testbildpaare
│   └── synthetic/synthetic_data.py       # synthetischer Szenengenerator (Ground-Truth F)
└── main.py                        # Einstiegspunkt / Testläufe
```

## Verwendung

```python
from os import path
from MyHelpers.Frame import Frame
from estimator.ransac import Ransac
from feature_extraction.features import Features
from models.v_umlaut_solver import VUmlaut

path_ = "data/inn/real/squirrel_images_and_data"
frame1 = Frame(path.join(path_, "image_34.jpg"), "frame1")
frame2 = Frame(path.join(path_, "image_33.jpg"), "frame2")

features = Features(5, 10000, frame1, frame2, Features.EXTRACTOR_SIFT)
model = VUmlaut(VUmlaut.METHOD_MATRIX)
ransac = Ransac(model, features, projection_threshold=1e-4, useLO=True)

for _ in range(10000):
    if ransac.best_inlier_count == features.features.shape[0]:
        break
    next(ransac)

print(model.model)  # geschätzte Fundamentalmatrix
```

Für synthetische Ground-Truth-Tests: `Features.EXTRACTOR_SYNTHETIC` statt `EXTRACTOR_SIFT`
verwenden (siehe `main.py` für ein vollständiges Beispiel inkl. Vergleich gegen OpenCV).


## Status

Laufendes Forschungsprojekt — begonnen als zweiwöchiges universitätsinternes Praktikum,
seitdem in Eigenregie weitergeführt. Aktuell in Arbeit: umfassendere experimentelle Evaluierung,
u.a. Vergleich gegen weitere SOTA-Modelle.

## Referenz

Basierend auf dem CVPR-2026-Paper zur linearen Fundamentalmatrix-Schätzung aus 7 bzw. 5 Punkten
(V-Umlaut-Methode).