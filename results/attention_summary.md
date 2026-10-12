# Attention, drift and tile probes on CAMELYON16 test tumour slides

Tumour tiles are at least 50% annotated tumour. Attention numbers are means over test tumour slides and seeds.

## Refitted heads against the committed run (clean test AUC, mean over seeds)

| encoder | training | committed | refitted |
|---|---|---|---|
| dinov2-l | clean | 77.3 | 77.3 |
| dinov2-l | mixed | 76.4 | 76.4 |
| kaiko-b16 | clean | 98.6 | 98.6 |
| kaiko-b16 | mixed | 98.5 | 98.5 |
| phikon | clean | 98.9 | 98.9 |
| phikon | mixed | 98.8 | 98.8 |
| phikon-v2 | clean | 96.4 | 96.4 |
| phikon-v2 | mixed | 93.8 | 93.8 |

## Attention tile AUC: tumour tiles against tumour-free tiles

| encoder | training | clean | blur:2 | blur:4 | jpeg:30 | jpeg:15 | jpeg:10 | stain:0.3 |
|---|---|---|---|---|---|---|---|---|
| dinov2-l | clean | 88.2 | 71.9 | 56.8 | 82.8 | 80.8 | 76.7 | 87.9 |
| dinov2-l | mixed | 85.8 | 71.4 | 49.8 | 83.4 | 83.0 | 80.6 | 85.9 |
| kaiko-b16 | clean | 97.6 | 96.1 | 73.7 | 93.0 | 84.2 | 78.4 | 97.5 |
| kaiko-b16 | mixed | 98.4 | 97.3 | 44.2 | 94.6 | 84.5 | 78.8 | 98.0 |
| phikon | clean | 93.1 | 81.1 | 79.3 | 84.2 | 72.9 | 67.8 | 94.0 |
| phikon | mixed | 95.9 | 88.9 | 47.6 | 92.1 | 82.0 | 73.2 | 96.6 |
| phikon-v2 | clean | 95.8 | 89.0 | 70.0 | 91.1 | 80.2 | 76.6 | 95.2 |
| phikon-v2 | mixed | 95.5 | 90.9 | 54.1 | 90.6 | 77.8 | 70.7 | 94.7 |

## Most-attended tile touches tumour (% of slides)

| encoder | training | clean | blur:2 | blur:4 | jpeg:30 | jpeg:15 | jpeg:10 | stain:0.3 |
|---|---|---|---|---|---|---|---|---|
| dinov2-l | clean | 64.1 | 50.2 | 22.4 | 58.8 | 54.7 | 49.0 | 62.4 |
| dinov2-l | mixed | 57.6 | 47.3 | 20.0 | 57.6 | 58.4 | 53.5 | 56.7 |
| kaiko-b16 | clean | 93.1 | 92.7 | 38.0 | 92.7 | 75.1 | 60.8 | 93.5 |
| kaiko-b16 | mixed | 92.2 | 91.0 | 27.8 | 91.0 | 83.3 | 70.6 | 91.8 |
| phikon | clean | 93.1 | 82.0 | 38.8 | 84.1 | 67.8 | 48.6 | 93.9 |
| phikon | mixed | 94.3 | 83.7 | 27.3 | 88.2 | 78.0 | 63.7 | 94.7 |
| phikon-v2 | clean | 91.0 | 84.1 | 35.1 | 80.0 | 56.7 | 42.4 | 92.2 |
| phikon-v2 | mixed | 86.9 | 81.6 | 28.2 | 80.0 | 55.9 | 35.5 | 88.2 |

## Attention on tumour tiles / their share of the slide

| encoder | training | clean | blur:2 | blur:4 | jpeg:30 | jpeg:15 | jpeg:10 | stain:0.3 |
|---|---|---|---|---|---|---|---|---|
| dinov2-l | clean | 15.2x | 8.3x | 1.5x | 15.2x | 16.4x | 9.3x | 14.9x |
| dinov2-l | mixed | 8.1x | 4.4x | 1.4x | 10.9x | 12.3x | 9.0x | 8.4x |
| kaiko-b16 | clean | 112.5x | 100.7x | 3.8x | 85.0x | 27.2x | 10.8x | 112.0x |
| kaiko-b16 | mixed | 103.0x | 87.5x | 1.7x | 97.2x | 41.6x | 17.6x | 99.7x |
| phikon | clean | 112.6x | 43.7x | 2.8x | 58.4x | 17.3x | 6.4x | 110.5x |
| phikon | mixed | 111.6x | 63.6x | 1.0x | 86.2x | 48.2x | 18.4x | 104.3x |
| phikon-v2 | clean | 84.6x | 39.4x | 2.3x | 29.4x | 7.9x | 2.9x | 85.9x |
| phikon-v2 | mixed | 79.4x | 53.2x | 1.5x | 41.7x | 8.0x | 3.1x | 73.3x |

## Drift: cosine between degraded and clean tile embeddings (tumour / tumour-free)

| encoder | blur:2 | blur:4 | jpeg:30 | jpeg:15 | jpeg:10 | stain:0.3 |
|---|---|---|---|---|---|---|
| dinov2-l | 0.49 / 0.56 | 0.03 / 0.18 | 0.76 / 0.77 | 0.52 / 0.59 | 0.36 / 0.47 | 0.97 / 0.96 |
| kaiko-b16 | 0.74 / 0.70 | 0.23 / 0.15 | 0.62 / 0.58 | 0.39 / 0.40 | 0.30 / 0.32 | 0.90 / 0.87 |
| phikon | 0.66 / 0.61 | 0.22 / 0.16 | 0.61 / 0.54 | 0.39 / 0.35 | 0.28 / 0.26 | 0.90 / 0.87 |
| phikon-v2 | 0.48 / 0.42 | 0.21 / 0.14 | 0.40 / 0.33 | 0.23 / 0.18 | 0.17 / 0.13 | 0.91 / 0.88 |

## Tile probe: tile AUC, tumour against normal-slide tiles

| encoder | training | clean | blur:2 | blur:4 | jpeg:30 | jpeg:15 | jpeg:10 | stain:0.3 |
|---|---|---|---|---|---|---|---|---|
| dinov2-l | clean | 98.4 | 93.5 | 64.9 | 97.0 | 94.1 | 90.4 | 98.1 |
| dinov2-l | mixed | 98.1 | 96.7 | 87.2 | 98.0 | 97.1 | 95.6 | 97.9 |
| kaiko-b16 | clean | 99.4 | 98.5 | 63.1 | 98.2 | 95.9 | 94.9 | 99.6 |
| kaiko-b16 | mixed | 99.5 | 98.6 | 93.2 | 99.3 | 98.8 | 98.4 | 99.5 |
| phikon | clean | 98.4 | 91.8 | 72.6 | 95.1 | 88.5 | 81.1 | 96.8 |
| phikon | mixed | 97.9 | 94.5 | 93.8 | 98.8 | 98.2 | 97.2 | 98.5 |
| phikon-v2 | clean | 98.4 | 93.1 | 79.9 | 96.7 | 95.5 | 94.1 | 98.7 |
| phikon-v2 | mixed | 98.7 | 96.6 | 92.2 | 98.6 | 97.9 | 96.9 | 99.1 |
