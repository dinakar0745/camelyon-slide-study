# Deep dive: tumour size, hospital and shared failures

Sensitivity is at 95% specificity: the threshold lets at most 5% of the 80 normal test slides through. Means over seeds.

## Sensitivity at 95% specificity, all 49 tumour slides

| encoder | head | training | clean | blur:2 | blur:4 | jpeg:30 | jpeg:15 | jpeg:10 | stain:0.3 |
|---|---|---|---|---|---|---|---|---|---|
| dinov2-l | abmil | clean | 53 | 45 | 16 | 49 | 47 | 44 | 56 |
| dinov2-l | abmil | mixed | 49 | 45 | 31 | 49 | 50 | 49 | 50 |
| dinov2-l | tileprobe | clean | 54 | 41 | 10 | 52 | 47 | 36 | 54 |
| dinov2-l | tileprobe | mixed | 49 | 48 | 25 | 49 | 48 | 47 | 53 |
| kaiko-b16 | abmil | clean | 93 | 90 | 28 | 82 | 55 | 45 | 95 |
| kaiko-b16 | abmil | mixed | 92 | 90 | 31 | 89 | 73 | 57 | 93 |
| kaiko-b16 | tileprobe | clean | 79 | 62 | 11 | 78 | 66 | 49 | 81 |
| kaiko-b16 | tileprobe | mixed | 73 | 68 | 47 | 79 | 74 | 69 | 72 |
| phikon | abmil | clean | 96 | 75 | 21 | 80 | 49 | 37 | 98 |
| phikon | abmil | mixed | 95 | 84 | 27 | 84 | 73 | 60 | 96 |
| phikon | tileprobe | clean | 76 | 49 | 13 | 62 | 52 | 44 | 74 |
| phikon | tileprobe | mixed | 74 | 69 | 44 | 71 | 68 | 61 | 76 |
| phikon-v2 | abmil | clean | 86 | 72 | 16 | 65 | 44 | 31 | 89 |
| phikon-v2 | abmil | mixed | 80 | 75 | 22 | 71 | 51 | 37 | 86 |
| phikon-v2 | tileprobe | clean | 83 | 52 | 13 | 62 | 42 | 40 | 80 |
| phikon-v2 | tileprobe | mixed | 76 | 64 | 46 | 66 | 63 | 56 | 70 |

## Sensitivity by tumour size under clean

| encoder | head | training | 1-10 tiles (n=14) | 11-50 tiles (n=11) | 51-200 tiles (n=11) | >200 tiles (n=13) |
|---|---|---|---|---|---|---|
| dinov2-l | abmil | clean | 10 | 29 | 76 | 100 |
| dinov2-l | abmil | mixed | 14 | 18 | 65 | 100 |
| dinov2-l | tileprobe | clean | 10 | 36 | 75 | 100 |
| dinov2-l | tileprobe | mixed | 4 | 22 | 75 | 100 |
| kaiko-b16 | abmil | clean | 79 | 98 | 100 | 100 |
| kaiko-b16 | abmil | mixed | 79 | 93 | 100 | 100 |
| kaiko-b16 | tileprobe | clean | 49 | 76 | 95 | 100 |
| kaiko-b16 | tileprobe | mixed | 39 | 67 | 91 | 100 |
| phikon | abmil | clean | 93 | 93 | 100 | 100 |
| phikon | abmil | mixed | 89 | 91 | 100 | 100 |
| phikon | tileprobe | clean | 47 | 64 | 98 | 100 |
| phikon | tileprobe | mixed | 53 | 51 | 95 | 100 |
| phikon-v2 | abmil | clean | 67 | 82 | 98 | 100 |
| phikon-v2 | abmil | mixed | 57 | 73 | 93 | 100 |
| phikon-v2 | tileprobe | clean | 67 | 75 | 93 | 100 |
| phikon-v2 | tileprobe | mixed | 46 | 71 | 89 | 100 |

## Sensitivity by tumour size under jpeg:15

| encoder | head | training | 1-10 tiles (n=14) | 11-50 tiles (n=11) | 51-200 tiles (n=11) | >200 tiles (n=13) |
|---|---|---|---|---|---|---|
| dinov2-l | abmil | clean | 17 | 18 | 53 | 98 |
| dinov2-l | abmil | mixed | 14 | 22 | 64 | 100 |
| dinov2-l | tileprobe | clean | 7 | 18 | 67 | 98 |
| dinov2-l | tileprobe | mixed | 4 | 20 | 69 | 100 |
| kaiko-b16 | abmil | clean | 43 | 22 | 60 | 91 |
| kaiko-b16 | abmil | mixed | 54 | 53 | 87 | 100 |
| kaiko-b16 | tileprobe | clean | 56 | 42 | 73 | 92 |
| kaiko-b16 | tileprobe | mixed | 47 | 69 | 85 | 98 |
| phikon | abmil | clean | 23 | 20 | 60 | 94 |
| phikon | abmil | mixed | 47 | 56 | 89 | 100 |
| phikon | tileprobe | clean | 27 | 20 | 69 | 91 |
| phikon | tileprobe | mixed | 36 | 55 | 87 | 98 |
| phikon-v2 | abmil | clean | 26 | 20 | 47 | 82 |
| phikon-v2 | abmil | mixed | 30 | 29 | 60 | 85 |
| phikon-v2 | tileprobe | clean | 10 | 27 | 51 | 80 |
| phikon-v2 | tileprobe | mixed | 27 | 49 | 82 | 97 |

## Sensitivity by tumour size under jpeg:10

| encoder | head | training | 1-10 tiles (n=14) | 11-50 tiles (n=11) | 51-200 tiles (n=11) | >200 tiles (n=13) |
|---|---|---|---|---|---|---|
| dinov2-l | abmil | clean | 16 | 11 | 49 | 97 |
| dinov2-l | abmil | mixed | 14 | 24 | 58 | 100 |
| dinov2-l | tileprobe | clean | 3 | 11 | 40 | 88 |
| dinov2-l | tileprobe | mixed | 7 | 16 | 67 | 100 |
| kaiko-b16 | abmil | clean | 33 | 9 | 55 | 80 |
| kaiko-b16 | abmil | mixed | 34 | 27 | 69 | 95 |
| kaiko-b16 | tileprobe | clean | 26 | 22 | 51 | 94 |
| kaiko-b16 | tileprobe | mixed | 29 | 69 | 85 | 98 |
| phikon | abmil | clean | 14 | 7 | 45 | 80 |
| phikon | abmil | mixed | 41 | 36 | 65 | 94 |
| phikon | tileprobe | clean | 17 | 18 | 53 | 86 |
| phikon | tileprobe | mixed | 27 | 45 | 84 | 92 |
| phikon-v2 | abmil | clean | 11 | 9 | 35 | 68 |
| phikon-v2 | abmil | mixed | 16 | 11 | 51 | 71 |
| phikon-v2 | tileprobe | clean | 13 | 20 | 47 | 80 |
| phikon-v2 | tileprobe | mixed | 13 | 47 | 78 | 92 |

## Sensitivity by tumour size under blur:4

| encoder | head | training | 1-10 tiles (n=14) | 11-50 tiles (n=11) | 51-200 tiles (n=11) | >200 tiles (n=13) |
|---|---|---|---|---|---|---|
| dinov2-l | abmil | clean | 9 | 4 | 15 | 37 |
| dinov2-l | abmil | mixed | 17 | 9 | 33 | 63 |
| dinov2-l | tileprobe | clean | 3 | 4 | 4 | 29 |
| dinov2-l | tileprobe | mixed | 1 | 4 | 16 | 75 |
| kaiko-b16 | abmil | clean | 9 | 5 | 25 | 69 |
| kaiko-b16 | abmil | mixed | 23 | 2 | 38 | 57 |
| kaiko-b16 | tileprobe | clean | 10 | 2 | 4 | 25 |
| kaiko-b16 | tileprobe | mixed | 19 | 22 | 53 | 92 |
| phikon | abmil | clean | 9 | 5 | 25 | 45 |
| phikon | abmil | mixed | 20 | 11 | 33 | 42 |
| phikon | tileprobe | clean | 11 | 2 | 13 | 25 |
| phikon | tileprobe | mixed | 9 | 35 | 51 | 83 |
| phikon-v2 | abmil | clean | 1 | 2 | 16 | 45 |
| phikon-v2 | abmil | mixed | 13 | 0 | 25 | 48 |
| phikon-v2 | tileprobe | clean | 6 | 5 | 2 | 38 |
| phikon-v2 | tileprobe | mixed | 7 | 24 | 55 | 98 |

## Hospital

Test slides (RUMC / UMCU): normal 50 / 30; macro 13 / 9; micro 11 / 16. AUC within each centre compares tumour and normal slides of that centre only. The last column is how well the score tells the centres apart among normal slides; 50 means not at all. The last is the Spearman correlation, among normal slides, between the score and the number of tissue tiles: pooling over more tiles gives more chances for a high-scoring false positive.

### clean

| encoder | head | training | AUC within RUMC | AUC within UMCU | micro AUC within RUMC | micro AUC within UMCU | centre from score | score vs tissue (rank corr.) |
|---|---|---|---|---|---|---|---|---|
| dinov2-l | abmil | clean | 81.6 | 72.4 | 63.7 | 56.9 | 51.4 | +0.21 |
| dinov2-l | abmil | mixed | 78.5 | 74.2 | 59.5 | 60.7 | 58.8 | +0.29 |
| dinov2-l | tileprobe | clean | 84.9 | 71.9 | 69.3 | 56.1 | 54.2 | +0.47 |
| dinov2-l | tileprobe | mixed | 79.9 | 71.2 | 59.4 | 55.0 | 42.6 | +0.39 |
| kaiko-b16 | abmil | clean | 98.2 | 99.2 | 96.1 | 98.7 | 57.6 | +0.34 |
| kaiko-b16 | abmil | mixed | 98.6 | 99.4 | 97.1 | 99.0 | 47.7 | +0.29 |
| kaiko-b16 | tileprobe | clean | 91.6 | 91.1 | 82.7 | 86.0 | 44.8 | +0.33 |
| kaiko-b16 | tileprobe | mixed | 90.0 | 90.2 | 79.8 | 84.7 | 47.6 | +0.38 |
| phikon | abmil | clean | 99.0 | 99.8 | 97.8 | 99.8 | 82.3 | +0.41 |
| phikon | abmil | mixed | 98.7 | 99.7 | 97.2 | 99.5 | 62.6 | +0.35 |
| phikon | tileprobe | clean | 94.8 | 90.8 | 89.3 | 85.5 | 51.4 | +0.37 |
| phikon | tileprobe | mixed | 96.0 | 89.9 | 92.1 | 84.2 | 24.0 | +0.04 |
| phikon-v2 | abmil | clean | 96.0 | 98.0 | 92.0 | 96.9 | 46.6 | +0.20 |
| phikon-v2 | abmil | mixed | 95.6 | 94.9 | 93.0 | 92.0 | 32.5 | +0.08 |
| phikon-v2 | tileprobe | clean | 95.3 | 90.8 | 91.2 | 85.6 | 46.0 | +0.15 |
| phikon-v2 | tileprobe | mixed | 90.6 | 85.7 | 83.4 | 77.6 | 33.4 | +0.17 |

### jpeg:15

| encoder | head | training | AUC within RUMC | AUC within UMCU | micro AUC within RUMC | micro AUC within UMCU | centre from score | score vs tissue (rank corr.) |
|---|---|---|---|---|---|---|---|---|
| dinov2-l | abmil | clean | 79.0 | 70.8 | 62.4 | 55.5 | 24.5 | -0.02 |
| dinov2-l | abmil | mixed | 81.7 | 72.4 | 66.3 | 57.5 | 36.0 | +0.09 |
| dinov2-l | tileprobe | clean | 87.6 | 63.5 | 76.1 | 43.2 | 79.7 | +0.66 |
| dinov2-l | tileprobe | mixed | 83.4 | 68.1 | 67.3 | 50.2 | 59.0 | +0.60 |
| kaiko-b16 | abmil | clean | 81.9 | 78.8 | 67.3 | 72.2 | 17.5 | -0.09 |
| kaiko-b16 | abmil | mixed | 90.6 | 90.8 | 80.3 | 89.9 | 42.4 | +0.17 |
| kaiko-b16 | tileprobe | clean | 89.9 | 85.5 | 85.5 | 81.2 | 68.5 | +0.54 |
| kaiko-b16 | tileprobe | mixed | 91.1 | 86.4 | 84.6 | 78.7 | 78.5 | +0.52 |
| phikon | abmil | clean | 73.1 | 77.2 | 61.1 | 67.0 | 38.8 | -0.09 |
| phikon | abmil | mixed | 86.8 | 85.9 | 72.3 | 78.8 | 65.6 | +0.35 |
| phikon | tileprobe | clean | 84.8 | 74.8 | 75.4 | 64.5 | 15.6 | +0.09 |
| phikon | tileprobe | mixed | 91.4 | 83.4 | 85.0 | 74.1 | 62.5 | +0.48 |
| phikon-v2 | abmil | clean | 73.7 | 69.9 | 62.4 | 55.8 | 64.8 | +0.31 |
| phikon-v2 | abmil | mixed | 78.1 | 75.3 | 69.8 | 65.3 | 48.2 | +0.17 |
| phikon-v2 | tileprobe | clean | 83.4 | 71.3 | 75.2 | 58.7 | 66.6 | +0.46 |
| phikon-v2 | tileprobe | mixed | 88.2 | 81.7 | 79.4 | 71.4 | 64.3 | +0.57 |

### jpeg:10

| encoder | head | training | AUC within RUMC | AUC within UMCU | micro AUC within RUMC | micro AUC within UMCU | centre from score | score vs tissue (rank corr.) |
|---|---|---|---|---|---|---|---|---|
| dinov2-l | abmil | clean | 76.6 | 69.3 | 60.0 | 54.5 | 26.9 | -0.03 |
| dinov2-l | abmil | mixed | 80.1 | 72.2 | 64.8 | 57.0 | 37.3 | +0.15 |
| dinov2-l | tileprobe | clean | 85.3 | 64.8 | 74.6 | 46.9 | 84.3 | +0.67 |
| dinov2-l | tileprobe | mixed | 83.6 | 67.5 | 67.4 | 49.5 | 69.2 | +0.62 |
| kaiko-b16 | abmil | clean | 78.4 | 69.9 | 62.4 | 59.2 | 18.9 | -0.06 |
| kaiko-b16 | abmil | mixed | 86.1 | 78.7 | 74.0 | 70.2 | 47.1 | +0.22 |
| kaiko-b16 | tileprobe | clean | 87.9 | 73.4 | 85.0 | 62.5 | 74.5 | +0.56 |
| kaiko-b16 | tileprobe | mixed | 91.3 | 86.0 | 84.6 | 78.4 | 80.2 | +0.61 |
| phikon | abmil | clean | 68.6 | 64.2 | 55.4 | 50.2 | 22.4 | -0.26 |
| phikon | abmil | mixed | 82.3 | 78.2 | 66.5 | 69.0 | 55.6 | +0.27 |
| phikon | tileprobe | clean | 82.5 | 72.5 | 76.1 | 61.5 | 10.8 | +0.01 |
| phikon | tileprobe | mixed | 92.6 | 83.9 | 88.6 | 74.9 | 83.9 | +0.63 |
| phikon-v2 | abmil | clean | 68.7 | 65.5 | 55.3 | 50.3 | 64.8 | +0.27 |
| phikon-v2 | abmil | mixed | 70.1 | 66.5 | 60.4 | 52.2 | 54.6 | +0.20 |
| phikon-v2 | tileprobe | clean | 80.7 | 71.0 | 74.0 | 57.8 | 90.5 | +0.57 |
| phikon-v2 | tileprobe | mixed | 86.0 | 76.5 | 79.0 | 63.3 | 68.2 | +0.60 |

### blur:4

| encoder | head | training | AUC within RUMC | AUC within UMCU | micro AUC within RUMC | micro AUC within UMCU | centre from score | score vs tissue (rank corr.) |
|---|---|---|---|---|---|---|---|---|
| dinov2-l | abmil | clean | 48.9 | 62.4 | 42.3 | 45.0 | 17.8 | -0.19 |
| dinov2-l | abmil | mixed | 64.1 | 54.5 | 55.7 | 33.0 | 26.5 | -0.07 |
| dinov2-l | tileprobe | clean | 55.9 | 51.6 | 46.1 | 37.4 | 56.5 | +0.56 |
| dinov2-l | tileprobe | mixed | 62.5 | 55.1 | 40.8 | 34.0 | 48.9 | +0.41 |
| kaiko-b16 | abmil | clean | 60.7 | 51.5 | 40.3 | 32.0 | 28.5 | -0.09 |
| kaiko-b16 | abmil | mixed | 64.3 | 58.6 | 54.8 | 41.5 | 36.6 | +0.13 |
| kaiko-b16 | tileprobe | clean | 51.7 | 58.3 | 49.6 | 41.5 | 18.1 | -0.04 |
| kaiko-b16 | tileprobe | mixed | 79.4 | 62.3 | 65.1 | 43.1 | 56.3 | +0.41 |
| phikon | abmil | clean | 57.5 | 63.1 | 34.9 | 47.1 | 2.5 | -0.40 |
| phikon | abmil | mixed | 58.8 | 52.8 | 48.1 | 40.8 | 33.1 | -0.11 |
| phikon | tileprobe | clean | 66.6 | 70.1 | 59.7 | 62.5 | 13.0 | -0.00 |
| phikon | tileprobe | mixed | 79.9 | 59.6 | 67.5 | 41.1 | 65.4 | +0.41 |
| phikon-v2 | abmil | clean | 51.6 | 45.1 | 36.5 | 22.8 | 100.0 | +0.36 |
| phikon-v2 | abmil | mixed | 56.2 | 46.4 | 47.4 | 26.6 | 83.6 | +0.22 |
| phikon-v2 | tileprobe | clean | 56.2 | 49.3 | 40.9 | 34.3 | 95.9 | +0.49 |
| phikon-v2 | tileprobe | mixed | 75.2 | 63.4 | 52.1 | 43.1 | 65.6 | +0.39 |

## Shared failures (mixed training)

A slide counts as detected if it clears the 95%-specificity threshold in at least three of the five seeds.

| encoder | condition | caught by both | attention only | tile head only | missed by both | missed by both: micro |
|---|---|---|---|---|---|---|
| dinov2-l | clean | 20 | 4 | 3 | 22 | 21 |
| dinov2-l | jpeg:15 | 20 | 5 | 4 | 20 | 19 |
| dinov2-l | jpeg:10 | 20 | 3 | 4 | 22 | 20 |
| dinov2-l | blur:4 | 8 | 7 | 4 | 30 | 23 |
| kaiko-b16 | clean | 36 | 9 | 0 | 4 | 4 |
| kaiko-b16 | jpeg:15 | 32 | 4 | 5 | 8 | 8 |
| kaiko-b16 | jpeg:10 | 25 | 3 | 9 | 12 | 12 |
| kaiko-b16 | blur:4 | 12 | 4 | 10 | 23 | 19 |
| phikon | clean | 36 | 11 | 0 | 2 | 2 |
| phikon | jpeg:15 | 30 | 6 | 2 | 11 | 11 |
| phikon | jpeg:10 | 24 | 7 | 5 | 13 | 11 |
| phikon | blur:4 | 9 | 4 | 13 | 23 | 17 |
| phikon-v2 | clean | 36 | 3 | 1 | 9 | 8 |
| phikon-v2 | jpeg:15 | 21 | 3 | 9 | 16 | 14 |
| phikon-v2 | jpeg:10 | 15 | 2 | 12 | 20 | 17 |
| phikon-v2 | blur:4 | 8 | 2 | 15 | 24 | 22 |
