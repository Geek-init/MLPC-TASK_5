# Classifier tuning summary (development validation)

- Configs evaluated: 36 (model x params x threshold)
- Selection metric: segment-based Macro F1 on the development validation split
- **Best: logistic (C=1.0,class_weight=None), threshold=0.3 -> dev Macro F1 0.4316**
- Best on non-hidden test (evaluated once): Macro F1 0.4360

## Top configurations

| model | params | threshold | dev Macro F1 |
|---|---|---|---|
| logistic | C=1.0,class_weight=None | 0.3 | 0.4316 |
| logistic | C=1.0,class_weight=None | 0.2 | 0.4312 |
| logistic | C=1.0,class_weight=None | 0.4 | 0.4223 |
| ridge | alpha=10.0 | 0.5 | 0.4098 |
| logistic | C=1.0,class_weight=None | 0.5 | 0.4090 |
| ridge | alpha=0.5 | 0.5 | 0.4048 |
| ridge | alpha=1.0 | 0.5 | 0.4017 |
| logistic | C=1.0,class_weight=balanced | 0.5 | 0.3955 |
| ridge | alpha=5.0 | 0.5 | 0.3936 |
| random_forest | n_estimators=100,max_depth=None | 0.3 | 0.3914 |
| ridge | alpha=0.1 | 0.5 | 0.3910 |
| random_forest | n_estimators=100,max_depth=20 | 0.3 | 0.3851 |

## Per-model best (dev Macro F1)

| model | best dev Macro F1 |
|---|---|
| logistic | 0.4316 |
| ridge | 0.4098 |
| random_forest | 0.3914 |