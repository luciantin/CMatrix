import numpy as np

y_pred = np.array([
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5]]
    )
y_true = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])

samples = len(y_pred)
y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

print(y_pred_clipped)

correct_confidences = 0

# Probabilities for target values -
# only if categorical labels
print(range(samples))
if len(y_true.shape) == 1:
    correct_confidences = y_pred_clipped[
        range(samples),
        y_true
    ]

# Mask values - only for one-hot encoded labels
elif len(y_true.shape) == 2:
    correct_confidences = np.sum(
        y_pred_clipped * y_true,
        axis=1
    )


print(correct_confidences)