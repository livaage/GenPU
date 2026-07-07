"""Detector geometry constants for the ColliderML tracker.

The tracker has 48 unique (volume_id, layer_id) combinations, split into:
- Barrel layers (volumes 17, 24, 29): concentric cylinders at fixed r, variable z
- Endcap discs (volumes 16, 18, 23, 25, 28, 30): discs at fixed z, variable r

For the generative model, we assign each (volume, layer) pair a layer_class
index (0-47). The model predicts layer_class as a categorical, then predicts
per-layer-standardized residuals for (r, phi, z, time). This ensures the model
always predicts ~N(0,1) values regardless of whether a coordinate is "fixed"
(barrel r, endcap z) or "free" (barrel z, endcap r).
"""

import numpy as np

# Per-layer statistics for all 4 continuous features, in physical units (mm, rad, ns).
# Computed from 50k particles of shard 0.
# (volume_id, layer_id) -> (mean_r, std_r, mean_z, std_z, mean_phi, std_phi, mean_time, std_time, is_barrel)
LAYER_GEOMETRY = {
    # Endcap − (volumes 16, 23, 28): negative z
    (16,  4): ( 94.77,  36.93, -1518.80,    3.42, -0.0799, 1.7714,  34.18,  362.07, False),
    (16,  6): ( 94.88,  38.39, -1318.78,    3.46, -0.0140, 1.8020,  26.15,  309.26, False),
    (16,  8): ( 96.35,  37.39, -1118.91,    3.48, -0.0610, 1.7732,  23.28,  304.92, False),
    (16, 10): ( 95.38,  38.14,  -978.85,    3.46,  0.0069, 1.7937,  63.14,  750.40, False),
    (16, 12): ( 94.32,  37.20,  -838.78,    3.45, -0.0998, 1.7950,  47.46,  546.47, False),
    (16, 14): ( 95.49,  36.89,  -718.89,    3.47, -0.0810, 1.7729,  56.51,  639.91, False),
    (16, 16): ( 95.68,  37.59,  -618.77,    3.44, -0.0949, 1.7665,  44.92,  476.98, False),
    # Barrel inner (volume 17): pixel
    (17,  2): ( 32.55,   0.67,    13.46,  246.15,  0.0278, 1.7698,  94.52,  586.74, True),
    (17,  4): ( 68.36,   0.61,     4.46,  270.68, -0.0584, 1.7794, 113.61,  761.73, True),
    (17,  6): (114.31,   0.59,     2.77,  274.60,  0.0270, 1.7832,  93.22,  562.13, True),
    (17,  8): (170.27,   0.59,    35.52,  278.57, -0.0456, 1.8129, 169.77,  779.87, True),
    # Endcap + (volumes 18, 25, 30): positive z
    (18,  2): ( 94.77,  37.86,   618.83,    3.42,  0.0478, 1.8115,  74.19,  610.13, False),
    (18,  4): ( 93.06,  37.87,   718.69,    3.41,  0.1202, 1.8129,  45.87,  426.64, False),
    (18,  6): ( 93.60,  37.89,   838.71,    3.41,  0.0118, 1.8398,  43.90,  395.49, False),
    (18,  8): ( 94.36,  37.73,   978.68,    3.39,  0.1388, 1.8133,  44.74,  411.61, False),
    (18, 10): ( 94.01,  37.61,  1118.68,    3.40,  0.1188, 1.7961,  46.94,  434.80, False),
    (18, 12): ( 94.85,  37.56,  1318.71,    3.43,  0.0706, 1.8319,  63.97,  547.29, False),
    (18, 14): ( 93.61,  36.30,  1518.56,    3.40,  0.0684, 1.8154,  29.74,  329.19, False),
    # Endcap − mid (volume 23)
    (23,  2): (426.10, 128.43, -2948.17,    5.26, -0.0560, 1.8322,  88.46,  660.99, False),
    (23,  4): (425.45, 127.77, -2548.19,    5.38, -0.1014, 1.8052,  80.39,  522.02, False),
    (23,  6): (420.55, 130.56, -2197.96,    5.16, -0.0144, 1.8026,  75.61,  558.09, False),
    (23,  8): (423.60, 129.90, -1848.13,    5.31, -0.0837, 1.8078, 108.45,  853.66, False),
    (23, 10): (425.48, 130.05, -1548.14,    5.29, -0.0309, 1.8436,  76.43,  551.59, False),
    (23, 12): (424.21, 127.07, -1298.16,    5.39, -0.0648, 1.8345,  58.74,  415.51, False),
    # Barrel mid (volume 24): strip
    (24,  2): (260.37,   2.09,     1.64,  637.66, -0.0854, 1.8076,  82.75,  467.04, True),
    (24,  4): (360.20,   2.07,     1.72,  646.39,  0.0018, 1.8664,  97.80,  765.78, True),
    (24,  6): (500.13,   2.07,   -20.79,  658.76, -0.1273, 1.9506, 148.78,  719.03, True),
    (24,  8): (660.07,   2.09,   -34.67,  666.17, -0.0099, 1.8011, 118.20,  717.33, True),
    # Endcap + mid (volume 25)
    (25,  2): (432.93, 130.57,  1298.19,    5.21,  0.1523, 1.8116,  73.40,  512.49, False),
    (25,  4): (430.01, 131.13,  1548.09,    5.36,  0.1509, 1.8366,  98.96,  631.67, False),
    (25,  6): (418.16, 130.14,  1847.95,    5.09,  0.0244, 1.8274,  71.77,  500.47, False),
    (25,  8): (420.05, 132.21,  2197.86,    5.13,  0.1166, 1.7711,  72.03,  518.51, False),
    (25, 10): (415.69, 131.63,  2547.77,    5.08,  0.1084, 1.8031,  82.16,  530.20, False),
    (25, 12): (420.86, 127.91,  2948.26,    5.28,  0.0141, 1.7829,  71.41,  501.80, False),
    # Endcap − outer (volume 28)
    (28,  2): (890.12,  83.60, -2997.48,   16.60,  0.0602, 1.8816, 197.72, 1020.13, False),
    (28,  4): (892.47,  84.05, -2597.57,   16.67, -0.0242, 1.8952, 210.82, 1004.82, False),
    (28,  6): (895.97,  84.55, -2248.05,   16.63,  0.1310, 1.8627, 149.02,  803.99, False),
    (28,  8): (892.11,  84.03, -1897.86,   16.56,  0.0835, 1.8245, 219.96, 1020.34, False),
    (28, 10): (895.60,  84.48, -1597.67,   16.68,  0.0569, 1.7832, 117.39,  661.82, False),
    (28, 12): (893.50,  84.17, -1297.71,   16.48,  0.1066, 1.8005, 220.41, 1003.69, False),
    # Barrel outer (volume 29)
    (29,  2): (820.51,   5.35,    25.08,  688.43,  0.0415, 1.8784, 175.61,  803.90, True),
    (29,  4): (1020.46,  5.19,   -28.49,  677.32,  0.0153, 1.8577, 224.29, 1034.86, True),
    # Endcap + outer (volume 30)
    (30,  2): (897.86,  84.78,  1298.85,   16.88,  0.0785, 1.8057, 169.21,  839.75, False),
    (30,  4): (896.03,  84.61,  1598.19,   16.98,  0.0435, 1.8612, 121.61,  662.99, False),
    (30,  6): (899.10,  84.87,  1898.42,   16.79,  0.1121, 1.8170, 207.74,  902.70, False),
    (30,  8): (895.34,  84.47,  2248.01,   17.02,  0.0624, 1.8594, 191.86,  916.32, False),
    (30, 10): (895.32,  84.47,  2598.90,   17.13,  0.0649, 1.8450, 133.65,  656.57, False),
    (30, 12): (890.09,  83.68,  2997.26,   17.10,  0.1895, 1.8204, 119.79,  632.28, False),
}

# Ordered list for indexing: layer_class 0..47
LAYER_LIST = sorted(LAYER_GEOMETRY.keys())
N_LAYERS = len(LAYER_LIST)

# Lookup tables as numpy arrays
LAYER_TO_CLASS = {vl: i for i, vl in enumerate(LAYER_LIST)}
CLASS_TO_LAYER = {i: vl for i, vl in enumerate(LAYER_LIST)}

# Per-layer arrays for vectorized operations: shape (48,)
# Order: (mean_r, std_r, mean_z, std_z, mean_phi, std_phi, mean_time, std_time, is_barrel)
LAYER_MEAN_R = np.array([LAYER_GEOMETRY[vl][0] for vl in LAYER_LIST], dtype=np.float32)
LAYER_STD_R = np.array([LAYER_GEOMETRY[vl][1] for vl in LAYER_LIST], dtype=np.float32)
LAYER_MEAN_Z = np.array([LAYER_GEOMETRY[vl][2] for vl in LAYER_LIST], dtype=np.float32)
LAYER_STD_Z = np.array([LAYER_GEOMETRY[vl][3] for vl in LAYER_LIST], dtype=np.float32)
LAYER_MEAN_PHI = np.array([LAYER_GEOMETRY[vl][4] for vl in LAYER_LIST], dtype=np.float32)
LAYER_STD_PHI = np.array([LAYER_GEOMETRY[vl][5] for vl in LAYER_LIST], dtype=np.float32)
LAYER_MEAN_TIME = np.array([LAYER_GEOMETRY[vl][6] for vl in LAYER_LIST], dtype=np.float32)
LAYER_STD_TIME = np.array([LAYER_GEOMETRY[vl][7] for vl in LAYER_LIST], dtype=np.float32)
LAYER_IS_BARREL = np.array([LAYER_GEOMETRY[vl][8] for vl in LAYER_LIST], dtype=bool)

# Combined arrays for convenient per-layer standardization: shape (48, 4)
# Feature order: (r, phi, z, time) — matches tracker_ar continuous output
LAYER_MEANS = np.stack([LAYER_MEAN_R, LAYER_MEAN_PHI, LAYER_MEAN_Z, LAYER_MEAN_TIME], axis=1)
LAYER_STDS = np.stack([LAYER_STD_R, LAYER_STD_PHI, LAYER_STD_Z, LAYER_STD_TIME], axis=1)

# Volume and layer IDs for reconstruction from class index
LAYER_VOLUME_IDS = np.array([vl[0] for vl in LAYER_LIST], dtype=np.float32)
LAYER_LAYER_IDS = np.array([vl[1] for vl in LAYER_LIST], dtype=np.float32)


def hits_to_layer_class(volume_ids: np.ndarray, layer_ids: np.ndarray) -> np.ndarray:
    """Convert (volume_id, layer_id) arrays to layer_class indices."""
    result = np.zeros(len(volume_ids), dtype=np.int32)
    for i, (v, l) in enumerate(zip(volume_ids.astype(int), layer_ids.astype(int))):
        result[i] = LAYER_TO_CLASS.get((v, l), 0)
    return result
