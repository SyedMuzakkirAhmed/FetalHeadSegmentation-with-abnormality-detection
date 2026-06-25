import os
import cv2
import math
import pandas as pd
import numpy as np

# ====================================
# SETTINGS — 
# ====================================

MASK_FOLDER = "U:/project/tensorf/src/Masks"
CSV_FILE = "U:/project/tensorf/src/pix.csv"
OUTPUT_CSV = "U:/project/tensorf/src/predicted_measurements.csv"
OUTPUT_VIS = "U:/project/tensorf/src/measurement_visuals"
os.makedirs(OUTPUT_VIS, exist_ok=True)

# model input size
MODEL_W = 512
MODEL_H = 512

# original size
ORIG_W = 800
ORIG_H = 540

PADDED_SIZE = 800   # max(original width, height)

SCALE_X = PADDED_SIZE / MODEL_W
SCALE_Y = PADDED_SIZE / MODEL_H

# ====================================
# LOAD CSV
# ====================================

df = pd.read_csv(CSV_FILE)

pixel_map = dict(zip(df["filename"], df["pixel size(mm)"]))
hc_gt_map = dict(zip(df["filename"], df["head circumference (mm)"]))

results = []

# ====================================
# PROCESS MASKS
# ====================================

for file in os.listdir(MASK_FOLDER):

    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(MASK_FOLDER, file)

    img = cv2.imread(path, 0)
    if img is None:
        continue

    # ==========================
    # BINARIZE
    # ==========================
    _, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    # ==========================
    # ***** CRITICAL FIX *****
    # FILL MASK INTERIOR
    # ==========================
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("No contour:", file)
        continue

    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)

    binary = filled

    # ==========================
    # FIND CONTOUR AGAIN
    # ==========================
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cnt = max(contours, key=cv2.contourArea)

    if len(cnt) < 5:
        print("Not enough points:", file)
        continue

    # ==========================
    # FIT ELLIPSE
    # ==========================

    ellipse = cv2.fitEllipse(cnt)
    (cx, cy), (major, minor), angle = ellipse
    # ==========================
    # DRAW VISUALIZATION
    # ==========================

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # draw fitted ellipse
    cv2.ellipse(vis, ellipse, (0,255,0), 2)

    # center
    center = (int(cx), int(cy))
    cv2.circle(vis, center, 3, (0,0,255), -1)

    # convert angle to radians
    theta = np.deg2rad(angle)

    # major axis (OFD)
    dx_major = int((major/2) * np.cos(theta))
    dy_major = int((major/2) * np.sin(theta))

    p1_major = (int(cx - dx_major), int(cy - dy_major))
    p2_major = (int(cx + dx_major), int(cy + dy_major))

    cv2.line(vis, p1_major, p2_major, (255,0,0), 2)

    # minor axis (BPD)
    theta_perp = theta + np.pi/2

    dx_minor = int((minor/2) * np.cos(theta_perp))
    dy_minor = int((minor/2) * np.sin(theta_perp))

    p1_minor = (int(cx - dx_minor), int(cy - dy_minor))
    p2_minor = (int(cx + dx_minor), int(cy + dy_minor))

    cv2.line(vis, p1_minor, p2_minor, (0,255,255), 2)

    if minor > major:
        major, minor = minor, major

    # ==========================
    # RESCALE TO ORIGINAL SIZE
    # ==========================

    major = major * SCALE_X
    minor = minor * SCALE_Y

    a = major / 2
    b = minor / 2

    # Ramanujan approximation
    HC_pixels = math.pi * (3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))

    BPD_pixels = minor
    OFD_pixels = major

    area_pixels = cv2.contourArea(cnt) * SCALE_X * SCALE_Y

    # ==========================
    # PIXEL SIZE
    # ==========================

    if file not in pixel_map:
        print("Pixel size missing:", file)
        continue

    pixel_size = pixel_map[file]

    HC_mm = HC_pixels * pixel_size
    BPD_mm = BPD_pixels * pixel_size
    OFD_mm = OFD_pixels * pixel_size
    Area_mm2 = area_pixels * (pixel_size ** 2)
    GA_weeks = (HC_mm / 10) + 5

    cv2.putText(vis, f"HC: {HC_mm:.2f} mm", (10,25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(vis, f"BPD: {BPD_mm:.2f} mm", (10,50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.putText(vis, f"OFD: {OFD_mm:.2f} mm", (10,75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    
    cv2.putText(vis, f"GA: {GA_weeks:.2f} weeks", (10,100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ==========================
    # HC COMPARISON
    # ==========================

    HC_gt = hc_gt_map.get(file, np.nan)

    if not np.isnan(HC_gt):
        HC_error = HC_mm - HC_gt
        HC_abs_error = abs(HC_error)
        HC_percent_error = (HC_abs_error / HC_gt) * 100
    else:
        HC_error = np.nan
        HC_abs_error = np.nan
        HC_percent_error = np.nan

    results.append({
        "filename": file,
        "HC_pred_mm": HC_mm,
        "HC_csv_mm": HC_gt,
        "HC_error_mm": HC_error,
        "HC_abs_error_mm": HC_abs_error,
        "HC_error_percent": HC_percent_error,
        "BPD_mm": BPD_mm,
        "OFD_mm": OFD_mm,
        "HeadArea_mm2": Area_mm2,
        "GA_weeks": GA_weeks
    })
    save_path = os.path.join(OUTPUT_VIS, file)
    cv2.imwrite(save_path, vis)
    print("Processed:", file)

# ====================================
# SAVE RESULTS
# ====================================

df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)

print("\nDONE — Saved to:")
print(OUTPUT_CSV)

# ====================================
# QUICK METRICS
# ====================================

print("\n===== SUMMARY =====")
print("Mean HC Absolute Error (mm):", df_out["HC_abs_error_mm"].mean())
print("Mean HC % Error:", df_out["HC_error_percent"].mean())