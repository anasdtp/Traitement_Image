#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 2 16:42:36 2022

@author: gouiffes
"""

# librairies
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import motion

BASE_DIR = Path(__file__).resolve().parents[1]
VIDEO_PATH = BASE_DIR / "im" / "Video" / "stmarc_video.avi"
OUT_DIR = BASE_DIR / "Rapports" / "Capture_ecran_TP3_Mouvement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHOW_WINDOWS = True
USE_PHOTOMETRIC_NORMALIZATION = True
THRESHOLD = 20
ALPHA_BG = 0.2
MIN_AREA = 10
MAX_FRAMES = 260

# Frames a exporter pour le rapport
SNAPSHOT_FRAMES = {10, 20, 30,60, 120}

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Impossible d'ouvrir la video: {VIDEO_PATH}")

# ouverture du flux video
ret, frame = cap.read()
if not ret or frame is None:
    raise RuntimeError("Lecture de la premiere frame impossible.")
# pour l'affichage du flot optique
hsv = np.zeros_like(frame)
hsv[..., 1] = 255

# conversion en niveaux de gris
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )

if USE_PHOTOMETRIC_NORMALIZATION:
    frame = motion.photometric_normalization(frame)

my_h, my_w = frame.shape[0], frame.shape[1]
scale_percent =100
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
im = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

background = np.asanyarray(im, np.uint8)
foreground = np.asanyarray(im, np.uint8)
regions = np.zeros(dim, np.uint16)
prev_gray = np.asanyarray(im, np.uint8)

counts_raw=[]
counts_filtered=[]
mean_flow_per_frame=[]
k=0

def compute_optical_flow(prev_im, curr_im, mask, regions_map, stats, min_area=120):
    flow = cv2.calcOpticalFlowFarneback(
        prev_im,
        curr_im,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    # Visualisation HSV du flot.
    hsv = np.zeros((curr_im.shape[0], curr_im.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    flow_means=[]
    for label in range(1, len(stats)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp_mask = np.logical_and(regions_map == label, mask > 0)
        if np.any(comp_mask):
            flow_means.append(float(np.mean(mag[comp_mask])))

    avg_flow = float(np.mean(flow_means)) if flow_means else 0.0
    return flow_bgr, flow_means, avg_flow


# boucle de lecture frame par frame
while ret:

    if k >= MAX_FRAMES:
        break

	# stocker l'image issue de la vidéo à l'instant t dans la variable "frame"
    ret, frame = cap.read() 
    if not ret or frame is None:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )

    if USE_PHOTOMETRIC_NORMALIZATION:
        frame = motion.photometric_normalization(frame)

    # redimensionnement image
    im = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    """ 1- Detection des pixels en mouvement """
    foreground = motion.background_subtract(background, im, THRESHOLD)

    """2- mise a jour du fond """
    background = motion.update_beackground_mean(background, im, ALPHA_BG)

    """3 morphologie mathematique"""
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg_morph = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel_open)
    fg_morph = cv2.morphologyEx(fg_morph, cv2.MORPH_CLOSE, kernel_close)

    """ 4 etiquetage en composantes connexes """
    retcc_raw, _, stats_raw, _ = cv2.connectedComponentsWithStats(np.uint8(foreground))
    retcc, regions, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(fg_morph))

    nb_raw = max(0, retcc_raw - 1)
    nb_filtered = 0
    annotated = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for label in range(1, retcc):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_AREA:
            continue
        nb_filtered += 1
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    counts_raw.append(nb_raw)
    counts_filtered.append(nb_filtered)

    """ 5 Flot Optique """
    flow_vis, mean_flow_components, avg_flow = compute_optical_flow(
        prev_gray,
        im,
        fg_morph,
        regions,
        stats,
        min_area=MIN_AREA,
    )
    mean_flow_per_frame.append(avg_flow)
    prev_gray = im.copy()

    if k in SNAPSHOT_FRAMES:
        cv2.imwrite(str(OUT_DIR / f"frame_{k:04d}_gray.png"), im)
        cv2.imwrite(str(OUT_DIR / f"frame_{k:04d}_foreground_raw.png"), foreground)
        cv2.imwrite(str(OUT_DIR / f"frame_{k:04d}_foreground_morph.png"), fg_morph)
        cv2.imwrite(str(OUT_DIR / f"frame_{k:04d}_flow.png"), flow_vis)
        cv2.imwrite(str(OUT_DIR / f"frame_{k:04d}_annotated.png"), annotated)

    if SHOW_WINDOWS:
        cv2.imshow('image', np.uint8(im))
        cv2.imshow('foreground', np.uint8(foreground))
        cv2.imshow('foreground_morph', np.uint8(fg_morph))
        cv2.imshow('fond', np.uint8(background))
        cv2.imshow('flow', flow_vis)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    if (k % 30) == 0:
        print(
            f"frame={k:04d} raw={nb_raw} filtered={nb_filtered} avg_flow={avg_flow:.3f}"
        )
    k += 1

# Graphes de synthese pour le rapport
if len(counts_raw) > 0:
    x = np.arange(len(counts_raw))
    plt.figure(figsize=(10, 4))
    plt.plot(x, counts_raw, label="objets sans morphologie", linewidth=1.2)
    plt.plot(x, counts_filtered, label="objets avec morphologie", linewidth=1.2)
    plt.xlabel("Frame")
    plt.ylabel("Nombre d'objets")
    plt.title("Comptage d'objets : impact de la morphologie")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "tp3_counts_comparison.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(x, mean_flow_per_frame, color="tab:orange")
    plt.xlabel("Frame")
    plt.ylabel("Flot optique moyen")
    plt.title("Evolution du flot optique moyen")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "tp3_mean_optical_flow.png"), dpi=180)
    plt.close()

    np.savetxt(
        str(OUT_DIR / "tp3_summary.csv"),
        np.column_stack([x, counts_raw, counts_filtered, mean_flow_per_frame]),
        delimiter=",",
        header="frame,count_raw,count_filtered,mean_flow",
        comments="",
        fmt=["%d", "%d", "%d", "%.6f"],
    )

print(f"Sorties TP3 generees dans: {OUT_DIR}")
 
#quiter le programme et fermer toutes les fenêtres ouvertes
cap.release()
cv2.destroyAllWindows()
