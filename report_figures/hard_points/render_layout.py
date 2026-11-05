#!/usr/bin/env python3
"""Crate seen from above, in true colour, from a saved whole-crate cloud; for scans that have no camera image.
A rendering of the 3D points, not a photo: anything the camera could not see stays dark.

  render_layout.py SCAN.pcd [OUT.png]     (default: SCAN_topdown.png)"""
import os, sys
import cv2
import numpy as np
X0, X1, Y0, Y1 = 0.335, 0.585, -0.08, 0.28  # crate interior in samla_crate.scene, as in the points files
MARGIN = 0.03; MM_PER_PX = 1.0; SCALE = 1

def main():
    path = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(path)[0]+'_topdown.png'
    c = np.loadtxt(path, skiprows=11)
    c = c[(c[:, 0] > X0-MARGIN) & (c[:, 0] < X1+MARGIN) & (c[:, 1] > Y0-MARGIN) & (c[:, 1] < Y1+MARGIN) & (c[:, 2] < .2)]
    c = c[np.argsort(c[:, 2])]  # lowest first, so the highest point of a pixel wins
    res = MM_PER_PX/1000; h = int((X1-X0+2*MARGIN)/res)+1; w = int((Y1-Y0+2*MARGIN)/res)+1
    # Image up = +x, image right = +y, the same orientation as points_hard_map.png
    r = (h-1-((c[:, 0]-X0+MARGIN)/res)).astype(int); s = ((c[:, 1]-Y0+MARGIN)/res).astype(int)
    rgb = c[:, 3].astype(np.uint32); img = np.full((h, w, 3), 24, np.uint8)
    img[r, s, 0] = rgb & 255; img[r, s, 1] = (rgb >> 8) & 255; img[r, s, 2] = (rgb >> 16) & 255
    # Fill the small holes between points, leave real gaps dark
    mask = (img == 24).all(2).astype(np.uint8); fill = cv2.inpaint(img, mask, 2, cv2.INPAINT_NS)
    near = cv2.dilate(1-mask, np.ones((5, 5), np.uint8)).astype(bool); img[near] = fill[near]
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST); k = int(2/res*1)
    a = (int((MARGIN)/res)*2, int((MARGIN)/res)*2); b = (int((Y1-Y0+MARGIN)/res)*2, int((X1-X0+MARGIN)/res)*2)
    cv2.rectangle(img, a, b, (255, 255, 255), 2)
    cv2.imwrite(out, img); print(out, img.shape[1], 'x', img.shape[0], 'px;', len(c), 'points')

if __name__ == '__main__': main()
