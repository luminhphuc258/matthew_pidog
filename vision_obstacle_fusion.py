#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vision_obstacle_fusion.py

Class độc lập để:
- detect gò/gờ trên sàn bằng OpenCV (FloorBumpDetector)
- detect chân bàn/cột mảnh (VerticalLegDetector)
- detect vật cản bằng YOLO (Ultralytics) + SAHI slice inference ✅ NEW
- fuse kết quả => quyết định obstacle trước mặt + loại vật cản
- export boxes để file khác vẽ lên camera frame

NOTE (SAHI):
- Cài SAHI:   pip3 install -U sahi
- Nếu SAHI không có, code tự fallback về YOLO thường (không crash).

Usage:

from vision_obstacle_fusion import VisionObstacleFusion

fusion = VisionObstacleFusion(
    enable_yolo=True,
    yolo_model="/home/matthewlupi/matthew_pidog/yolov8n.pt",
    device="cpu",
    yolo_use_sahi=True,
    sahi_slice_w=320,
    sahi_slice_h=320,
    sahi_overlap=0.22,
    yolo_roi_mode="FLOOR",  # chỉ detect vùng sàn để bắt vật nhỏ + nhẹ hơn
)

dec = fusion.update(frame_bgr)
out = fusion.draw_overlay(frame_bgr)
payload = fusion.get_boxes_payload()
"""

import time
import math
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np


# =========================
# Data structures
# =========================
@dataclass
class BumpHit:
    ok: bool
    score: float
    band_bbox: Optional[Tuple[int, int, int, int]]  # (x1,y1,x2,y2) full-frame bbox
    reason: str


@dataclass
class LegHit:
    ok: bool
    score: float
    bbox: Optional[Tuple[int, int, int, int]]       # (x1,y1,x2,y2) full-frame bbox
    reason: str


@dataclass
class FusionDecision:
    has_obstacle: bool
    obstacle_type: str
    confidence: float
    zone: str
    source: str
    detail: Dict[str, Any]


# =========================
# 1) Floor bump detector (OpenCV)
# =========================
class FloorBumpDetector:
    """
    Detect gò/gờ trên sàn bằng cách tìm dải line ngang trong ROI dưới (Canny + HoughLinesP).
    """
    def __init__(
        self,
        roi_y1_ratio: float = 0.58,
        roi_y2_ratio: float = 0.98,
        min_line_len_ratio: float = 0.45,
        max_line_gap: int = 18,
        hough_thresh: int = 45,
        angle_tol_deg: float = 12.0,
        band_merge_px: int = 14,
        band_min_thickness_px: int = 12,
        band_min_width_ratio: float = 0.65,
        score_trigger: float = 1.0,
        stable_frames: int = 3,
        decay: float = 0.85,
    ):
        self.ry1 = float(roi_y1_ratio)
        self.ry2 = float(roi_y2_ratio)
        self.min_line_len_ratio = float(min_line_len_ratio)
        self.max_line_gap = int(max_line_gap)
        self.hough_thresh = int(hough_thresh)
        self.angle_tol_deg = float(angle_tol_deg)
        self.band_merge_px = int(band_merge_px)
        self.band_min_thickness_px = int(band_min_thickness_px)
        self.band_min_width_ratio = float(band_min_width_ratio)
        self.score_trigger = float(score_trigger)
        self.stable_frames = int(stable_frames)
        self.decay = float(decay)

        self._ema_score = 0.0
        self._streak = 0

    def _roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        H, W = frame.shape[:2]
        y1 = int(H * self.ry1)
        y2 = int(H * self.ry2)
        y1 = max(0, min(H - 2, y1))
        y2 = max(y1 + 2, min(H, y2))
        return frame[y1:y2, :], y1, y2

    def detect(self, frame_bgr: np.ndarray) -> BumpHit:
        if frame_bgr is None or frame_bgr.size < 10:
            return BumpHit(False, 0.0, None, "no_frame")

        roi, y1, y2 = self._roi(frame_bgr)
        h, w = roi.shape[:2]
        if h < 40 or w < 80:
            return BumpHit(False, 0.0, None, "roi_small")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 140)

        min_len = int(self.min_line_len_ratio * w)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180.0,
            threshold=self.hough_thresh,
            minLineLength=max(30, min_len),
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            self._ema_score *= self.decay
            self._streak = max(0, self._streak - 1)
            return BumpHit(False, float(self._ema_score), None, "no_lines")

        horiz = []
        for l in lines:
            x1l, y1l, x2l, y2l = map(int, l[0])
            dx = x2l - x1l
            dy = y2l - y1l
            ang = abs(math.degrees(math.atan2(dy, dx + 1e-6)))
            if ang <= self.angle_tol_deg:
                horiz.append((x1l, y1l, x2l, y2l))

        if not horiz:
            self._ema_score *= self.decay
            self._streak = max(0, self._streak - 1)
            return BumpHit(False, float(self._ema_score), None, "no_horizontal")

        horiz.sort(key=lambda t: (t[1] + t[3]) // 2)
        bands = []  # [ymin,ymax,xmin,xmax,count]
        for (x1l, y1l, x2l, y2l) in horiz:
            yc = (y1l + y2l) // 2
            xmin = min(x1l, x2l)
            xmax = max(x1l, x2l)
            placed = False
            for b in bands:
                if abs(yc - ((b[0] + b[1]) // 2)) <= self.band_merge_px:
                    b[0] = min(b[0], yc)
                    b[1] = max(b[1], yc)
                    b[2] = min(b[2], xmin)
                    b[3] = max(b[3], xmax)
                    b[4] += 1
                    placed = True
                    break
            if not placed:
                bands.append([yc, yc, xmin, xmax, 1])

        best = None
        for (by0, by1, bx0, bx1, cnt) in bands:
            thickness = (by1 - by0) + 1
            width = (bx1 - bx0) + 1
            wr = width / float(w)

            if wr < self.band_min_width_ratio:
                continue
            if thickness < self.band_min_thickness_px:
                continue

            score = (0.6 * min(2.0, thickness / 18.0)) + (0.3 * min(2.0, cnt / 6.0)) + (0.4 * wr)
            if best is None or score > best[0]:
                best = (score, (bx0, by0, bx1, by1), cnt, thickness, wr)

        if best is None:
            self._ema_score *= self.decay
            self._streak = max(0, self._streak - 1)
            return BumpHit(False, float(self._ema_score), None, "bands_not_match")

        score, (bx0, by0, bx1, by1), cnt, thickness, wr = best

        self._ema_score = self._ema_score * self.decay + score * (1.0 - self.decay)
        if score >= self.score_trigger:
            self._streak += 1
        else:
            self._streak = max(0, self._streak - 1)

        hit_ok = (self._streak >= self.stable_frames) and (self._ema_score >= self.score_trigger * 0.85)

        x1f = int(bx0); x2f = int(bx1)
        y1f = int(y1 + by0); y2f = int(y1 + by1)
        reason = f"band(thick={thickness}px cnt={cnt} wr={wr:.2f}) ema={self._ema_score:.2f} streak={self._streak}"
        return BumpHit(bool(hit_ok), float(self._ema_score), (x1f, y1f, x2f, y2f), reason)


# =========================
# 1.5) Vertical leg / pole detector (OpenCV)
# =========================
class VerticalLegDetector:
    """
    Bắt "chân bàn/cột mảnh" (vật thể dọc, hẹp, cao) bằng gradient-x + morphology + contour filter.
    """
    def __init__(
        self,
        roi_y1_ratio: float = 0.25,
        roi_y2_ratio: float = 0.98,
        min_height_ratio: float = 0.28,
        max_width_ratio: float = 0.22,
        min_area_ratio: float = 0.003,
        aspect_min: float = 2.8,
        bottom_touch_ratio: float = 0.16,
        score_trigger: float = 0.55,
        stable_frames: int = 2,
        decay: float = 0.80,
    ):
        self.ry1 = float(roi_y1_ratio)
        self.ry2 = float(roi_y2_ratio)
        self.min_h = float(min_height_ratio)
        self.max_w = float(max_width_ratio)
        self.min_area = float(min_area_ratio)
        self.aspect_min = float(aspect_min)
        self.bottom_touch_ratio = float(bottom_touch_ratio)

        self.score_trigger = float(score_trigger)
        self.stable_frames = int(stable_frames)
        self.decay = float(decay)

        self._ema = 0.0
        self._streak = 0
        self._last_bbox = None
        self._last_reason = "init"

    def _roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        H, W = frame.shape[:2]
        y1 = int(H * self.ry1)
        y2 = int(H * self.ry2)
        y1 = max(0, min(H - 2, y1))
        y2 = max(y1 + 2, min(H, y2))
        return frame[y1:y2, :], y1, y2

    def detect(self, frame_bgr: np.ndarray) -> LegHit:
        if frame_bgr is None or frame_bgr.size < 10:
            return LegHit(False, 0.0, None, "no_frame")

        roi, y1f, y2f = self._roi(frame_bgr)
        rh, rw = roi.shape[:2]
        if rh < 60 or rw < 120:
            return LegHit(False, float(self._ema), None, "roi_small")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        agx = cv2.convertScaleAbs(gx)

        agx = cv2.GaussianBlur(agx, (5, 5), 0)
        _, bw = cv2.threshold(agx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        vk = max(11, int(rh * 0.10))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vk))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_v, iterations=1)
        bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)), iterations=1)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._ema *= self.decay
            self._streak = max(0, self._streak - 1)
            self._last_bbox = None
            self._last_reason = "no_contours"
            return LegHit(False, float(self._ema), None, self._last_reason)

        best = None
        roi_bottom = rh
        bottom_margin = int(self.bottom_touch_ratio * rh)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue

            hr = h / float(rh)
            wr = w / float(rw)
            ar = (w * h) / float(rw * rh + 1e-6)
            aspect = h / float(w + 1e-6)

            if hr < self.min_h:
                continue
            if wr > self.max_w:
                continue
            if ar < self.min_area:
                continue
            if aspect < self.aspect_min:
                continue
            if (y + h) < (roi_bottom - bottom_margin):
                continue

            patch = bw[y:y+h, x:x+w]
            ed = float(np.mean(patch > 0)) if patch.size else 0.0

            score = 0.55 * min(1.0, hr / 0.55) + 0.25 * min(1.0, ed / 0.55) + 0.20 * (1.0 - min(1.0, wr / 0.25))
            score = float(max(0.0, min(0.99, score)))

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "bbox_roi": (x, y, x + w, y + h),
                    "hr": hr, "wr": wr, "ar": ar, "aspect": aspect, "ed": ed
                }

        if best is None:
            self._ema *= self.decay
            self._streak = max(0, self._streak - 1)
            self._last_bbox = None
            self._last_reason = "no_candidate"
            return LegHit(False, float(self._ema), None, self._last_reason)

        score = float(best["score"])
        self._ema = self._ema * self.decay + score * (1.0 - self.decay)

        if score >= self.score_trigger:
            self._streak += 1
        else:
            self._streak = max(0, self._streak - 1)

        ok = (self._streak >= self.stable_frames) and (self._ema >= self.score_trigger * 0.85)

        x1, y1, x2, y2 = best["bbox_roi"]
        bbox_full = (int(x1), int(y1f + y1), int(x2), int(y1f + y2))

        self._last_bbox = bbox_full
        self._last_reason = f"leg(hr={best['hr']:.2f} wr={best['wr']:.2f} asp={best['aspect']:.1f} ed={best['ed']:.2f}) ema={self._ema:.2f} streak={self._streak}"
        return LegHit(bool(ok), float(self._ema), bbox_full, self._last_reason)


# =========================
# 2) YOLO detector (Ultralytics) + SAHI slice inference ✅
# =========================
class YoloDetector:
    """
    - YOLO thường: ultralytics.YOLO().predict
    - SAHI: get_sliced_prediction (cắt ảnh thành nhiều miếng nhỏ -> bắt small objects tốt hơn)

    Tip hiệu quả nhất cho vật nhỏ trên sàn:
    - yolo_roi_mode="FLOOR" (chỉ detect vùng dưới ảnh)
    - SAHI slice 320/384 + overlap ~0.2
    """
    def __init__(
        self,
        model_path: str,
        img_size: int = 416,
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "cpu",
        max_det: int = 60,
        every_n_frames: int = 2,

        # SAHI
        use_sahi: bool = True,
        sahi_slice_w: int = 320,
        sahi_slice_h: int = 320,
        sahi_overlap: float = 0.22,
        sahi_postprocess: str = "NMS",

        # ROI mode để nhẹ + tăng khả năng bắt vật nhỏ
        # "FULL" | "FLOOR"
        roi_mode: str = "FLOOR",
        roi_y1_ratio: float = 0.42,
        roi_y2_ratio: float = 0.98,
    ):
        self.model_path = str(model_path)
        self.img_size = int(img_size)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = str(device)
        self.max_det = int(max_det)
        self.every_n_frames = max(1, int(every_n_frames))

        self.use_sahi = bool(use_sahi)
        self.sahi_slice_w = int(sahi_slice_w)
        self.sahi_slice_h = int(sahi_slice_h)
        self.sahi_overlap = float(sahi_overlap)
        self.sahi_postprocess = str(sahi_postprocess or "NMS")

        self.roi_mode = str(roi_mode or "FULL").upper()
        self.roi_y1_ratio = float(roi_y1_ratio)
        self.roi_y2_ratio = float(roi_y2_ratio)

        self._frame_i = 0
        self._last: List[Dict[str, Any]] = []
        self._last_ts = 0.0

        # load YOLO
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)

        # SAHI init (optional)
        self._sahi_ok = False
        self._sahi_err = ""
        self._sahi_model = None
        self._sahi_get_sliced_prediction = None

        if self.use_sahi:
            self._init_sahi()

    def _init_sahi(self):
        """
        Robust SAHI init:
        - try AutoDetectionModel.from_pretrained(model_type="yolov8")
        - fallback to Yolov8DetectionModel
        """
        try:
            from sahi.predict import get_sliced_prediction
            self._sahi_get_sliced_prediction = get_sliced_prediction
        except Exception as e:
            self._sahi_ok = False
            self._sahi_err = f"import_sahi_predict_err:{e}"
            return

        detection_model = None
        last_err = ""

        # Try AutoDetectionModel
        try:
            from sahi import AutoDetectionModel
            # SAHI expects device string like "cpu" or "cuda:0"
            detection_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=self.model_path,
                confidence_threshold=float(self.conf),
                device=str(self.device),
            )
        except Exception as e:
            last_err = f"auto_model_err:{e}"
            detection_model = None

        # Fallback explicit Yolov8DetectionModel (some SAHI versions)
        if detection_model is None:
            try:
                from sahi.models.yolov8 import Yolov8DetectionModel
                detection_model = Yolov8DetectionModel(
                    model_path=self.model_path,
                    confidence_threshold=float(self.conf),
                    device=str(self.device),
                )
            except Exception as e:
                last_err = (last_err + " | " if last_err else "") + f"yolov8_model_err:{e}"
                detection_model = None

        if detection_model is None:
            self._sahi_ok = False
            self._sahi_err = last_err or "sahi_model_init_failed"
            return

        self._sahi_model = detection_model
        self._sahi_ok = True
        self._sahi_err = ""

    def _get_roi(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        return roi_bgr, (x_offset, y_offset) to map bbox back to full frame
        """
        if self.roi_mode != "FLOOR":
            return frame_bgr, (0, 0)

        H, W = frame_bgr.shape[:2]
        y1 = int(H * self.roi_y1_ratio)
        y2 = int(H * self.roi_y2_ratio)
        y1 = max(0, min(H - 2, y1))
        y2 = max(y1 + 2, min(H, y2))
        roi = frame_bgr[y1:y2, :]
        return roi, (0, y1)

    def infer(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if frame_bgr is None or frame_bgr.size < 10:
            return []

        self._frame_i += 1
        if (self._frame_i % self.every_n_frames) != 0:
            return self._last

        H0, W0 = frame_bgr.shape[:2]
        roi_bgr, (ox, oy) = self._get_roi(frame_bgr)
        Hr, Wr = roi_bgr.shape[:2]

        # If ROI too small => fallback full
        if Hr < 60 or Wr < 120:
            roi_bgr = frame_bgr
            ox, oy = 0, 0
            Hr, Wr = H0, W0

        out: List[Dict[str, Any]] = []

        # ---- SAHI path (best for small objects) ----
        if self.use_sahi and self._sahi_ok and self._sahi_model is not None and self._sahi_get_sliced_prediction is not None:
            try:
                # SAHI often assumes RGB; convert to be safe
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

                sliced = self._sahi_get_sliced_prediction(
                    image=roi_rgb,
                    detection_model=self._sahi_model,
                    slice_height=int(self.sahi_slice_h),
                    slice_width=int(self.sahi_slice_w),
                    overlap_height_ratio=float(self.sahi_overlap),
                    overlap_width_ratio=float(self.sahi_overlap),
                    postprocess_type=str(self.sahi_postprocess),
                    postprocess_match_metric="IOU",
                    postprocess_match_threshold=float(self.iou),
                    verbose=False,
                )

                obj_list = getattr(sliced, "object_prediction_list", None) or []
                for obj in obj_list:
                    try:
                        # category
                        cat = getattr(obj, "category", None)
                        cls_id = int(getattr(cat, "id", -1)) if cat is not None else -1
                        name = str(getattr(cat, "name", "obj")) if cat is not None else "obj"

                        # score
                        score = getattr(obj, "score", None)
                        conf = float(getattr(score, "value", 0.0)) if score is not None else 0.0

                        # bbox in ROI coords
                        bb = getattr(obj, "bbox", None)
                        if bb is None:
                            continue
                        x1 = int(getattr(bb, "minx", 0))
                        y1 = int(getattr(bb, "miny", 0))
                        x2 = int(getattr(bb, "maxx", 0))
                        y2 = int(getattr(bb, "maxy", 0))

                        # map back to full frame coords
                        x1f = max(0, min(W0 - 1, x1 + ox))
                        y1f = max(0, min(H0 - 1, y1 + oy))
                        x2f = max(0, min(W0 - 1, x2 + ox))
                        y2f = max(0, min(H0 - 1, y2 + oy))
                        if x2f <= x1f or y2f <= y1f:
                            continue

                        out.append({"cls": cls_id, "name": name, "conf": conf, "bbox": [x1f, y1f, x2f, y2f], "via": "sahi"})
                    except Exception:
                        continue

                # limit max_det
                if len(out) > self.max_det:
                    out.sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)
                    out = out[: self.max_det]

                self._last = out
                self._last_ts = time.time()
                return out

            except Exception as e:
                # SAHI fail -> fallback YOLO thường
                self._sahi_ok = False
                self._sahi_err = f"sahi_runtime_err:{e}"

        # ---- YOLO normal path ----
        try:
            res = self.model.predict(
                source=roi_bgr,
                imgsz=self.img_size,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                max_det=self.max_det,
                verbose=False
            )

            if not res:
                self._last = []
                return []

            r0 = res[0]
            names = getattr(r0, "names", None) or getattr(self.model, "names", {})

            boxes = getattr(r0, "boxes", None)
            if boxes is None:
                self._last = []
                return []

            for b in boxes:
                try:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    conf = float(b.conf[0].cpu().numpy())
                    cls = int(b.cls[0].cpu().numpy())
                except Exception:
                    continue

                x1, y1, x2, y2 = xyxy

                # map ROI coords -> full frame coords
                x1f = max(0, min(W0 - 1, int(x1 + ox)))
                y1f = max(0, min(H0 - 1, int(y1 + oy)))
                x2f = max(0, min(W0 - 1, int(x2 + ox)))
                y2f = max(0, min(H0 - 1, int(y2 + oy)))
                if x2f <= x1f or y2f <= y1f:
                    continue

                name = str(names.get(cls, str(cls))) if isinstance(names, dict) else str(cls)
                out.append({"cls": cls, "name": name, "conf": conf, "bbox": [x1f, y1f, x2f, y2f], "via": "yolo"})

            self._last = out
            self._last_ts = time.time()
            return out

        except Exception:
            self._last = []
            return []


# =========================
# 3) Fusion class
# =========================
class VisionObstacleFusion:
    """
    - update(frame_bgr) => quyết định
    - get_decision()
    - get_boxes_payload()
    - draw_overlay()
    """

    def __init__(
        self,
        enable_yolo: bool = True,
        yolo_model: str = "yolov8n.pt",
        yolo_imgsz: int = 416,
        yolo_conf: float = 0.30,     # ↓ chút để bắt vật nhỏ
        yolo_iou: float = 0.45,
        device: str = "cpu",
        yolo_every_n: int = 2,

        # SAHI
        yolo_use_sahi: bool = True,
        sahi_slice_w: int = 320,
        sahi_slice_h: int = 320,
        sahi_overlap: float = 0.22,
        sahi_postprocess: str = "NMS",

        # ROI cho YOLO (để nhẹ + tăng pixel cho vật nhỏ)
        # "FULL" | "FLOOR"
        yolo_roi_mode: str = "FLOOR",
        yolo_roi_y1_ratio: float = 0.42,
        yolo_roi_y2_ratio: float = 0.98,

        # ROI nguy hiểm (vật cản trước mặt)
        danger_y1_ratio: float = 0.45,
        danger_y2_ratio: float = 0.98,
        danger_x1_ratio: float = 0.20,
        danger_x2_ratio: float = 0.80,

        # lọc box nhỏ (để bắt chân bàn: giảm mạnh)
        min_box_area_ratio: float = 0.0025,   # ✅ nhỏ hơn trước

        # bump detector tuning
        bump_roi_y1_ratio: float = 0.58,
        bump_roi_y2_ratio: float = 0.98,

        # leg detector tuning
        enable_leg: bool = True,
        leg_roi_y1_ratio: float = 0.25,
        leg_roi_y2_ratio: float = 0.98,

        # prefer classes (YOLO)
        prefer_classes: Optional[List[str]] = None,

        thread_safe: bool = True,
    ):
        self.dy1 = float(danger_y1_ratio)
        self.dy2 = float(danger_y2_ratio)
        self.dx1 = float(danger_x1_ratio)
        self.dx2 = float(danger_x2_ratio)
        self.min_box_area_ratio = float(min_box_area_ratio)

        self.prefer_classes = [c.lower() for c in (prefer_classes or ["person", "chair", "table", "bench", "sofa"])]

        self.bump = FloorBumpDetector(
            roi_y1_ratio=bump_roi_y1_ratio,
            roi_y2_ratio=bump_roi_y2_ratio
        )

        self.leg: Optional[VerticalLegDetector] = None
        self.leg_status: Dict[str, Any] = {"enabled": bool(enable_leg), "ok": False, "err": ""}

        if enable_leg:
            try:
                self.leg = VerticalLegDetector(
                    roi_y1_ratio=leg_roi_y1_ratio,
                    roi_y2_ratio=leg_roi_y2_ratio
                )
                self.leg_status["ok"] = True
            except Exception as e:
                self.leg = None
                self.leg_status["ok"] = False
                self.leg_status["err"] = str(e)

        self.yolo: Optional[YoloDetector] = None
        self.yolo_status: Dict[str, Any] = {
            "enabled": bool(enable_yolo),
            "ok": False,
            "err": "",
            "sahi": {"enabled": bool(yolo_use_sahi), "ok": False, "err": ""},
            "roi_mode": str(yolo_roi_mode).upper(),
        }

        if enable_yolo:
            try:
                self.yolo = YoloDetector(
                    model_path=yolo_model,
                    img_size=yolo_imgsz,
                    conf=yolo_conf,
                    iou=yolo_iou,
                    device=device,
                    max_det=60,
                    every_n_frames=yolo_every_n,

                    use_sahi=yolo_use_sahi,
                    sahi_slice_w=sahi_slice_w,
                    sahi_slice_h=sahi_slice_h,
                    sahi_overlap=sahi_overlap,
                    sahi_postprocess=sahi_postprocess,

                    roi_mode=yolo_roi_mode,
                    roi_y1_ratio=yolo_roi_y1_ratio,
                    roi_y2_ratio=yolo_roi_y2_ratio,
                )
                self.yolo_status["ok"] = True
                # SAHI status
                if self.yolo is not None:
                    self.yolo_status["sahi"]["ok"] = bool(getattr(self.yolo, "_sahi_ok", False))
                    self.yolo_status["sahi"]["err"] = str(getattr(self.yolo, "_sahi_err", "") or "")
            except Exception as e:
                self.yolo = None
                self.yolo_status["ok"] = False
                self.yolo_status["err"] = str(e)

        self._lock = threading.Lock() if thread_safe else None
        self._last_dec = FusionDecision(False, "none", 0.0, "NONE", "none", {"reason": "init"})
        self._last_boxes_payload: Dict[str, Any] = {"ok": False, "reason": "init"}
        self._last_frame_wh = (0, 0)

    # ---------- utils ----------
    @staticmethod
    def _zone_from_cx(cx: float, W: int) -> str:
        if cx < W / 3:
            return "LEFT"
        if cx > 2 * W / 3:
            return "RIGHT"
        return "CENTER"

    @staticmethod
    def _iou_rect(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    def _set_state(self, dec: FusionDecision, payload: Dict[str, Any], wh: Tuple[int, int]):
        if self._lock:
            with self._lock:
                self._last_dec = dec
                self._last_boxes_payload = payload
                self._last_frame_wh = wh
        else:
            self._last_dec = dec
            self._last_boxes_payload = payload
            self._last_frame_wh = wh

    def _get_state(self) -> Tuple[FusionDecision, Dict[str, Any], Tuple[int, int]]:
        if self._lock:
            with self._lock:
                return self._last_dec, dict(self._last_boxes_payload), tuple(self._last_frame_wh)
        return self._last_dec, dict(self._last_boxes_payload), tuple(self._last_frame_wh)

    # ---------- core ----------
    def update(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        now = time.time()
        if frame_bgr is None or frame_bgr.size < 10:
            dec = FusionDecision(False, "none", 0.0, "NONE", "none", {"reason": "no_frame"})
            payload = {"ok": False, "reason": "no_frame", "ts": now}
            self._set_state(dec, payload, (0, 0))
            return self.get_decision()

        H, W = frame_bgr.shape[:2]
        danger = [int(W * self.dx1), int(H * self.dy1), int(W * self.dx2), int(H * self.dy2)]

        bump_hit = self.bump.detect(frame_bgr)

        leg_hit = LegHit(False, 0.0, None, "disabled")
        if self.leg is not None:
            try:
                leg_hit = self.leg.detect(frame_bgr)
            except Exception as e:
                leg_hit = LegHit(False, 0.0, None, f"leg_err:{e}")

        dets: List[Dict[str, Any]] = []
        yolo_err = ""
        if self.yolo is not None:
            try:
                dets = self.yolo.infer(frame_bgr)
                # update SAHI status live
                self.yolo_status["sahi"]["ok"] = bool(getattr(self.yolo, "_sahi_ok", False))
                self.yolo_status["sahi"]["err"] = str(getattr(self.yolo, "_sahi_err", "") or "")
            except Exception as e:
                dets = []
                yolo_err = str(e)

        # pick best yolo inside danger zone
        best_yolo = None
        best_score = -1.0
        for d in dets:
            conf = float(d.get("conf", 0.0))
            x1, y1, x2, y2 = map(int, d.get("bbox", [0, 0, 0, 0]))

            area = max(0, (x2 - x1) * (y2 - y1))
            if (W * H) > 0 and (area / float(W * H)) < self.min_box_area_ratio:
                continue

            iou = self._iou_rect([x1, y1, x2, y2], danger)
            if iou <= 0.015:
                continue

            name = str(d.get("name", "obj"))
            bonus = 0.15 if (name.lower() in self.prefer_classes) else 0.0
            via = str(d.get("via", "yolo"))
            bonus2 = 0.06 if via == "sahi" else 0.0  # ưu tiên nhẹ cho SAHI khi cạnh tranh
            score = conf + 0.85 * iou + bonus + bonus2

            if score > best_score:
                best_score = score
                best_yolo = {**d, "iou_danger": float(iou), "score": float(score)}

        # leg meaningful if overlap danger zone
        leg_ok = False
        leg_zone = "CENTER"
        if leg_hit.ok and leg_hit.bbox:
            lx1, ly1, lx2, ly2 = leg_hit.bbox
            if self._iou_rect([lx1, ly1, lx2, ly2], danger) > 0.08:
                leg_ok = True
                leg_zone = self._zone_from_cx(0.5 * (lx1 + lx2), W)

        # bump meaningful
        bump_ok = False
        bump_zone = "CENTER"
        if bump_hit.ok and bump_hit.band_bbox:
            bx1, by1, bx2, by2 = bump_hit.band_bbox
            if self._iou_rect([bx1, by1, bx2, by2], danger) > 0.10:
                cx = 0.5 * (bx1 + bx2)
                bump_zone = self._zone_from_cx(cx, W)
                bump_ok = (bump_zone == "CENTER")

        # ---------------- fusion decision ----------------
        # Ưu tiên: YOLO (SAHI) > LEG > BUMP
        if best_yolo is not None:
            typ = str(best_yolo.get("name", "object"))
            conf = float(best_yolo.get("conf", 0.0))
            zone = self._zone_from_cx(0.5 * (best_yolo["bbox"][0] + best_yolo["bbox"][2]), W)
            src = "yolo_sahi" if str(best_yolo.get("via", "")) == "sahi" else "yolo"
            dec = FusionDecision(
                True, typ, conf, zone, src,
                {"danger": {"bbox": danger},
                 "leg": {"ok": leg_hit.ok, "score": leg_hit.score, "bbox": leg_hit.bbox, "reason": leg_hit.reason},
                 "bump": {"ok": bump_hit.ok, "score": bump_hit.score, "bbox": bump_hit.band_bbox, "reason": bump_hit.reason},
                 "yolo": best_yolo, "yolo_err": yolo_err}
            )

        elif leg_ok:
            conf = float(min(0.99, max(0.50, leg_hit.score)))
            dec = FusionDecision(
                True, "table_leg", conf, leg_zone, "leg",
                {"danger": {"bbox": danger},
                 "leg": {"score": leg_hit.score, "bbox": leg_hit.bbox, "reason": leg_hit.reason},
                 "bump": {"ok": bump_hit.ok, "score": bump_hit.score, "bbox": bump_hit.band_bbox, "reason": bump_hit.reason},
                 "yolo": None, "yolo_err": yolo_err}
            )

        elif bump_ok:
            conf = float(min(0.99, 0.55 + 0.45 * float(bump_hit.score)))
            dec = FusionDecision(
                True, "floor_bump", conf, bump_zone, "bump",
                {"danger": {"bbox": danger},
                 "leg": {"ok": leg_hit.ok, "score": leg_hit.score, "bbox": leg_hit.bbox, "reason": leg_hit.reason},
                 "bump": {"score": bump_hit.score, "bbox": bump_hit.band_bbox, "reason": bump_hit.reason},
                 "yolo": None, "yolo_err": yolo_err}
            )
        else:
            dec = FusionDecision(
                False, "none", 0.0, "NONE", "none",
                {"danger": {"bbox": danger},
                 "leg": {"ok": leg_hit.ok, "score": leg_hit.score, "bbox": leg_hit.bbox, "reason": leg_hit.reason},
                 "bump": {"ok": bump_hit.ok, "score": bump_hit.score, "bbox": bump_hit.band_bbox, "reason": bump_hit.reason},
                 "yolo": None, "yolo_err": yolo_err}
            )

        # ---------------- build boxes payload ----------------
        boxes: List[Dict[str, Any]] = []

        if bump_hit.band_bbox:
            bx1, by1, bx2, by2 = bump_hit.band_bbox
            boxes.append({
                "kind": "bump",
                "name": "floor_bump",
                "conf": float(min(0.99, max(0.0, bump_hit.score / 2.0))),
                "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
                "zone": self._zone_from_cx(0.5 * (bx1 + bx2), W),
                "meta": {"reason": bump_hit.reason, "ok": bool(bump_hit.ok), "score": float(bump_hit.score)}
            })

        if leg_hit.bbox:
            lx1, ly1, lx2, ly2 = leg_hit.bbox
            boxes.append({
                "kind": "leg",
                "name": "table_leg",
                "conf": float(min(0.99, max(0.0, leg_hit.score))),
                "bbox": [int(lx1), int(ly1), int(lx2), int(ly2)],
                "zone": self._zone_from_cx(0.5 * (lx1 + lx2), W),
                "meta": {"reason": leg_hit.reason, "ok": bool(leg_hit.ok), "score": float(leg_hit.score)}
            })

        for d in dets:
            conf = float(d.get("conf", 0.0))
            if conf < 1e-6:
                continue
            x1, y1, x2, y2 = map(int, d.get("bbox", [0, 0, 0, 0]))
            if x2 <= x1 or y2 <= y1:
                continue
            area = max(0, (x2 - x1) * (y2 - y1))
            if (W * H) > 0 and (area / float(W * H)) < self.min_box_area_ratio:
                continue
            iou = self._iou_rect([x1, y1, x2, y2], danger)
            boxes.append({
                "kind": "yolo",
                "name": str(d.get("name", "obj")),
                "conf": float(conf),
                "bbox": [x1, y1, x2, y2],
                "zone": self._zone_from_cx(0.5 * (x1 + x2), W),
                "meta": {"iou_danger": float(iou), "cls": int(d.get("cls", -1)), "via": str(d.get("via", "yolo"))}
            })

        payload = {
            "ok": True,
            "ts": now,
            "frame": {"w": int(W), "h": int(H)},
            "danger_bbox": danger,
            "decision": {
                "has_obstacle": bool(dec.has_obstacle),
                "obstacle_type": dec.obstacle_type,
                "confidence": float(dec.confidence),
                "zone": dec.zone,
                "source": dec.source
            },
            "boxes": boxes,
            "yolo_status": dict(self.yolo_status),
            "leg_status": dict(self.leg_status),
        }

        self._set_state(dec, payload, (W, H))
        return self.get_decision()

    # ---------- getters ----------
    def get_decision(self) -> Dict[str, Any]:
        dec, _, _ = self._get_state()
        return {
            "ok": True,
            "has_obstacle": bool(dec.has_obstacle),
            "obstacle_type": dec.obstacle_type,
            "confidence": float(dec.confidence),
            "zone": dec.zone,
            "source": dec.source,
            "detail": dec.detail,
            "yolo_status": dict(self.yolo_status),
            "leg_status": dict(self.leg_status),
        }

    def get_boxes_payload(self) -> Dict[str, Any]:
        _, payload, _ = self._get_state()
        return payload

    # ---------- overlay ----------
    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        out = frame_bgr.copy()
        if out is None or out.size < 10:
            return out

        payload = self.get_boxes_payload()
        if not payload.get("ok", False):
            return out

        danger = payload.get("danger_bbox", None)
        if isinstance(danger, list) and len(danger) == 4:
            x1, y1, x2, y2 = map(int, danger)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 2)

        boxes = payload.get("boxes", [])
        if isinstance(boxes, list):
            for b in boxes:
                try:
                    kind = str(b.get("kind", "obj"))
                    name = str(b.get("name", "obj"))
                    conf = float(b.get("conf", 0.0))
                    x1, y1, x2, y2 = map(int, b.get("bbox", [0, 0, 0, 0]))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # colors:
                    # yolo: green (SAHI brighter)
                    # bump: yellow
                    # leg: cyan
                    if kind == "yolo":
                        via = ""
                        try:
                            via = str((b.get("meta", {}) or {}).get("via", ""))
                        except Exception:
                            via = ""
                        color = (0, 255, 80) if via == "sahi" else (0, 255, 0)
                    elif kind == "bump":
                        color = (0, 255, 255)
                    elif kind == "leg":
                        color = (255, 255, 0)
                    else:
                        color = (255, 255, 255)

                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

                    tag = name
                    if kind == "yolo":
                        try:
                            via = str((b.get("meta", {}) or {}).get("via", ""))
                            if via:
                                tag = f"{name}({via})"
                        except Exception:
                            pass

                    cv2.putText(out, f"{tag} {conf:.2f}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception:
                    pass

        dec = payload.get("decision", {})
        try:
            txt = f"OBS={dec.get('has_obstacle')} type={dec.get('obstacle_type')} conf={float(dec.get('confidence',0.0)):.2f} zone={dec.get('zone')} src={dec.get('source')}"
            cv2.putText(out, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
            cv2.putText(out, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        except Exception:
            pass

        return out
