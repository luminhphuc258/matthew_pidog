#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vision_obstacle_fusion.py

Class độc lập để:
- detect gò/gờ trên sàn bằng OpenCV (FloorBumpDetector)
- detect chân bàn/cột mảnh (VerticalLegDetector)  ✅ NEW
- detect vật cản bằng YOLO (Ultralytics)
- fuse kết quả => quyết định obstacle trước mặt + loại vật cản
- export boxes để file khác vẽ lên camera frame

Usage:

from vision_obstacle_fusion import VisionObstacleFusion

fusion = VisionObstacleFusion(
    enable_yolo=True,
    yolo_model="/home/matthewlupi/matthew_pidog/yolov8n.pt",
    device="cpu",
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
# 1) Floor bump detector (OpenCV) - giữ như bạn
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
# 1.5) Vertical leg / pole detector (OpenCV) ✅ NEW
# =========================
class VerticalLegDetector:
    """
    Bắt "chân bàn/cột mảnh" (vật thể dọc, hẹp, cao) bằng gradient-x + morphology + contour filter.

    Mục tiêu: trường hợp YOLO không bắt được table/chair nhưng chân bàn rõ trong ảnh.
    """
    def __init__(
        self,
        roi_y1_ratio: float = 0.25,     # bắt đầu cao hơn để thấy chân bàn (ảnh của bạn chân bàn lên khá cao)
        roi_y2_ratio: float = 0.98,
        min_height_ratio: float = 0.28, # chân bàn phải đủ cao
        max_width_ratio: float = 0.22,  # chân bàn phải hẹp (so với khung hình)
        min_area_ratio: float = 0.003,  # lọc noise
        aspect_min: float = 2.8,        # h/w tối thiểu
        bottom_touch_ratio: float = 0.16,  # đáy bbox gần đáy ROI (chân bàn chạm sàn)
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
        H, W = frame_bgr.shape[:2]
        rh, rw = roi.shape[:2]
        if rh < 60 or rw < 120:
            return LegHit(False, float(self._ema), None, "roi_small")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # vertical structures -> gradient theo trục x
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        agx = cv2.convertScaleAbs(gx)

        # normalize + Otsu threshold (tự thích nghi ánh sáng)
        agx = cv2.GaussianBlur(agx, (5, 5), 0)
        _, bw = cv2.threshold(agx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # morphology: nối dọc (vertical close)
        vk = max(11, int(rh * 0.10))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vk))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_v, iterations=1)

        # dày thêm chút để contour ổn định
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

            # size filters
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

            # chân bàn thường chạm gần đáy ROI
            if (y + h) < (roi_bottom - bottom_margin):
                continue

            # edge density trong bbox (để chắc là có “structure”)
            patch = bw[y:y+h, x:x+w]
            ed = float(np.mean(patch > 0)) if patch.size else 0.0  # 0..1

            # score: ưu tiên cao + hẹp + edge density vừa đủ
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
        # convert roi->full frame
        bbox_full = (int(x1), int(y1f + y1), int(x2), int(y1f + y2))

        self._last_bbox = bbox_full
        self._last_reason = f"leg(hr={best['hr']:.2f} wr={best['wr']:.2f} asp={best['aspect']:.1f} ed={best['ed']:.2f}) ema={self._ema:.2f} streak={self._streak}"
        return LegHit(bool(ok), float(self._ema), bbox_full, self._last_reason)


# =========================
# 2) YOLO detector (Ultralytics)
# =========================
class YoloDetector:
    def __init__(
        self,
        model_path: str,
        img_size: int = 416,
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "cpu",
        max_det: int = 30,
        every_n_frames: int = 2
    ):
        self.model_path = str(model_path)
        self.img_size = int(img_size)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = str(device)
        self.max_det = int(max_det)
        self.every_n_frames = max(1, int(every_n_frames))

        self._frame_i = 0
        self._last: List[Dict[str, Any]] = []
        self._last_ts = 0.0

        from ultralytics import YOLO
        self.model = YOLO(self.model_path)

    def infer(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if frame_bgr is None or frame_bgr.size < 10:
            return []

        self._frame_i += 1
        if (self._frame_i % self.every_n_frames) != 0:
            return self._last

        H, W = frame_bgr.shape[:2]
        res = self.model.predict(
            source=frame_bgr,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            max_det=self.max_det,
            verbose=False
        )

        out: List[Dict[str, Any]] = []
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
            x1 = max(0, min(W - 1, int(x1)))
            y1 = max(0, min(H - 1, int(y1)))
            x2 = max(0, min(W - 1, int(x2)))
            y2 = max(0, min(H - 1, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            name = str(names.get(cls, str(cls))) if isinstance(names, dict) else str(cls)
            out.append({"cls": cls, "name": name, "conf": conf, "bbox": [x1, y1, x2, y2]})

        self._last = out
        self._last_ts = time.time()
        return out


# =========================
# 3) Fusion class
# =========================
class VisionObstacleFusion:
    """
    Class độc lập:
      - update(frame_bgr) => quyết định
      - get_decision() => dict decision
      - get_boxes_payload() => dict boxes để web/overlay
      - draw_overlay(frame_bgr) => vẽ debug

    Không tạo webserver, không endpoints.
    """

    def __init__(
        self,
        enable_yolo: bool = True,
        yolo_model: str = "yolov8n.pt",
        yolo_imgsz: int = 416,
        yolo_conf: float = 0.35,
        yolo_iou: float = 0.45,
        device: str = "cpu",
        yolo_every_n: int = 2,

        # ROI nguy hiểm (vật cản trước mặt)
        danger_y1_ratio: float = 0.45,
        danger_y2_ratio: float = 0.98,
        danger_x1_ratio: float = 0.20,
        danger_x2_ratio: float = 0.80,

        # lọc box nhỏ
        min_box_area_ratio: float = 0.010,

        # bump detector tuning
        bump_roi_y1_ratio: float = 0.58,
        bump_roi_y2_ratio: float = 0.98,

        # leg detector tuning ✅
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

        # thêm table/chair ưu tiên, vì chân bàn hay đi kèm
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
        self.yolo_status: Dict[str, Any] = {"enabled": bool(enable_yolo), "ok": False, "err": ""}

        if enable_yolo:
            try:
                self.yolo = YoloDetector(
                    model_path=yolo_model,
                    img_size=yolo_imgsz,
                    conf=yolo_conf,
                    iou=yolo_iou,
                    device=device,
                    every_n_frames=yolo_every_n
                )
                self.yolo_status["ok"] = True
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
            except Exception as e:
                dets = []
                yolo_err = str(e)

        # pick best yolo inside danger zone
        best_yolo = None
        best_score = -1.0
        for d in dets:
            conf = float(d.get("conf", 0.0))
            x1, y1, x2, y2 = map(int, d.get("bbox", [0, 0, 0, 0]))
            if conf < 0.0:
                continue
            area = max(0, (x2 - x1) * (y2 - y1))
            if (W * H) > 0 and (area / float(W * H)) < self.min_box_area_ratio:
                continue

            iou = self._iou_rect([x1, y1, x2, y2], danger)
            if iou <= 0.02:
                continue

            name = str(d.get("name", "obj"))
            bonus = 0.15 if (name.lower() in self.prefer_classes) else 0.0
            score = conf + 0.8 * iou + bonus
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

        # bump meaningful (giữ logic cũ)
        bump_ok = False
        bump_zone = "CENTER"
        if bump_hit.ok and bump_hit.band_bbox:
            bx1, by1, bx2, by2 = bump_hit.band_bbox
            if self._iou_rect([bx1, by1, bx2, by2], danger) > 0.10:
                cx = 0.5 * (bx1 + bx2)
                bump_zone = self._zone_from_cx(cx, W)
                bump_ok = (bump_zone == "CENTER")

        # ---------------- fusion decision ----------------
        # Ưu tiên: YOLO (nếu có) > LEG > BUMP
        if best_yolo is not None:
            typ = str(best_yolo.get("name", "object"))
            conf = float(best_yolo.get("conf", 0.0))
            zone = self._zone_from_cx(0.5 * (best_yolo["bbox"][0] + best_yolo["bbox"][2]), W)
            src = "yolo"
            dec = FusionDecision(
                True, typ, conf, zone, src,
                {"danger": {"bbox": danger},
                 "leg": {"ok": leg_hit.ok, "score": leg_hit.score, "bbox": leg_hit.bbox, "reason": leg_hit.reason},
                 "bump": {"ok": bump_hit.ok, "score": bump_hit.score, "bbox": bump_hit.band_bbox, "reason": bump_hit.reason},
                 "yolo": best_yolo, "yolo_err": yolo_err}
            )

        elif leg_ok:
            # chân bàn/cột mảnh
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

        # danger ROI
        # (để file khác vẽ)
        # bump box
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

        # leg box ✅
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

        # yolo boxes
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
                "meta": {"iou_danger": float(iou), "cls": int(d.get("cls", -1))}
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

        # draw danger ROI
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
                    # yolo: green
                    # bump: yellow
                    # leg: cyan
                    if kind == "yolo":
                        color = (0, 255, 0)
                    elif kind == "bump":
                        color = (0, 255, 255)
                    elif kind == "leg":
                        color = (255, 255, 0)
                    else:
                        color = (255, 255, 255)

                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out, f"{name} {conf:.2f}", (x1, max(20, y1 - 8)),
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
