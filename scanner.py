"""
Auto Document Scanner — Text-based detection
Author: Bhai ka project 😎
"""

import cv2
import numpy as np
import os
import sys
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_DIM = 1600              # resize max side
MIN_TEXT_AREA = 400         # min text box area (px)
MAX_ASPECT = 15.0           # text box aspect limit
CLUSTER_GAP = 40            # gap to merge text boxes
PADDING = 25                # crop padding
CONF_THRESHOLD = 0.5        # OCR confidence

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
log = logging.getLogger("DocScanner")


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class TextBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    text: str = ""

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


# ─────────────────────────────────────────────
# MAIN SCANNER
# ─────────────────────────────────────────────
class DocumentScanner:
    def __init__(self, lang: str = "en", debug: bool = False):
        self.lang = lang
        self.debug = debug
        self.ocr = None
        self._init_ocr()

    # ---------- OCR setup ----------
    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            log.info("Loading PaddleOCR...")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                show_log=False
            )
            log.info("PaddleOCR ready ✅")
        except Exception as e:
            log.warning(f"PaddleOCR failed: {e}")
            log.warning("Fallback: OpenCV-only mode")
            self.ocr = None

    # ---------- Load image ----------
    def load_image(self, path: str) -> Optional[np.ndarray]:
        if not os.path.exists(path):
            log.error(f"File not found: {path}")
            return None

        img = cv2.imread(path)
        if img is None:
            log.error("Could not read image")
            return None

        # Resize if too big
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            log.info(f"Resized to {img.shape[1]}x{img.shape[0]}")

        return img

    # ---------- Text detection ----------
    def detect_text_regions(self, img: np.ndarray) -> List[TextBox]:
        boxes: List[TextBox] = []

        if self.ocr is not None:
            try:
                result = self.ocr.ocr(img, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        pts = line[0]
                        text, conf = line[1][0], line[1][1]
                        if conf < CONF_THRESHOLD:
                            continue
                        xs = [int(p[0]) for p in pts]
                        ys = [int(p[1]) for p in pts]
                        boxes.append(TextBox(
                            min(xs), min(ys), max(xs), max(ys),
                            conf, text
                        ))
            except Exception as e:
                log.warning(f"OCR error: {e}")

        # Fallback: OpenCV-based text detection
        if len(boxes) < 3:
            log.info("Using OpenCV fallback for text detection")
            boxes.extend(self._cv_text_detect(img))

        # Filter
        filtered = []
        for b in boxes:
            if b.area < MIN_TEXT_AREA:
                continue
            if b.height < 10:
                continue
            aspect = b.width / max(b.height, 1)
            if aspect > MAX_ASPECT:
                continue
            filtered.append(b)

        log.info(f"Detected {len(filtered)} text boxes")
        return filtered

    def _cv_text_detect(self, img: np.ndarray) -> List[TextBox]:
        """OpenCV blackhat-based text detection fallback"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
        grad = np.absolute(grad)
        grad = cv2.normalize(grad, grad, 0, 255, cv2.NORM_MINMAX)
        grad = grad.astype("uint8")
        grad = cv2.GaussianBlur(grad, (5, 5), 0)

        _, thresh = cv2.threshold(
            grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=3)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < MIN_TEXT_AREA or h < 10:
                continue
            boxes.append(TextBox(x, y, x + w, y + h, 1.0))
        return boxes

    # ---------- Clustering ----------
    def cluster_text_boxes(self, boxes: List[TextBox]) -> List[List[TextBox]]:
        if not boxes:
            return []

        n = len(boxes)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

        for i in range(n):
            for j in range(i + 1, n):
                if self._boxes_close(boxes[i], boxes[j], CLUSTER_GAP):
                    union(i, j)

        groups: Dict[int, List[TextBox]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(boxes[i])

        clusters = list(groups.values())
        log.info(f"Formed {len(clusters)} clusters")
        return clusters

    @staticmethod
    def _boxes_close(a: TextBox, b: TextBox, gap: int) -> bool:
        return not (
            a.x2 + gap < b.x1 or b.x2 + gap < a.x1 or
            a.y2 + gap < b.y1 or b.y2 + gap < a.y1
        )

    @staticmethod
    def _cluster_bbox(cluster: List[TextBox]) -> Tuple[int, int, int, int]:
        x1 = min(b.x1 for b in cluster)
        y1 = min(b.y1 for b in cluster)
        x2 = max(b.x2 for b in cluster)
        y2 = max(b.y2 for b in cluster)
        return x1, y1, x2, y2

    @staticmethod
    def _cluster_score(cluster: List[TextBox]) -> float:
        x1, y1, x2, y2 = DocumentScanner._cluster_bbox(cluster)
        area = (x2 - x1) * (y2 - y1)
        n = len(cluster)
        text_area = sum(b.area for b in cluster)
        density = text_area / max(area, 1)
        # Bada area + zyada text boxes + high density = best
        return area * (1 + n * 0.1) * (1 + density)

    # ---------- Best cluster ----------
    def pick_best_cluster(self, clusters: List[List[TextBox]]) -> Optional[List[TextBox]]:
        if not clusters:
            return None
        scored = [(self._cluster_score(c), c) for c in clusters]
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        log.info(f"Best cluster has {len(best)} text boxes")
        return best

    # ---------- Contour refinement ----------
    def refine_with_contour(
        self, img: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        pad = 60
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.shape[1], x2 + pad)
        y2 = min(img.shape[0], y2 + pad)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Try multiple thresholds
        for low, high in [(30, 100), (75, 200), (50, 150)]:
            edged = cv2.Canny(gray, low, high)
            k = np.ones((5, 5), np.uint8)
            edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, k)

            contours, _ = cv2.findContours(
                edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            roi_area = roi.shape[0] * roi.shape[1]
            for c in contours:
                area = cv2.contourArea(c)
                if area < 0.4 * roi_area:
                    continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2)
                    pts[:, 0] += x1
                    pts[:, 1] += y1
                    return pts.astype("float32")
        return None

    # ---------- Perspective transform ----------
    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def warp_perspective(self, img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        wA = np.linalg.norm(br - bl)
        wB = np.linalg.norm(tr - tl)
        maxW = max(int(wA), int(wB))

        hA = np.linalg.norm(tr - br)
        hB = np.linalg.norm(tl - bl)
        maxH = max(int(hA), int(hB))

        if maxW < 50 or maxH < 50:
            return img

        dst = np.array([
            [0, 0],
            [maxW - 1, 0],
            [maxW - 1, maxH - 1],
            [0, maxH - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (maxW, maxH))

    # ---------- Scan enhancement ----------
    @staticmethod
    def enhance_scan(img: np.ndarray, mode: str = "color") -> np.ndarray:
        if mode == "gray":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 10
            )
        else:
            # Shadow removal
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            planes = cv2.split(rgb)
            result = []
            for p in planes:
                blur = cv2.GaussianBlur(p, (0, 0), sigmaX=30)
                div = cv2.divide(p, blur, scale=255)
                result.append(div)
            merged = cv2.merge(result)
            out = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
            # Slight contrast boost
            out = cv2.convertScaleAbs(out, alpha=1.15, beta=5)
            return out

    # ---------- Detect document type ----------
    @staticmethod
    def detect_type(text: str) -> str:
        import re
        t = text.replace(" ", "").upper()
        if re.search(r"\d{4}\d{4}\d{4}", t) or "AADHAAR" in t or "आधार" in text:
            return "aadhar"
        if re.search(r"[A-Z]{5}\d{4}[A-Z]", t) or "INCOMETAX" in t:
            return "pan"
        if "MARKSHEET" in t or "STATEMENTOFMARKS" in t or "MARKS" in t:
            return "marksheet"
        if "PASSPORT" in t or "REPUBLICOFINDIA" in t:
            return "passport"
        return "unknown"

    # ---------- MAIN SCAN ----------
    def scan(self, input_path: str, output_path: str = "output.jpg") -> Dict:
        log.info(f"Scanning: {input_path}")
        img = self.load_image(input_path)
        if img is None:
            return {"success": False, "error": "Image load failed"}

        original = img.copy()

        # 1. Text detect
        boxes = self.detect_text_regions(img)
        if not boxes:
            log.warning("No text found — returning original")
            cv2.imwrite(output_path, original)
            return {
                "success": False,
                "error": "No text detected",
                "output_path": output_path,
                "confidence": 0.0
            }

        # 2. Cluster
        clusters = self.cluster_text_boxes(boxes)
        best = self.pick_best_cluster(clusters)
        if not best:
            return {"success": False, "error": "Clustering failed"}

        # 3. Bounding box
        x1, y1, x2, y2 = self._cluster_bbox(best)

        # 4. Try contour refinement
        contour = self.refine_with_contour(img, (x1, y1, x2, y2))

        if contour is not None:
            log.info("4-corner contour found → perspective warp")
            cropped = self.warp_perspective(original, contour)
            method = "perspective"
        else:
            log.info("No contour → straight crop")
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(img.shape[1], x2 + PADDING)
            y2 = min(img.shape[0], y2 + PADDING)
            cropped = original[y1:y2, x1:x2]
            method = "straight"

        # 5. Enhance
        enhanced = self.enhance_scan(cropped, mode="color")

        # 6. Save
        cv2.imwrite(output_path, enhanced)

        # 7. Extract text
        extracted = " ".join(b.text for b in best)
        doc_type = self.detect_type(extracted)

        # 8. Debug
        if self.debug:
            dbg = original.copy()
            for b in boxes:
                cv2.rectangle(dbg, (b.x1, b.y1), (b.x2, b.y2),
                              (0, 255, 0), 2)
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.imwrite("debug_boxes.jpg", dbg)
            log.info("Debug: debug_boxes.jpg saved")

        result = {
            "success": True,
            "output_path": output_path,
            "bbox": (x1, y1, x2, y2),
            "confidence": round(sum(b.conf for b in best) / len(best), 3),
            "detected_type": doc_type,
            "method": method,
            "text_boxes": len(best),
            "extracted_text": extracted[:500]
        }
        log.info(f"Done ✅ → {output_path}")
        return result


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python scanner.py <input> [output] [--debug]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "output.jpg"
    debug = "--debug" in sys.argv

    scanner = DocumentScanner(lang="en", debug=debug)
    result = scanner.scan(inp, out)

    print("\n" + "=" * 50)
    for k, v in result.items():
        print(f"{k:20s}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
