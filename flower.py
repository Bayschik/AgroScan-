"""
Модуль анализа изображений растений с использованием OpenCV.
Определяет признаки заболеваний по цвету, текстуре и морфологии листьев.
"""

import cv2
import numpy as np
import base64
from pathlib import Path



class PlantImageAnalyzer:
    """Анализатор изображений растений на основе OpenCV."""

    def __init__(self):
        self.results = {}

    # ─── Публичный API ────────────────────────────────────────────────────────

    def analyze(self, image_source) -> dict:
        """
        Принимает путь к файлу, bytes или base64-строку.
        Возвращает словарь с обнаруженными признаками и диагностическими данными.
        """
        img = self._load_image(image_source)
        if img is None:
            return {"error": "Не удалось загрузить изображение", "detected_symptoms": []}

        hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total = img.shape[0] * img.shape[1]

        color_analysis   = self._analyze_colors(hsv, total)
        texture_analysis = self._analyze_texture(gray, img)
        health_score     = self._calculate_health_score(color_analysis, texture_analysis)
        symptoms         = self._map_to_symptoms(color_analysis, texture_analysis, health_score)
        debug_images     = self._generate_debug_images(img, hsv, color_analysis)

        return {
            "detected_symptoms": symptoms,
            "color_analysis":    color_analysis,
            "texture_analysis":  texture_analysis,
            "health_score":      health_score,
            "image_stats": {
                "width":    img.shape[1],
                "height":   img.shape[0],
                "channels": img.shape[2] if len(img.shape) > 2 else 1,
            },
            "debug_images": debug_images,
        }

    # ─── Загрузка изображения ──────────────────────────────────────────────

    def _load_image(self, source):
        """Загружает изображение из файла, bytes или base64."""
        try:
            if isinstance(source, (str, Path)) and Path(source).exists():
                img = cv2.imread(str(source))
            elif isinstance(source, bytes):
                arr = np.frombuffer(source, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            elif isinstance(source, str) and source.startswith("data:image"):
                # base64 data-URL
                b64 = source.split(",", 1)[1]
                arr = np.frombuffer(base64.b64decode(b64), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                return None

            if img is not None:
                # Нормализуем размер: не более 1024 по длинной стороне
                h, w = img.shape[:2]
                if max(h, w) > 1024:
                    scale = 1024 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
            return img
        except Exception as e:
            print(f"[PlantAnalyzer] Ошибка загрузки: {e}")
            return None

    # ─── Цветовой анализ ──────────────────────────────────────────────────

    def _analyze_colors(self, hsv: np.ndarray, total: int) -> dict:
        """Вычисляет долю пикселей каждого цветового класса."""

        color_ranges = {
            "healthy_green": {
                "ranges": [([35, 40, 40], [85, 255, 255])],
                "description": "Здоровая зелёная ткань",
            },
            "yellow": {
                "ranges": [([20, 80, 100], [35, 255, 255])],
                "description": "Пожелтение (хлороз, некроз)",
            },
            "brown": {
                "ranges": [([8, 60, 40], [20, 200, 180])],
                "description": "Коричневые некрозы",
            },
            "white_pale": {
                "ranges": [([0, 0, 200], [180, 40, 255])],
                "description": "Белый/бледный налёт",
            },
            "dark_necrosis": {
                "ranges": [([0, 0, 0], [180, 255, 60])],
                "description": "Тёмные некрозы",
            },
            "red_orange": {
                "ranges": [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],
                "description": "Красно-оранжевые пятна (ржавчина, клещ)",
            },
        }

        results = {}
        for name, cfg in color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in cfg["ranges"]:
                m = cv2.inRange(hsv, np.array(lo), np.array(hi))
                mask = cv2.bitwise_or(mask, m)
            # Морфологическое сглаживание
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            pixels = int(np.sum(mask > 0))
            results[name] = {
                "pixel_count": pixels,
                "ratio":       round(pixels / max(total, 1), 4),
                "description": cfg["description"],
                "mask":        mask,
            }
        return results

    # ─── Текстурный анализ ─────────────────────────────────────────────────

    def _analyze_texture(self, gray: np.ndarray, bgr: np.ndarray) -> dict:
        """Обнаруживает текстурные аномалии: пятна, точки, ободки."""

        # Детектор пятен (Laplacian of Gaussian)
        log    = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(np.var(log))
        sharp  = variance > 100

        # CLAHE-Enhanced для обнаружения точек
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Адаптивный порог для сегментации пятен
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Поиск контуров (пятна, некрозы)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_px    = gray.shape[0] * gray.shape[1]

        spots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < total_px * 0.2:  # отсекаем шум и фон
                x, y, w, h = cv2.boundingRect(cnt)
                spots.append({
                    "area":   int(area),
                    "x": x, "y": y,
                    "w": w, "h": h,
                    "circularity": self._circularity(cnt),
                })

        spot_area = sum(s["area"] for s in spots)
        spot_ratio = spot_area / max(total_px, 1)

        # Edge density — краевые артефакты (сухие края)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / max(total_px, 1)

        # Webbing detection через тонкие линии (Hough)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=5)
        web_score = 0.0
        if lines is not None:
            # Паутинка = много коротких хаотичных линий
            lengths  = [np.hypot(l[0][2] - l[0][0], l[0][3] - l[0][1]) for l in lines]
            avg_len  = float(np.mean(lengths)) if lengths else 0
            web_score = min(len(lines) / 200, 1.0) if avg_len < 40 else 0.0

        return {
            "sharpness":      round(variance, 2),
            "is_sharp":       sharp,
            "spot_count":     len(spots),
            "spot_ratio":     round(spot_ratio, 4),
            "spots":          spots[:20],  # первые 20
            "edge_density":   round(edge_density, 4),
            "web_score":      round(web_score, 4),
            "has_many_spots": len(spots) > 15,
            "has_webbing":    web_score > 0.3,
            "has_dry_edges":  edge_density > 0.08,
        }

    @staticmethod
    def _circularity(cnt) -> float:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            return 0.0
        return round(4 * np.pi * area / (perim ** 2), 3)

    # ─── Скор здоровья ─────────────────────────────────────────────────────

    def _calculate_health_score(self, color: dict, texture: dict) -> int:
        """
        Возвращает скор здоровья 0–100.
        100 = идеально здорово, 0 = критически поражено.
        """
        green_ratio = color.get("healthy_green", {}).get("ratio", 0)
        yellow_ratio = color.get("yellow", {}).get("ratio", 0)
        brown_ratio  = color.get("brown", {}).get("ratio", 0)
        dark_ratio   = color.get("dark_necrosis", {}).get("ratio", 0)
        white_ratio  = color.get("white_pale", {}).get("ratio", 0)
        spot_ratio   = texture.get("spot_ratio", 0)

        score = 100
        score -= yellow_ratio * 120
        score -= brown_ratio  * 180
        score -= dark_ratio   * 200
        score -= white_ratio  * 100
        score -= spot_ratio   * 150
        score += green_ratio  * 30   # бонус за здоровую зелень
        score = max(0, min(100, int(score)))
        return score

    # ─── Маппинг на симптомы ──────────────────────────────────────────────

    def _map_to_symptoms(self, color: dict, texture: dict, health_score: int) -> list[str]:
        """Конвертирует числовые метрики в список симптомов (строки)."""

        syms = []

        c = {k: v.get("ratio", 0) for k, v in color.items()}
        t = texture

        if c["white_pale"] > 0.05:
            syms.append("white_coating")
        if c["white_pale"] > 0.03 and t["has_many_spots"]:
            syms.append("white_spots")
        if c["yellow"] > 0.08:
            syms.append("yellow_leaves")
        if c["yellow"] > 0.12 and c["healthy_green"] < 0.3:
            syms.append("yellowing_lower")
        if c["brown"] > 0.06:
            syms.append("brown_spots")
        if c["dark_necrosis"] > 0.05:
            syms.append("dark_lesions")
        if c["dark_necrosis"] > 0.07:
            syms.append("black_lesions")
        if t["has_many_spots"] and t["spot_ratio"] > 0.04:
            syms.append("leaf_spots")
        if t["has_many_spots"] and t["spot_ratio"] > 0.06:
            syms.append("stippling")
        if t["has_webbing"]:
            syms.append("webbing")
        if t["has_dry_edges"]:
            syms.append("dry_edges")
        if health_score < 40:
            syms.append("stunted_growth")
        if health_score < 25:
            syms.append("wilting")
        if c["red_orange"] > 0.04:
            syms.append("rapid_spread")

        return list(set(syms))  # убираем дубли

    # ─── Генерация отладочных изображений ──────────────────────────────────

    def _generate_debug_images(self, img: np.ndarray, hsv: np.ndarray, color: dict) -> dict:
        """Создаёт Base64-PNG для отображения результатов анализа в UI."""

        result = {}

        # 1. Тепловая карта поражений (heatmap)
        disease_mask = np.zeros(img.shape[:2], dtype=np.float32)
        for key in ("yellow", "brown", "dark_necrosis", "white_pale"):
            mask = color.get(key, {}).get("mask")
            if mask is not None:
                disease_mask = np.maximum(disease_mask, mask.astype(np.float32) / 255)

        heatmap = cv2.applyColorMap(
            (disease_mask * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        result["heatmap"] = self._to_base64(overlay)

        # 2. Здоровые зоны (зелёная маска)
        green_mask = color.get("healthy_green", {}).get("mask")
        if green_mask is not None:
            green_vis = np.zeros_like(img)
            green_vis[green_mask > 0] = [0, 200, 60]
            result["healthy_regions"] = self._to_base64(
                cv2.addWeighted(img, 0.5, green_vis, 0.5, 0)
            )

        return result

    @staticmethod
    def _to_base64(img: np.ndarray) -> str:
        """Кодирует OpenCV-изображение в base64 PNG data-URL."""
        _, buf = cv2.imencode(".png", img)
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/png;base64,{b64}"