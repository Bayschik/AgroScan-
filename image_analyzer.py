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

        # Проверяем, что на изображении растение
        is_plant, plant_confidence = self._detect_if_plant(img)
        if not is_plant:
            return {
                "error": "На изображении не обнаружено растение. Пожалуйста, загрузите фото растения.",
                "detected_symptoms": [],
                "is_plant": False,
                "plant_confidence": plant_confidence
            }

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
            "is_plant":          True,
            "plant_confidence":  plant_confidence,
            "image_stats": {
                "width":    img.shape[1],
                "height":   img.shape[0],
                "channels": img.shape[2] if len(img.shape) > 2 else 1,
            },
            "debug_images": debug_images,
        }

    # ─── Определение, является ли изображение растением ──────────────────────

    def _detect_if_plant(self, img: np.ndarray) -> tuple:
        """
        Определяет, является ли изображение растением, на основе:
        - Доминирующий зелёный цвет
        - Текстурные характеристики (листья имеют определённую структуру)
        - Наличие зелёных пикселей
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Диапазон зелёного цвета в HSV
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        green_ratio = np.sum(green_mask > 0) / max(img.shape[0] * img.shape[1], 1)
        
        # Анализ текстуры (растения имеют определённую структурность)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Вычисляем вариацию градиента (признак растительной текстуры)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_variation = np.var(grad_magnitude)
        
        # Вычисляем энтропию изображения
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        
        # Определяем, есть ли достаточно зелёного цвета и растительная текстура
        is_plant = green_ratio > 0.15 and grad_variation > 50 and entropy > 5.0
        
        if green_ratio < 0.05:
            is_plant = False
        
        confidence = min(100, int(green_ratio * 100 + min(grad_variation / 10, 30)))
        
        return is_plant, confidence

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
                b64 = source.split(",", 1)[1]
                arr = np.frombuffer(base64.b64decode(b64), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                return None

            if img is not None:
                h, w = img.shape[:2]
                if max(h, w) > 1024:
                    scale = 1024 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
            return img
        except Exception as e:
            print(f"[PlantAnalyzer] Ошибка загрузки: {e}")
            return None

    # ─── Цветовой анализ (с порогами для заболеваний) ─────────────────────

    def _analyze_colors(self, hsv: np.ndarray, total: int) -> dict:
        """
        Вычисляет долю пикселей каждого цветового класса.
        Используются более высокие пороги для выявления реальных заболеваний.
        """
        # Здоровый зелёный - широкий диапазон
        healthy_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        
        # Жёлтый - только выраженное пожелтение (болезненное)
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
        
        # Коричневый - некрозы (только темно-коричневые, не тени)
        brown_mask = cv2.inRange(hsv, np.array([8, 80, 60]), np.array([20, 200, 150]))
        
        # Белый налёт (мучнистая роса) - очень специфичный
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
        
        # Тёмные некрозы
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
        
        # Ржавчина/красные пятна
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Морфологическая обработка для удаления шума
        kernel = np.ones((3, 3), np.uint8)
        
        healthy_pixels = int(np.sum(healthy_mask > 0))
        yellow_pixels = int(np.sum(yellow_mask > 0))
        brown_pixels = int(np.sum(brown_mask > 0))
        white_pixels = int(np.sum(white_mask > 0))
        dark_pixels = int(np.sum(dark_mask > 0))
        red_pixels = int(np.sum(red_mask > 0))
        
        results = {
            "healthy_green": {
                "pixel_count": healthy_pixels,
                "ratio": round(healthy_pixels / max(total, 1), 4),
                "description": "Здоровая зелёная ткань",
                "mask": healthy_mask,
            },
            "yellow": {
                "pixel_count": yellow_pixels,
                "ratio": round(yellow_pixels / max(total, 1), 4),
                "description": "Пожелтение (хлороз, некроз)",
                "mask": yellow_mask,
            },
            "brown": {
                "pixel_count": brown_pixels,
                "ratio": round(brown_pixels / max(total, 1), 4),
                "description": "Коричневые некрозы",
                "mask": brown_mask,
            },
            "white_pale": {
                "pixel_count": white_pixels,
                "ratio": round(white_pixels / max(total, 1), 4),
                "description": "Белый/бледный налёт",
                "mask": white_mask,
            },
            "dark_necrosis": {
                "pixel_count": dark_pixels,
                "ratio": round(dark_pixels / max(total, 1), 4),
                "description": "Тёмные некрозы",
                "mask": dark_mask,
            },
            "red_orange": {
                "pixel_count": red_pixels,
                "ratio": round(red_pixels / max(total, 1), 4),
                "description": "Красно-оранжевые пятна",
                "mask": red_mask,
            },
        }
        
        return results

    # ─── Текстурный анализ (с улучшенной фильтрацией) ─────────────────────

    def _analyze_texture(self, gray: np.ndarray, bgr: np.ndarray) -> dict:
        """Обнаруживает текстурные аномалии с фильтрацией ложных срабатываний."""
        
        total_px = gray.shape[0] * gray.shape[1]
        
        # Используем CLAHE для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Медианный фильтр для удаления шума
        denoised = cv2.medianBlur(enhanced, 5)
        
        # Адаптивный порог с большим размером окна
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 5  # Увеличен размер окна
        )
        
        # Морфологическое открытие и закрытие
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Поиск контуров
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Фильтрация пятен по размеру и форме
        spots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Пятна должны быть заметного размера (не шум)
            if 50 < area < total_px * 0.15:
                x, y, w, h = cv2.boundingRect(cnt)
                circularity = self._circularity(cnt)
                # Отфильтровываем слишком круглые пятна (могут быть естественными)
                if circularity < 0.85:  # Не слишком круглые
                    spots.append({
                        "area": int(area),
                        "x": x, "y": y,
                        "w": w, "h": h,
                        "circularity": circularity,
                    })
        
        spot_area = sum(s["area"] for s in spots)
        spot_ratio = spot_area / max(total_px, 1)
        
        # Edge analysis - только значительные сухие края
        edges = cv2.Canny(gray, 60, 180)
        # Удаляем границы изображения
        h, w = edges.shape
        edges[0:int(h*0.05), :] = 0
        edges[int(h*0.95):h, :] = 0
        edges[:, 0:int(w*0.05)] = 0
        edges[:, int(w*0.95):w] = 0
        
        edge_density = float(np.sum(edges > 0)) / max(total_px, 1)
        
        # Webbing detection - только явная паутина
        web_score = 0.0
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=20, maxLineGap=8)
        if lines is not None and len(lines) > 10:
            lengths = [np.hypot(l[0][2] - l[0][0], l[0][3] - l[0][1]) for l in lines]
            avg_len = float(np.mean(lengths)) if lengths else 0
            if avg_len < 50 and len(lines) > 20:
                web_score = min(len(lines) / 150, 0.8)
        
        return {
            "spot_count": len(spots),
            "spot_ratio": round(spot_ratio, 4),
            "spots": spots[:20],
            "edge_density": round(edge_density, 4),
            "web_score": round(web_score, 4),
            "has_many_spots": len(spots) > 25,  # Более высокий порог
            "has_webbing": web_score > 0.4,
            "has_dry_edges": edge_density > 0.12,  # Более высокий порог
        }

    @staticmethod
    def _circularity(cnt) -> float:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            return 0.0
        return round(4 * np.pi * area / (perim ** 2), 3)

    # ─── Скор здоровья (более строгий) ─────────────────────────────────────

    def _calculate_health_score(self, color: dict, texture: dict) -> int:
        """
        Возвращает скор здоровья 0-100.
        Здоровое растение получает >85 только при отсутствии симптомов.
        """
        green_ratio = color.get("healthy_green", {}).get("ratio", 0)
        yellow_ratio = color.get("yellow", {}).get("ratio", 0)
        brown_ratio = color.get("brown", {}).get("ratio", 0)
        dark_ratio = color.get("dark_necrosis", {}).get("ratio", 0)
        white_ratio = color.get("white_pale", {}).get("ratio", 0)
        red_ratio = color.get("red_orange", {}).get("ratio", 0)
        spot_ratio = texture.get("spot_ratio", 0)
        web_score = texture.get("web_score", 0)
        
        # Начинаем с 100
        score = 100.0
        
        # Штрафы за патологические признаки (с более высокими порогами)
        if yellow_ratio > 0.12:
            score -= min(yellow_ratio * 150, 60)
        elif yellow_ratio > 0.08:
            score -= min(yellow_ratio * 100, 30)
        
        if brown_ratio > 0.08:
            score -= min(brown_ratio * 200, 70)
        elif brown_ratio > 0.05:
            score -= min(brown_ratio * 150, 30)
        
        if dark_ratio > 0.06:
            score -= min(dark_ratio * 250, 80)
        
        # Белый налёт - только при значительном количестве
        if white_ratio > 0.08:
            score -= min(white_ratio * 150, 50)
        
        if red_ratio > 0.06:
            score -= min(red_ratio * 200, 60)
        
        if spot_ratio > 0.08:
            score -= min(spot_ratio * 200, 70)
        elif spot_ratio > 0.04:
            score -= min(spot_ratio * 120, 30)
        
        if web_score > 0.4:
            score -= web_score * 50
        
        # Бонус за здоровую зелень
        if green_ratio > 0.4:
            score += min(green_ratio * 20, 15)
        
        # Если нет признаков болезней, скор остается высоким
        has_disease_signs = (yellow_ratio > 0.05 or brown_ratio > 0.03 or 
                            dark_ratio > 0.02 or white_ratio > 0.03 or 
                            spot_ratio > 0.02 or web_score > 0.2)
        
        if not has_disease_signs and green_ratio > 0.3:
            score = min(score, 95)
        
        score = max(0, min(100, int(score)))
        return score

    # ─── Маппинг на симптомы (более строгие пороги) ───────────────────────

    def _map_to_symptoms(self, color: dict, texture: dict, health_score: int) -> list:
        """Конвертирует числовые метрики в список симптомов с высокими порогами."""
        
        syms = []
        
        c = {k: v.get("ratio", 0) for k, v in color.items()}
        t = texture
        
        # Белый налёт - только если его много
        if c["white_pale"] > 0.10:
            syms.append("white_coating")
        
        if c["white_pale"] > 0.06 and t["has_many_spots"]:
            syms.append("white_spots")
        
        # Пожелтение - только выраженное
        if c["yellow"] > 0.12:
            syms.append("yellow_leaves")
        
        if c["yellow"] > 0.18 and c["healthy_green"] < 0.25:
            syms.append("yellowing_lower")
        
        # Коричневые пятна
        if c["brown"] > 0.08:
            syms.append("brown_spots")
        
        if c["dark_necrosis"] > 0.08:
            syms.append("dark_lesions")
        
        if c["dark_necrosis"] > 0.12:
            syms.append("black_lesions")
        
        # Пятна на листьях - только если их много
        if t["has_many_spots"] and t["spot_ratio"] > 0.06:
            syms.append("leaf_spots")
        
        if t["has_many_spots"] and t["spot_ratio"] > 0.10:
            syms.append("stippling")
        
        # Паутина
        if t["has_webbing"]:
            syms.append("webbing")
        
        # Сухие края
        if t["has_dry_edges"]:
            syms.append("dry_edges")
        
        # Системные симптомы
        if health_score < 45:
            syms.append("stunted_growth")
        
        if health_score < 30:
            syms.append("wilting")
        
        if c["red_orange"] > 0.08:
            syms.append("rapid_spread")
        
        return list(set(syms))

    # ─── Генерация отладочных изображений ──────────────────────────────────

    def _generate_debug_images(self, img: np.ndarray, hsv: np.ndarray, color: dict) -> dict:
        """Создаёт Base64-PNG для отображения результатов анализа в UI."""

        result = {}

        # Тепловая карта поражений
        disease_mask = np.zeros(img.shape[:2], dtype=np.float32)
        for key in ("yellow", "brown", "dark_necrosis", "white_pale"):
            mask = color.get(key, {}).get("mask")
            if mask is not None:
                disease_mask = np.maximum(disease_mask, mask.astype(np.float32) / 255)

        if np.max(disease_mask) > 0:
            heatmap = cv2.applyColorMap(
                (disease_mask * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            result["heatmap"] = self._to_base64(overlay)

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