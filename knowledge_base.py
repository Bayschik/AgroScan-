"""
Экспертная система диагностики заболеваний растений
База знаний: симптомы → диагнозы → рекомендации
"""

DISEASES = {
    "powdery_mildew": {
        "name": "Мучнистая роса",
        "name_en": "Powdery Mildew",
        "severity": "medium",
        "description": "Грибковое заболевание, вызывающее белый мучнистый налёт на листьях и стеблях.",
        "symptoms": [
            "white_coating",       # белый налёт
            "yellow_leaves",       # пожелтение
            "curling_leaves",      # скручивание листьев
            "leaf_spots",          # пятна на листьях
            "stunted_growth",      # задержка роста
        ],
        "opencv_features": {
            "color_range_hsv": {"lower": [0, 0, 200], "upper": [180, 30, 255]},  # белый/светлый
            "texture": "powdery",
            "coverage_threshold": 0.05,
        },
        "recommendations": [
            "Обработать фунгицидом на основе серы или меди",
            "Улучшить вентиляцию и снизить влажность воздуха",
            "Удалить и уничтожить поражённые листья",
            "Избегать полива сверху — лить под корень",
            "Обработать раствором пищевой соды (1 ч.л. на 1 л воды)",
        ],
        "prevention": [
            "Соблюдать расстояние между растениями для проветривания",
            "Не допускать избыточного азотного удобрения",
            "Проводить профилактические обработки фунгицидом",
        ],
    },

    "leaf_blight": {
        "name": "Ожог листьев (Блайт)",
        "name_en": "Leaf Blight",
        "severity": "high",
        "description": "Бактериальное или грибковое поражение, вызывающее быстрое отмирание тканей листа.",
        "symptoms": [
            "brown_spots",         # коричневые пятна
            "wilting",             # увядание
            "leaf_spots",          # пятна
            "yellow_leaves",       # пожелтение
            "black_lesions",       # чёрные очаги
            "rapid_spread",        # быстрое распространение
        ],
        "opencv_features": {
            "color_range_hsv": {"lower": [10, 100, 50], "upper": [30, 255, 200]},  # коричневый
            "texture": "irregular_spots",
            "coverage_threshold": 0.08,
        },
        "recommendations": [
            "Немедленно удалить все поражённые листья и части растения",
            "Обработать фунгицидом / бактерицидом на основе меди",
            "Дезинфицировать инструменты после обрезки",
            "Уменьшить полив и не допускать застоя воды",
            "При тяжёлом поражении — удалить растение целиком",
        ],
        "prevention": [
            "Не поливать в вечернее время",
            "Обеспечить хороший дренаж почвы",
            "Применять профилактические фунгициды в сезон дождей",
        ],
    },

    "root_rot": {
        "name": "Корневая гниль",
        "name_en": "Root Rot",
        "severity": "high",
        "description": "Грибковое поражение корневой системы из-за переувлажнения или плохого дренажа.",
        "symptoms": [
            "wilting",             # увядание
            "yellowing_lower",     # желтеют нижние листья
            "yellow_leaves",       # пожелтение
            "stunted_growth",      # задержка роста
            "soft_stem",           # мягкий стебель
            "dark_roots",          # тёмные корни
            "musty_smell",         # запах сырости
        ],
        "opencv_features": {
            "color_range_hsv": {"lower": [35, 50, 30], "upper": [80, 200, 120]},  # тёмно-зелёный/коричневый
            "texture": "necrotic",
            "coverage_threshold": 0.10,
        },
        "recommendations": [
            "Прекратить полив на 7-10 дней",
            "Пересадить в свежий субстрат с хорошим дренажом",
            "Обрезать поражённые корни до здоровой ткани",
            "Обработать корни фунгицидом (Фитоспорин, Триходермин)",
            "Обеспечить дренажные отверстия в горшке",
        ],
        "prevention": [
            "Поливать только при подсыхании верхнего слоя почвы",
            "Использовать лёгкий дренирующий субстрат",
            "Добавлять дренажный слой из керамзита",
        ],
    },

    "chlorosis": {
        "name": "Хлороз",
        "name_en": "Chlorosis",
        "severity": "low",
        "description": "Нехватка питательных веществ (железо, магний), вызывающая пожелтение листьев.",
        "symptoms": [
            "yellow_leaves",       # пожелтение
            "pale_color",          # бледность
            "green_veins",         # жилки остаются зелёными
            "stunted_growth",      # задержка роста
            "leaf_drop",           # листопад
        ],
        "opencv_features": {
            "color_range_hsv": {"lower": [25, 80, 150], "upper": [45, 255, 255]},  # жёлтый
            "texture": "uniform_yellowing",
            "coverage_threshold": 0.15,
        },
        "recommendations": [
            "Внести хелат железа или сернокислое железо в почву",
            "Добавить сульфат магния (1 г на 1 л воды) при поливе",
            "Проверить pH почвы — оптимум 6.0–6.5",
            "Провести внекорневую подкормку микроэлементами",
            "При защелачивании — добавить торф или лимонную кислоту",
        ],
        "prevention": [
            "Регулярно проводить комплексные подкормки",
            "Контролировать кислотность почвы",
            "Использовать воду для полива без хлора",
        ],
    },

    "spider_mite": {
        "name": "Паутинный клещ",
        "name_en": "Spider Mite",
        "severity": "medium",
        "description": "Мелкий паразит, высасывающий сок из клеток листа; оставляет белёсые точки и паутину.",
        "symptoms": [
            "white_spots",         # белые точки
            "webbing",             # паутинка
            "yellow_leaves",       # пожелтение
            "curling_leaves",      # скручивание
            "stippling",           # крапчатость
            "dry_edges",           # сухие края
        ],
        "opencv_features": {
            "color_range_hsv": {"lower": [0, 0, 180], "upper": [30, 60, 255]},  # белёсые точки
            "texture": "stippled",
            "coverage_threshold": 0.05,
        },
        "recommendations": [
            "Обработать акарицидом (Актофит, Фитоверм, Акарин)",
            "Протереть листья мыльным раствором с двух сторон",
            "Повысить влажность воздуха — клещи не любят сырость",
            "Провести 3-4 обработки с интервалом 5-7 дней",
            "Убрать поражённые части растения",
        ],
        "prevention": [
            "Регулярно опрыскивать листья водой",
            "Поддерживать влажность воздуха выше 60%",
            "Периодически осматривать обратную сторону листьев",
        ],
    },

    "bacterial_canker": {
        "name": "Бактериальный рак",
        "name_en": "Bacterial Canker",
        "severity": "critical",
        "description": "Тяжёлое бактериальное заболевание, поражающее ствол и ветви с образованием язв.",
        "symptoms": [
            "dark_lesions",        # тёмные язвы
            "oozing_sap",          # камедетечение
            "wilting",             # увядание
            "bark_cracking",       # трещины коры
            "black_lesions",       # чёрные участки
            "dieback",             # усыхание ветвей
        ],
        "opencv_features": {
            "color_range_hsv": {"lower": [0, 50, 20], "upper": [20, 200, 100]},  # тёмный
            "texture": "necrotic_patches",
            "coverage_threshold": 0.03,
        },
        "recommendations": [
            "Вырезать поражённые ткани до здоровой древесины",
            "Все срезы обработать садовым варом или медным купоросом",
            "Провести опрыскивание бордоской жидкостью",
            "Уничтожить срезанный материал — не компостировать",
            "При обширном поражении удалить растение целиком",
        ],
        "prevention": [
            "Не допускать механических повреждений коры",
            "Проводить профилактические обработки медьсодержащими препаратами",
            "Приобретать посадочный материал только в проверенных питомниках",
        ],
    },

    "healthy": {
        "name": "Растение здорово",
        "name_en": "Healthy Plant",
        "severity": "none",
        "description": "Видимых признаков заболеваний не обнаружено. Растение выглядит здоровым.",
        "symptoms": [],
        "opencv_features": {
            "color_range_hsv": {"lower": [35, 60, 60], "upper": [85, 255, 255]},  # зелёный
            "texture": "uniform",
            "coverage_threshold": 0.0,
        },
        "recommendations": [
            "Продолжайте соблюдать режим полива",
            "Проводите регулярные подкормки",
            "Периодически осматривайте растение на наличие вредителей",
        ],
        "prevention": [],
    },
}

# Весовая матрица: симптом → {болезнь: вес}
SYMPTOM_WEIGHTS = {
    "white_coating":    {"powdery_mildew": 0.9, "spider_mite": 0.2},
    "white_spots":      {"spider_mite": 0.8, "powdery_mildew": 0.3},
    "yellow_leaves":    {"chlorosis": 0.7, "root_rot": 0.5, "spider_mite": 0.4, "powdery_mildew": 0.3, "leaf_blight": 0.4},
    "brown_spots":      {"leaf_blight": 0.9, "bacterial_canker": 0.4},
    "wilting":          {"root_rot": 0.8, "leaf_blight": 0.6, "bacterial_canker": 0.5},
    "curling_leaves":   {"spider_mite": 0.7, "powdery_mildew": 0.4},
    "leaf_spots":       {"leaf_blight": 0.8, "powdery_mildew": 0.5},
    "stunted_growth":   {"root_rot": 0.6, "chlorosis": 0.7, "powdery_mildew": 0.3},
    "soft_stem":        {"root_rot": 0.9},
    "dark_roots":       {"root_rot": 0.95},
    "musty_smell":      {"root_rot": 0.85},
    "pale_color":       {"chlorosis": 0.8},
    "green_veins":      {"chlorosis": 0.9},
    "leaf_drop":        {"chlorosis": 0.5, "root_rot": 0.4},
    "webbing":          {"spider_mite": 0.95},
    "stippling":        {"spider_mite": 0.85},
    "dry_edges":        {"spider_mite": 0.6, "chlorosis": 0.3},
    "dark_lesions":     {"bacterial_canker": 0.9, "leaf_blight": 0.5},
    "black_lesions":    {"bacterial_canker": 0.85, "leaf_blight": 0.6},
    "bark_cracking":    {"bacterial_canker": 0.9},
    "oozing_sap":       {"bacterial_canker": 0.85},
    "dieback":          {"bacterial_canker": 0.7, "root_rot": 0.4},
    "rapid_spread":     {"leaf_blight": 0.7, "bacterial_canker": 0.5},
    "yellowing_lower":  {"root_rot": 0.8, "chlorosis": 0.4},
}

# Все симптомы с описаниями для UI
SYMPTOM_LIST = {
    "white_coating":   {"label": "Белый мучнистый налёт", "icon": "❄"},
    "white_spots":     {"label": "Белые точки/пятна", "icon": "⚪"},
    "yellow_leaves":   {"label": "Пожелтение листьев", "icon": "🟡"},
    "brown_spots":     {"label": "Коричневые пятна", "icon": "🟤"},
    "wilting":         {"label": "Увядание/поникание", "icon": "🌿"},
    "curling_leaves":  {"label": "Скручивание листьев", "icon": "🌀"},
    "leaf_spots":      {"label": "Пятна на листьях", "icon": "🔵"},
    "stunted_growth":  {"label": "Замедленный рост", "icon": "📉"},
    "soft_stem":       {"label": "Мягкий/гнилой стебель", "icon": "💧"},
    "dark_roots":      {"label": "Тёмные/чёрные корни", "icon": "⬛"},
    "musty_smell":     {"label": "Запах сырости/гнили", "icon": "💨"},
    "pale_color":      {"label": "Бледная окраска", "icon": "⬜"},
    "green_veins":     {"label": "Жилки остаются зелёными", "icon": "🌱"},
    "leaf_drop":       {"label": "Опадание листьев", "icon": "🍂"},
    "webbing":         {"label": "Тонкая паутинка", "icon": "🕸"},
    "stippling":       {"label": "Мелкая крапчатость", "icon": "🔸"},
    "dry_edges":       {"label": "Сухие края листьев", "icon": "🔥"},
    "dark_lesions":    {"label": "Тёмные язвы/некрозы", "icon": "🔲"},
    "black_lesions":   {"label": "Чёрные очаги поражения", "icon": "⚫"},
    "bark_cracking":   {"label": "Трещины коры/стебля", "icon": "🪵"},
    "oozing_sap":      {"label": "Выделение сока/камеди", "icon": "💦"},
    "dieback":         {"label": "Усыхание ветвей/побегов", "icon": "🌵"},
    "rapid_spread":    {"label": "Быстрое распространение", "icon": "⚡"},
    "yellowing_lower": {"label": "Желтеют нижние листья", "icon": "🔽"},
}
