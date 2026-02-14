FEATURE_NAMES = [
    "social_energy",
    "alone_time_preference",
    "talkativeness",
    "deep_reflection",
    "group_comfort",
    "party_liking",
    "leadership",
    "risk_taking",
    "public_speaking_comfort",
    "excitement_seeking",
    "adventurousness",
    "reading_habit",
]

QUESTIONS = {
    "social_energy": "Заряжает ли вас общение с людьми?",
    "alone_time_preference": "Вы любите проводить время в одиночестве?",
    "talkativeness": "Вы разговорчивы?",
    "deep_reflection": "Вам свойственны глубокие размышления?",
    "group_comfort": "Комфортно ли вам в группе людей?",
    "party_liking": "Нравятся ли вам вечеринки/мероприятия?",
    "leadership": "Склонны ли вы брать на себя лидерство?",
    "risk_taking": "Готовы ли вы рисковать?",
    "public_speaking_comfort": "Комфортно ли вам выступать публично?",
    "excitement_seeking": "Ищете ли вы новые эмоции/впечатления?",
    "adventurousness": "Любите ли вы приключения?",
    "reading_habit": "Любите ли вы читать?",
}

# 5 вариантов → шкала 0..10
OPTIONS = [
    ("Совсем не про меня", 0.0),
    ("Скорее нет", 2.5),
    ("Нейтрально / иногда", 5.0),
    ("Скорее да", 7.5),
    ("Полностью про меня", 10.0),
]