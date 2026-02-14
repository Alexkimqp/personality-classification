import os
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from src.inference import Predictor
from src.telegram_questionnaire import FEATURE_NAMES, QUESTIONS, OPTIONS

predictor = Predictor.load()


def build_answers_keyboard() -> InlineKeyboardMarkup:
    # Префикс ans: — чтобы отличать ответы от других кнопок
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text, callback_data=f"ans:{value}")] for text, value in OPTIONS]
    )


def build_ready_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Да, поехали", callback_data="ready:yes")],
            [InlineKeyboardButton("Нет", callback_data="ready:no")],
        ]
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    context.user_data["state"] = "WAIT_READY"

    full_name = (update.effective_user.full_name or "").strip() or "друг"
    text = f"Привет, {full_name}! Вы готовы ответить на несколько вопросов?"
    await update.effective_chat.send_message(text, reply_markup=build_ready_keyboard())


async def on_ready(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "ready:no":
        context.user_data.clear()
        await update.effective_chat.send_message("Хорошо. Напишите /start, когда будете готовы.")
        return

    # ready:yes
    context.user_data["state"] = "IN_QUIZ"
    context.user_data["i"] = 0
    context.user_data["answers"] = {}
    await ask_current(update, context)


async def ask_current(update: Update, context: ContextTypes.DEFAULT_TYPE):
    i = context.user_data["i"]
    feature = FEATURE_NAMES[i]
    text = f"Вопрос {i+1}/12:\n{QUESTIONS[feature]}"
    await update.effective_chat.send_message(text, reply_markup=build_answers_keyboard())


async def on_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    # ответы принимаем только когда реально в опросе
    if context.user_data.get("state") != "IN_QUIZ":
        await update.effective_chat.send_message("Нажмите /start, чтобы начать опрос.")
        return

    if not (q.data or "").startswith("ans:"):
        return

    i = context.user_data.get("i", 0)
    feature = FEATURE_NAMES[i]

    # ans:7.5 -> 7.5
    value = float(q.data.split("ans:", 1)[1])

    context.user_data.setdefault("answers", {})[feature] = value
    i += 1
    context.user_data["i"] = i

    if i < len(FEATURE_NAMES):
        await ask_current(update, context)
        return

    payload = context.user_data["answers"]
    proba = predictor.predict_proba(payload)
    proba_sorted = sorted(proba.items(), key=lambda x: x[1], reverse=True)

    if not proba_sorted:
        await update.effective_chat.send_message("Не удалось получить результат. Попробуйте /start ещё раз.")
        context.user_data.clear()
        return

    PRIMARY_PHRASE = {
        "Introvert": "интровертная направленность",
        "Extrovert": "экстравертная направленность",
        "Ambivert": "амбивертная направленность",
    }
    SECONDARY_TRAITS = {
        "Introvert": "интровертных черт",
        "Extrovert": "экстравертных черт",
        "Ambivert": "амбивертных черт",
    }
    CLASS_NOUN = {
        "Introvert": "интроверт",
        "Extrovert": "экстраверт",
        "Ambivert": "амбиверт",
    }

    primary_label, p1 = proba_sorted[0]

    # Проверка на "100%" (порог, чтобы не зависеть от погрешности float)
    if p1 >= 0.9995:
        noun = CLASS_NOUN.get(str(primary_label), str(primary_label))
        text = f"По результатам оценки вы на 100% {noun}."
        await update.effective_chat.send_message(text)
        context.user_data.clear()
        return

    # Обычный “человеческий” вывод: топ-2 класса
    if len(proba_sorted) == 1:
        primary_text = PRIMARY_PHRASE.get(str(primary_label), str(primary_label))
        p1_pct = round(p1 * 100)
        text = f"По результатам оценки у вас преобладает {primary_text} ({p1_pct} %)."
        await update.effective_chat.send_message(text)
        context.user_data.clear()
        return

    secondary_label, p2 = proba_sorted[1]

    s = (p1 + p2) or 1.0
    p1_pct = round(p1 / s * 100)
    p2_pct = 100 - p1_pct  # чтобы сумма была ровно 100%

    primary_text = PRIMARY_PHRASE.get(str(primary_label), str(primary_label))
    secondary_text = SECONDARY_TRAITS.get(str(secondary_label), str(secondary_label))

    text = (
        f"По результатам оценки у вас преобладает {primary_text} ({p1_pct} %), "
        f"при наличии выраженных {secondary_text} ({p2_pct} %)."
    )
    await update.effective_chat.send_message(text)

    context.user_data.clear()


def main():
    token = os.environ["BOT_TOKEN"]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(on_ready, pattern=r"^ready:(yes|no)$"))
    app.add_handler(CallbackQueryHandler(on_answer, pattern=r"^ans:"))

    app.run_polling()


if __name__ == "__main__":
    main()