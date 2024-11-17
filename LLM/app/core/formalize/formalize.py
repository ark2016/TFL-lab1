import time
import logging
from app.core.mistral.mistral import get_chat_response

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Контексты
context_for_TRS = (
    "Пожалуйста, переформулируйте следующий запрос на проверку завершаемости TRS в формате, который "
    "можно использовать для запроса на составление грамматики TRS для последующего запроса на проверку "
    "этой грамматики TRS-решателем, сделайте запрос максимально понятным и точным."
    "Не решай задачу, просто переформулируй вопрос. Сделай его более понятным и точным. "
    "Верни своим ответом более точный вопрос, не сокращай умножение (не надо a*b = ab, надо a*b = a*a). "
    "например степень x^n или x**n пишется x{n}, а умножение после степени опускается (например x^n * "
    "y правильно "
    "записан только так: x{n}y)."
    "ВАЖНО!!!! степень x^n или x**n пишется x{n}, а умножение после степени опускается (например x^n * "
    "y правильно "
    "записан только так: x{n}y)."
    "Константа ЭТО НЕ ПЕРЕМЕННАЯ"
    "Знак умножения `*` обязательно ставится только между коэффициентом и переменной. Между переменными "
    "знак `*` не ставится."
)

context_TRS = (
    "верни только TRS, основываясь на грамматике и решении выше, не добавляй никаких комментариев и заметок"
    "ответ будет проанализирован, не нужны никакие лишние комментарии и неиспоьзуемые куски текста"
    "используй следующий формат "
    "variables = ([буква],)∗[буква]"
    "([терм] = [терм][eol])+"
    "в variables перечисляй через запятую и пробел только переменные, без внешних скобок и спецсимволов, если "
    "они не входят в название переменной, в variables пиши только НЕ конструкторы"
    "правила пиши на следующей строке, без названия, без внешних скобок и ключевых слов, просто перечисли "
    "правила на каждой новой строке. Используй = вместо стрелок и подобных символов"
    "также добавь интерпретацию через 10 -, проведя черту, также без внешних скобок и объявления, что это "
    "интерпретация"
    "не сокращай умножение (не надо a*b = ab, надо a*b = a*a)."
    "x{n} - это ЕДИНСТВЕННЫЙ ВАРИАНТ СТЕПЕНИ"
    "например степень x^n или x**n пишется x{n}, а умножение после степени опускается (например x^n * y "
    "правильно "
    "записан только так: x{n}y)."
    "ВАЖНО: не заноси конструкторы в переменные, если они не используются в качестве переменных"
    "Константа ЭТО НЕ ПЕРЕМЕННАЯ"
    "Знак умножения `*` обязательно ставится только между коэффициентом и переменной. Между переменными знак"
    "`*` не ставится."
    "   - Далее следует ряд примеров, как ты должна отвечать, в формате:\n"
    "   `Запрос пользователя: ...\n"
    "    Правильный ответ: ...`\n"
    "1. Запрос пользователя: f(x) = x^3 + 3x\n"
    "   Правильный ответ: f(x) = x{3} + 3*x\n"
    "2. Запрос пользователя: f(x) = 7x\n"
    "   Правильный ответ: f(x) = 7*x\n"
    "3. Запрос пользователя: g(x, y) = 91y + 4*x\n"
    "   Правильный ответ: g(x, y) = 91*y + 4*x\n"
    "4. Запрос пользователя: f(x, y) = x*y\n"
    "   Правильный ответ: f(x, y) = xy\n"
    "5. Запрос пользователя: g(x, y) = 4*x*y\n"
    "   Правильный ответ: g(x, y) = 4*xy\n"
    "6. Запрос пользователя: g(x, y) = 2*x*y*x + 5y\n"
    "   Правильный ответ: g(x, y) = 2*xyx + 5*y\n\n"
)


def get_refactored_question(question, context):
    """
    Получает переформулированный вопрос на основе контекста.
    """
    try:
        refactored_question = get_chat_response(question, context=context, model="mistral-small-latest")
        logger.info("Переформулированный вопрос: %s", refactored_question)
        return refactored_question
    except Exception as e:
        logger.error("Ошибка при получении переформулированного вопроса: %s", e)
        raise


def get_answer(question, context):
    """
    Получает ответ на вопрос на основе контекста.
    """
    try:
        answer = get_chat_response(question, context=context, model="mistral-large-latest")
        logger.info("Ответ: %s", answer)
        return answer
    except Exception as e:
        logger.error("Ошибка при получении ответа: %s", e)
        raise


def process_answer(answer):
    """
    Обрабатывает ответ, удаляя лишние символы и добавляя перенос строки.
    """
    try:
        processed_answer = answer.replace("```", "").strip() + "\n"
        return processed_answer
    except Exception as e:
        logger.error("Ошибка при обработке ответа: %s", e)
        raise


def formalize(question, context_for_TRS=context_for_TRS, context_TRS=context_TRS):
    """
    Формализует вопрос и получает ответ.
    """
    refactored_question = get_refactored_question(question, context_for_TRS)
    logger.info("-" * 40)
    time.sleep(1)

    answer = get_answer(refactored_question, context_TRS)
    logger.info("-" * 40)

    processed_answer = process_answer(answer)
    return processed_answer