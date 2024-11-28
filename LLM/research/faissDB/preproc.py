import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3

nltk.download('punkt')
nltk.download('stopwords')

# Инициализация морфологического анализатора
morph = pymorphy3.MorphAnalyzer()

# Загрузка стоп-слов для русского языка
stop_words = stopwords.words('russian')


def normalize_sentence(text):
    """
    Нормализует предложение: приводит к нижнему регистру, удаляет пунктуацию.

    Args:
        text: Строка - предложение.

    Returns:
        Строка - нормализованное предложение.
    """
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации
    return text


def tokenize_sentence(text):
    """
    Токенизирует предложение на слова.

    Args:
        text: Строка - предложение.

    Returns:
        Список слов - токенов.
    """
    tokens = word_tokenize(text, language='russian')
    return tokens


def remove_stopwords(tokens):
    """
    Удаляет стоп-слова из списка токенов.

    Args:
        tokens: Список слов - токенов.

    Returns:
        Список слов - токенов без стоп-слов.
    """
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


def lemmatize_tokens(tokens):
    """
    Лемматизирует список токенов.

    Args:
        tokens: Список слов - токенов.

    Returns:
        Список лемм.
    """
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]
    return lemmas


def preprocess_text(text):
    """
    Выполняет полную предобработку текста: нормализацию, токенизацию, удаление стоп-слов и лемматизацию.

    Args:
        text: Строка - текст.

    Returns:
        Список лемм.
    """
    normalized_text = normalize_sentence(text)
    tokens = tokenize_sentence(normalized_text)
    tokens_without_stopwords = remove_stopwords(tokens)
    lemmas = lemmatize_tokens(tokens_without_stopwords)
    return ' '.join(lemmas)

# # Пример использования:
# text = "Язык, распознаваемый недетерминированным конечным автоматом (НКА) – это все такие слова, по которым существует хотя бы один путь из стартовой вершины в терминальную."
# preprocessed_text = preprocess_text(text)
# print(f"Исходный текст: {text}")
# print(f"Обработанный текст (леммы): {preprocessed_text}")
