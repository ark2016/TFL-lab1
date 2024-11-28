import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Инициализация лемматизатора
lemmatizer = WordNetLemmatizer()

# Загрузка стоп-слов для английского языка
stop_words = stopwords.words('english')

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
    tokens = word_tokenize(text, language='english')
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
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
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
    # return ' '.join(tokens_without_stopwords)
    lemmas = lemmatize_tokens(tokens_without_stopwords)
    return ' '.join(lemmas)

# # Пример использования:
# text = "The quick brown foxes jumps over the lazy dogs!"
# preprocessed_text = preprocess_text(text)
# print(f"Исходный текст: {text}")
# # print(f"Обработанный текст (леммы): {preprocessed_text}")
# print(f"Обработанный текст: {preprocessed_text}")
