import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import string

def extract_keywords(text, top_n=5):
    """
    Извлекает ключевые слова из текста с использованием NLTK.

    :param text: Текст для анализа.
    :param top_n: Количество ключевых слов для извлечения.
    :return: Список ключевых слов.
    """
    stop_words = set(stopwords.words('russian'))
    words = word_tokenize(text)

    freq_table = defaultdict(int)
    for word in words:
        word = word.lower()
        if word not in stop_words and word not in string.punctuation:
            freq_table[word] += 1

    keywords = sorted(freq_table, key=freq_table.get, reverse=True)[:top_n]
    return keywords

# text = "Язык, распознаваемый недетерминированным конечным автоматом (НКА) – это все такие слова, по которым существует хотя бы один путь из стартовой вершины в терминальную."
# keywords = extract_keywords(text, top_n=5)
# print("Ключевые слова:", ' '.join(keywords))