import spacy

# Загрузка модели SpaCy
nlp = spacy.load("en_core_web_sm")


def extract_keywords(text, top_n=5):
    """
    Извлекает ключевые слова из текста.

    :param text: Текст для анализа.
    :param top_n: Количество ключевых слов для извлечения.
    :return: Список ключевых слов.
    """
    # Обработка текста с помощью SpaCy
    doc = nlp(text)

    # Извлечение ключевых слов на основе частотности и значимости
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]

    # Сортировка ключевых слов по частотности
    freq_dict = {word: keywords.count(word) for word in set(keywords)}
    sorted_keywords = sorted(freq_dict, key=freq_dict.get, reverse=True)

    # Возвращение топ-n ключевых слов
    return sorted_keywords[:top_n]


# Пример использования
# text = "Язык, распознаваемый недетерминированным конечным автоматом (НКА) – это все такие слова, по которым существует хотя бы один путь из стартовой вершины в терминальную."
# keywords = extract_keywords(text, top_n=5)
# print("Ключевые слова:", keywords)
