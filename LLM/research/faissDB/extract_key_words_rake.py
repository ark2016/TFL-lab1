from rake_nltk import Rake
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
def extract_keywords(text, top_n=5):
    """
    Извлекает ключевые слова из текста с использованием RAKE.

    :param text: Текст для анализа.
    :param top_n: Количество ключевых слов для извлечения.
    :return: Список ключевых слов.
    """
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()[:top_n]
    return keywords

# text = "Язык, распознаваемый недетерминированным конечным автоматом (НКА) – это все такие слова, по которым существует хотя бы один путь из стартовой вершины в терминальную."
# keywords = extract_keywords(text, top_n=5)
# print("Ключевые слова:", keywords)