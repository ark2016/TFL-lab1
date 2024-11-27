def generalized_top_k_accuracy(predicted, expected, k, s):
    """
    Вычисляет обобщенную версию Top-k Accuracy.

    :param predicted: Список предсказанных номеров элементов.
    :param expected: Список ожидаемых номеров элементов.
    :param k: Пороговое значение ранга.
    :param s: Параметр "мягкости".
    :return: Значение обобщенной Top-k Accuracy.
    """
    total_weight = 0
    correct_weight = 0

    for exp in expected:
        weight_sum = 0
        correct_sum = 0
        for i, pred in enumerate(predicted[:k]):
            weight = 1 / (i + 1) ** s
            weight_sum += weight
            if pred == exp:
                correct_sum += weight
        total_weight += weight_sum
        correct_weight += correct_sum

    return correct_weight / total_weight if total_weight > 0 else 0

def generalized_jaccard_index(predicted, expected, k, s):
    """
    Вычисляет обобщенную версию Jaccard Index.

    :param predicted: Список предсказанных номеров элементов.
    :param expected: Список ожидаемых номеров элементов.
    :param k: Пороговое значение ранга.
    :param s: Параметр "мягкости".
    :return: Значение обобщенного Jaccard Index.
    """
    union_set = set(predicted[:k]) | set(expected)
    intersection_sum = 0
    union_sum = 0

    for item in union_set:
        pred_rank = predicted[:k].index(item) + 1 if item in predicted[:k] else k + 1
        exp_rank = expected.index(item) + 1 if item in expected else k + 1
        intersection_sum += min(1, (pred_rank / k) ** (-1 / s)) + min(1, (exp_rank / k) ** (-1 / s))
        union_sum += 2

    return (intersection_sum / union_sum) - 1 if union_sum > 0 else 0
