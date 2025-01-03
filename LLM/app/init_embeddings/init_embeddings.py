import faiss
import yaml
import sys

from sentence_transformers import SentenceTransformer

import app.core.vector_db.questions_preprocessing as question_preprocessor
import app.config.config as config


def create_embeddings(model, data):
    texts = []

    for item in data:
        for question in item["questions"]:
            texts.append(question_preprocessor.prepocess_question(question))

    embeddings = model.encode(texts)

    faiss.normalize_L2(embeddings)

    return embeddings


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Нормализуем векторы перед добавлением в индекс
    faiss.normalize_L2(embeddings)

    index.add(embeddings)

    return index


def init_embeddings(model, data, output_filename) -> None:
    """
    builds embeddings, saves them to index in and saves it to the file
    :param model:
    :param data:
    :param output_filename:
    :return:
    """
    embeddings = create_embeddings(model, data)

    index = create_faiss_index(embeddings)

    faiss.write_index(index, output_filename)


def main():
    cfg = config.SingletonConfig.get_instance()

    with open(sys.argv[1], 'r', encoding='utf-8') as file:
        content = yaml.safe_load(file)

    model = SentenceTransformer(cfg.get_sentence_transformer_name())

    init_embeddings(
        model,
        content,
        sys.argv[2],
    )


if __name__ == "__main__":
    main()
