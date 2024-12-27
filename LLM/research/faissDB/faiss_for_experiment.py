import os
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
import faiss
import numpy as np
import pickle


def initialize_model():
    # Загрузка конфигурации с обработкой отсутствующего атрибута
    config = AutoConfig.from_pretrained('ai-sage/Giga-Embeddings-instruct', trust_remote_code=True)

    if not hasattr(config.latent_attention_config, '_attn_implementation_internal'):
        config.latent_attention_config._attn_implementation_internal = None

    # Загрузка модели с исправленной конфигурацией
    model = AutoModel.from_pretrained('ai-sage/Giga-Embeddings-instruct', config=config, trust_remote_code=True)

    # Загрузка токенизатора, если требуется
    tokenizer = AutoTokenizer.from_pretrained('ai-sage/Giga-Embeddings-instruct', trust_remote_code=True)

    return model, tokenizer


def create_embeddings(model, tokenizer, data):
    # Предполагается, что каждый элемент в data имеет ключи "question" и "answer"
    task_name_to_instruct = {
        "example": "Given a question, retrieve passages that answer the question",
    }

    query_prefix = task_name_to_instruct["example"] + "\nquestion: "
    passage_prefix = ""  # Нет инструкции для пассажа

    queries = [item["question"] for item in data]
    passages = [item["answer"] for item in data]

    # Получение эмбеддингов
    query_embeddings = model.encode(queries, instruction=query_prefix)
    passage_embeddings = model.encode(passages, instruction=passage_prefix)

    # Объединение эмбеддингов вопросов и ответов
    embeddings = np.concatenate((query_embeddings, passage_embeddings), axis=0)

    # Нормализация эмбеддингов
    embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1).numpy()

    return embeddings


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def search_similar(model, tokenizer, index, query, data, task_instruct, k_max=2, similarity_threshold=0.69):
    query_prefix = task_instruct + "\nquestion: "
    query_instruct = query_prefix + query
    query_embedding = model.encode([query], instruction=query_prefix)

    # Нормализация эмбеддингов
    query_embedding = F.normalize(torch.tensor(query_embedding), p=2, dim=1).numpy()

    # Поиск в FAISS
    D, I = index.search(query_embedding, k_max)

    dynamic_k = 1
    for i in range(1, k_max):
        if D[0][i] < similarity_threshold:
            break
        dynamic_k += 1

    similar_items = []
    for idx, distance in zip(I[0][:dynamic_k], D[0][:dynamic_k]):
        similar_items.append({"distance": distance, "item": data[idx]})
        print(f"Distance: {distance}, Question: {data[idx]['question']}")

    return [item["item"] for item in similar_items]


def save_vectorized_data(data, embeddings, index, filename):
    with open(f"{filename}_data.pkl", "wb") as f:
        pickle.dump(data, f)
    np.save(f"{filename}_embeddings.npy", embeddings)
    faiss.write_index(index, f"{filename}_index.faiss")


def load_vectorized_data(filename):
    with open(f"{filename}_data.pkl", "rb") as f:
        data = pickle.load(f)
    embeddings = np.load(f"{filename}_embeddings.npy")
    index = faiss.read_index(f"{filename}_index.faiss")
    return data, embeddings, index


def process_questions(questions_list, use_saved=False, filename="vectorized_data"):
    model, tokenizer = initialize_model()

    if use_saved and os.path.exists(f"{filename}_data.pkl"):
        data, embeddings, index = load_vectorized_data(filename)
    else:
        data = questions_list
        embeddings = create_embeddings(model, tokenizer, data)
        index = create_faiss_index(embeddings)
        save_vectorized_data(data, embeddings, index, filename)

    # Определение инструкции для поиска
    task_name_to_instruct = {
        "example": "Given a question, retrieve passages that answer the question",
    }
    task_instruct = task_name_to_instruct["example"]

    results = []
    for item in questions_list:
        query = item["question"]
        similar_objects = search_similar(model, tokenizer, index, query, data, task_instruct)
        result_str = f"Запрос: {query}\nПохожие объекты:\n"
        for obj in similar_objects:
            result_str += f"Вопрос: {obj['question']}, Ответ: {obj['answer']}\n"
        results.append(result_str)

    return results


def add_new_questions(new_questions, filename="vectorized_data"):
    model, tokenizer = initialize_model()

    if os.path.exists(f"{filename}_data.pkl"):
        data, embeddings, index = load_vectorized_data(filename)
    else:
        raise FileNotFoundError(f"No existing data found at {filename}. Please initialize the database first.")

    new_embeddings = create_embeddings(model, tokenizer, new_questions)

    updated_data = data + new_questions
    updated_embeddings = np.vstack((embeddings, new_embeddings))

    index.add(new_embeddings)

    save_vectorized_data(updated_data, updated_embeddings, index, filename)

    print(f"Added {len(new_questions)} new questions to the database.")
