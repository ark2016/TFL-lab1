import yaml

# Путь к вашему YAML-файлу
file_path = '../../../data/data.yaml'
# file_path = 'dataset.yaml'

# Чтение YAML-файла
with open(file_path, 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

# print(data)
# Вывод данных
for item in data:
    question = item['question']
    # answer = item['answer']
    # author = item['author']
    print(question)
    # print(f"Answer: {answer}")
    # print(f"Author: {author}")
    # print("-" * 40)
    # break
