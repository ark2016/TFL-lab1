import yaml

# Путь к вашему YAML-файлу
file_path = 'data_for_test.yaml'

# Чтение YAML-файла
with open(file_path, 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

# Замена поля 'author' на 'number' и пронумерование вопросов
for index, item in enumerate(data, start=1):
    item['number'] = index
    if 'author' in item:
        del item['author']

# Сохранение измененных данных обратно в YAML-файл
with open(file_path, 'w', encoding='utf-8') as file:
    yaml.safe_dump(data, file, allow_unicode=True)

print(f"Файл {file_path} успешно обновлен.")
