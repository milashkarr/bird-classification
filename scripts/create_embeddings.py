import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import sys

# Добавляем chromadb в путь на случай если не установлен
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    print("ChromaDB не установлен. Устанавливаем...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb"])
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True


def create_embeddings_model():
    """Создаем модель для извлечения эмбеддингов"""
    print("Загрузка модели для эмбеддингов...")

    # Используем MobileNetV2 как базовую модель
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Удаляем последний слой (классификатор)
    # MobileNetV2.classifier это Sequential с Linear слоем
    model.classifier = nn.Sequential(
        *list(model.classifier.children())[:-1]  # Оставляем только эмбеддинг слой
    )

    model.eval()
    return model


def extract_embedding(model, image_path, transform):
    """Извлекаем эмбеддинг из одного изображения"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = model(image_tensor)

        return embedding.squeeze().numpy()
    except Exception as e:
        print(f"  Ошибка обработки {image_path}: {e}")
        return None


def create_vector_database():
    """Создаем векторную БД с эмбеддингами"""
    print("=" * 50)
    print("СОЗДАНИЕ ВЕКТОРНОЙ БАЗЫ ДАННЫХ")
    print("=" * 50)

    # Загрузка данных
    print("Загрузка CSV файлов...")
    try:
        train_df = pd.read_csv("data/processed/train.csv")
        val_df = pd.read_csv("data/processed/val.csv")
        test_df = pd.read_csv("data/processed/test.csv")
        print(f"  Train: {len(train_df)} изображений")
        print(f"  Val: {len(val_df)} изображений")
        print(f"  Test: {len(test_df)} изображений")

    except Exception as e:
        print(f"Ошибка загрузки CSV файлов: {e}")
        return False

    # Объединяем все данные с уникальными индексами
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"Всего изображений в датасете: {len(all_data)}")

    # Определяем правильное название колонки с путем
    path_column = None
    for col in ['local_path', 'image_path', 'path', 'filepath']:
        if col in all_data.columns:
            path_column = col
            break

    if not path_column:
        print("ОШИБКА: Не найдена колонка с путями к изображениям!")
        print(f"Доступные колонки: {all_data.columns.tolist()}")
        return False

    print(f"Используем колонку: '{path_column}' для путей к изображениям")

    # Трансформации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Модель для эмбеддингов
    model = create_embeddings_model()

    # Создаем ChromaDB
    print("\nИнициализация векторной БД...")
    try:
        # Удаляем старую папку если есть
        if Path("./chroma_db").exists():
            import shutil
            shutil.rmtree("./chroma_db", ignore_errors=True)
            print("  Удалена старая папка chroma_db")

        chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )

        # Удаляем старую коллекцию если есть
        try:
            chroma_client.delete_collection(name="bird_images")
            print("  Старая коллекция удалена")
        except:
            print("  Нет старой коллекции")

        collection = chroma_client.create_collection(name="bird_images")
        print("  Новая коллекция создана")

    except Exception as e:
        print(f"Ошибка инициализации ChromaDB: {e}")
        return False

    # Извлекаем эмбеддинги
    print("\nИзвлечение эмбеддингов из изображений...")
    embeddings = []
    metadatas = []
    ids = []

    successful = 0
    failed = 0

    for idx, row in all_data.iterrows():
        # Получаем путь из правильной колонки
        img_path = row[path_column]

        # Проверяем несколько возможных путей
        possible_paths = [
            img_path,
            f"data/raw/{row['class_name']}/{Path(img_path).name}",
            f"../data/raw/{row['class_name']}/{Path(img_path).name}",
            img_path.replace('/', '\\'),
            f"data\\raw\\{row['class_name']}\\{Path(img_path).name}",
        ]

        found_path = None
        for path in possible_paths:
            if Path(path).exists():
                found_path = path
                break

        if found_path:
            embedding = extract_embedding(model, found_path, transform)
            if embedding is not None:
                embeddings.append(embedding.tolist())
                metadatas.append({
                    "class_name": row['class_name'],
                    "filename": Path(found_path).name,
                    "path": str(found_path),
                    "label": int(row['label']) if 'label' in row else 0
                })
                # Уникальный ID - используем имя файла без расширения
                file_id = Path(found_path).stem
                ids.append(f"{row['class_name']}_{file_id}")
                successful += 1
            else:
                failed += 1
        else:
            print(f"  Файл не найден: {img_path}")
            failed += 1

        # Прогресс
        if (idx + 1) % 50 == 0:
            print(f"  Обработано: {idx + 1}/{len(all_data)}")

    print(f"\nУспешно обработано: {successful}")
    print(f"Не удалось обработать: {failed}")

    # Проверяем уникальность ID перед добавлением
    if len(set(ids)) != len(ids):
        print(f"\nПРЕДУПРЕЖДЕНИЕ: Найдены дублирующиеся ID!")
        print(f"Уникальных ID: {len(set(ids))}")
        print(f"Всего ID: {len(ids)}")

        # Создаем уникальные ID с суффиксом
        unique_ids = []
        id_count = {}
        for i, id_ in enumerate(ids):
            if id_ in id_count:
                id_count[id_] += 1
                new_id = f"{id_}_{id_count[id_]}"
            else:
                id_count[id_] = 0
                new_id = id_
            unique_ids.append(new_id)

        ids = unique_ids
        print("Созданы уникальные ID с суффиксами")

    # Добавляем в БД
    if embeddings:
        print(f"\nДобавление {len(embeddings)} эмбеддингов в БД...")
        try:
            collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print("  Эмбеддинги успешно добавлены")

            # Сохраняем информацию о коллекции
            collection_info = {
                "total_images": len(embeddings),
                "classes": list(all_data['class_name'].unique()),
                "successful": successful,
                "failed": failed,
                "collection_name": "bird_images",
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                "path_column_used": path_column
            }

            with open("chroma_db_info.json", "w", encoding="utf-8") as f:
                json.dump(collection_info, f, ensure_ascii=False, indent=2)

            print(f"\nВекторная БД создана!")
            print(f"  Папка: chroma_db/")
            print(f"  Коллекция: bird_images")
            print(f"  Размерность эмбеддингов: {len(embeddings[0]) if embeddings else 0}")

            return True

        except Exception as e:
            print(f"Ошибка добавления эмбеддингов в БД: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("Не удалось извлечь ни одного эмбеддинга")
        return False


def test_vector_search():
    """Тестирование поиска в векторной БД"""
    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ ПОИСКА")
    print("=" * 50)

    try:
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        collection = chroma_client.get_collection(name="bird_images")

        # Берем первый эмбеддинг для теста
        test_result = collection.peek()

        # Проверяем есть ли эмбеддинги
        if test_result and test_result.get('embeddings') and len(test_result['embeddings']) > 0:
            results = collection.query(
                query_embeddings=[test_result['embeddings'][0]],
                n_results=3
            )

            print("Тест поиска успешен!")
            print(f"Найдено похожих: {len(results['ids'][0])}")
            print(f"Пример найденного: {results['ids'][0][0]}")
            print(f"Расстояние: {results['distances'][0][0]:.4f}")

            return True
        else:
            print("Нет эмбеддингов в коллекции")
            return False
    except Exception as e:
        print(f"Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Устанавливаем текущую директорию
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Создаем БД
    success = create_vector_database()

    if success:
        # Тестируем
        test_success = test_vector_search()
        print("\n" + "=" * 50)
        if test_success:
            print("ВЕКТОРНАЯ БАЗА ДАННЫХ УСПЕШНО СОЗДАНА И ПРОТЕСТИРОВАНА!")
        else:
            print("ВЕКТОРНАЯ БАЗА СОЗДАНА, НО ТЕСТИРОВАНИЕ НЕ УДАЛОСЬ")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("ОШИБКА СОЗДАНИЯ ВЕКТОРНОЙ БАЗЫ ДАННЫХ")
        print("=" * 50)
        sys.exit(1)
