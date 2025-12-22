import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
import numpy as np


class MinimalBirdClassifier:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Классификатор птиц")
        self.root.geometry("900x650")
        self.root.resizable(True, True)

        self.center_window()

        self.class_names = ['Зяблик', 'Пеночка', 'Щегол', 'Поползень', 'Ласточка']
        self.class_name_mapping = {
            'Зяблик': 'brambling',
            'Пеночка': 'chiffchaff',
            'Щегол': 'goldfinch',
            'Поползень': 'nuthatch',
            'Ласточка': 'swallow'
        }
        self.model = self.load_model()

        self.embedding_model = self.create_embeddings_model()

        self.chroma_client = None
        self.collection = None
        self.connect_to_vector_db()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.current_image = None
        self.photo = None

        self.setup_ui()

    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def create_embeddings_model(self):
        print("Загрузка модели для эмбеддингов...")

        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 5)
        )

        checkpoint = torch.load("models/best_model.pth", map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Загружена наша модель ({checkpoint.get('val_acc', 'N/A')}%)")
        else:
            model.load_state_dict(checkpoint)

        model.fc = nn.Identity()

        model.eval()
        return model

    def load_model(self):
        model = models.resnet18()

        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 5)  # 5 классов
        )

        try:
            if Path("models/best_model.pth").exists():
                checkpoint = torch.load("models/best_model.pth",
                                        map_location='cpu',
                                        weights_only=False)

                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Загружена модель ResNet18 с точностью {checkpoint.get('val_acc', 'N/A')}%")
                else:
                    model.load_state_dict(checkpoint)
                    print("Загружена модель ResNet18")
        except Exception as e:
            print(f"ОШИБКА ЗАГРУЗКИ МОДЕЛИ: {e}")

        model.eval()
        return model

    def connect_to_vector_db(self):
        try:
            if Path("./chroma_db").exists():
                self.chroma_client = chromadb.PersistentClient(
                    path="./chroma_db",
                    settings=Settings(anonymized_telemetry=False)
                )
                self.collection = self.chroma_client.get_collection(name="bird_images")
                print("Векторная БД подключена успешно")
            else:
                print("Векторная БД не найдена")
                self.collection = None
        except Exception as e:
            print(f"Ошибка подключения к векторной БД: {e}")
            self.collection = None

    def extract_embedding(self, image):
        img_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = self.embedding_model(img_tensor)

        return embedding.squeeze().numpy().tolist()

    def search_similar_images(self, embedding, n_results=10):
        if self.collection is None:
            print("Векторная БД не подключена")
            return []

        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=50
            )

            with torch.no_grad():
                img_tensor = self.transform(self.current_image).unsqueeze(0)
                outputs = self.model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                prob, predicted_idx = torch.max(outputs, 1)
                predicted_class_ru = self.class_names[predicted_idx.item()]

                predicted_class_en = self.class_name_mapping.get(predicted_class_ru, predicted_class_ru)

            class_counts = {}
            for metadata in results['metadatas'][0]:
                class_name = metadata['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            similar_images = []
            other_class_images = []

            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]

                if metadata['class_name'] == predicted_class_en:
                    similar_images.append({
                        'id': results['ids'][0][i],
                        'path': metadata['path'],
                        'class_name': metadata['class_name'],
                        'distance': results['distances'][0][i]
                    })
                else:
                    other_class_images.append({
                        'id': results['ids'][0][i],
                        'path': metadata['path'],
                        'class_name': metadata['class_name'],
                        'distance': results['distances'][0][i]
                    })

            if len(similar_images) < n_results:
                need_more = n_results - len(similar_images)
                similar_images.extend(other_class_images[:need_more])

            return similar_images[:n_results]

        except Exception as e:
            print(f"Ошибка поиска похожих изображений: {e}")
            import traceback
            traceback.print_exc()
            return []

    def setup_ui(self):
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        # 1. ЗАГОЛОВОК
        title_label = tk.Label(
            main_container,
            text="КЛАССИФИКАТОР ПТИЦ",
            font=('Arial', 16, 'bold'),
            fg='#2c3e50'
        )
        title_label.pack(pady=(0, 10))

        # 2. ПАНЕЛЬ УПРАВЛЕНИЯ
        control_frame = tk.Frame(main_container, bg='#ecf0f1', relief=tk.RAISED, borderwidth=1)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        btn_frame = tk.Frame(control_frame, bg='#ecf0f1')
        btn_frame.pack(pady=8)

        self.select_btn = tk.Button(
            btn_frame,
            text="Выбрать изображение",
            command=self.select_image,
            font=('Arial', 10),
            bg="#3498db",
            fg="white",
            padx=15,
            pady=6
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.classify_btn = tk.Button(
            btn_frame,
            text="Классифицировать",
            command=self.classify_image,
            font=('Arial', 10),
            bg="#2ecc71",
            fg="white",
            padx=15,
            pady=6,
            state=tk.DISABLED
        )
        self.classify_btn.pack(side=tk.LEFT, padx=5)

        # 3. ОСНОВНОЕ СОДЕРЖИМОЕ
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(content_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # А. ИЗОБРАЖЕНИЕ
        image_frame = tk.LabelFrame(
            left_panel,
            text="Изображение",
            font=('Arial', 11, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        image_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(
            image_frame,
            text="",
            font=('Arial', 10),
            fg="#7f8c8d",
            bg="white",
            justify=tk.CENTER
        )
        self.image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Б. РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ
        result_frame = tk.LabelFrame(
            right_panel,
            text="Результаты",
            font=('Arial', 11, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        result_text_frame = tk.Frame(result_frame, height=220)
        result_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        result_text_frame.pack_propagate(False)

        self.result_text = tk.Text(
            result_text_frame,
            font=('Arial', 10),
            bg="#f8f9fa",
            fg="#2c3e50",
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=10
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

        self.result_text.insert(1.0, "")
        self.result_text.config(state=tk.DISABLED)

        # В. ПОХОЖИЕ ИЗОБРАЖЕНИЯ
        similar_frame = tk.LabelFrame(
            right_panel,
            text="Похожие изображения",
            font=('Arial', 11, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        similar_frame.pack(fill=tk.BOTH, expand=True)

        self.similar_canvas = tk.Canvas(similar_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(similar_frame, orient="vertical", command=self.similar_canvas.yview)

        self.scrollable_frame = tk.Frame(self.similar_canvas, bg='white')
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.similar_canvas.configure(scrollregion=self.similar_canvas.bbox("all"))
        )

        self.similar_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.similar_canvas.configure(yscrollcommand=scrollbar.set)

        self.similar_canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)

        tk.Label(
            self.scrollable_frame,
            text="Похожие изображения появятся здесь",
            font=('Arial', 9),
            fg="#95a5a6",
            bg="white",
            justify=tk.CENTER
        ).pack(pady=40)

        self.status_bar = tk.Label(
            main_container,
            text="Готов",
            font=('Arial', 9),
            bg='#34495e',
            fg='white',
            anchor=tk.W,
            padx=10
        )
        self.status_bar.pack(fill=tk.X, pady=(10, 0))

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.jpg *.jpeg *.png")]
        )

        if file_path:
            try:
                image = Image.open(file_path)
                preview = image.copy()
                preview.thumbnail((320, 240), Image.Resampling.LANCZOS)

                self.photo = ImageTk.PhotoImage(preview)
                self.image_label.config(
                    image=self.photo,
                    text="",
                    bg="white"
                )

                self.current_image = image.convert('RGB')

                self.classify_btn.config(state=tk.NORMAL, bg="#27ae60")

                self.status_bar.config(text="Изображение загружено")

                self.clear_results()

            except Exception as e:
                self.status_bar.config(text=f"Ошибка загрузки")

    def classify_image(self):
        if self.current_image is None:
            return

        try:
            self.status_bar.config(text="Классификация...")
            self.classify_btn.config(state=tk.DISABLED, bg="#95a5a6")
            self.root.update()

            img_tensor = self.transform(self.current_image).unsqueeze(0)

            # 1. Классификация
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                prob, idx = torch.max(probs, dim=1)

            bird_class = self.class_names[idx.item()]
            confidence = prob.item() * 100

            # 2. Поиск похожих изображений
            embedding = self.extract_embedding(self.current_image)
            similar_images = self.search_similar_images(embedding, n_results=10)

            self.show_results(bird_class, confidence, probs)
            self.show_similar_images(similar_images)

            self.status_bar.config(text="Готов")
            self.classify_btn.config(state=tk.NORMAL, bg="#2ecc71")

        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
            self.status_bar.config(text="Ошибка классификации")
            self.classify_btn.config(state=tk.NORMAL, bg="#2ecc71")

    def show_results(self, bird_class, confidence, probs):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        # Форматируем результат
        result = "Основной класс:\n"
        result += f"  {bird_class}\n\n"

        result += "Уверенность:\n"
        result += f"  {confidence:.1f}%\n\n"

        result += "Вероятности по другим классам:\n"

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        for i in range(len(self.class_names)):
            if i == 0:
                continue

            class_idx = sorted_indices[0][i].item()
            class_name = self.class_names[class_idx]
            prob_value = sorted_probs[0][i].item() * 100

            result += f"{class_name:10} - {prob_value:5.1f}%\n"

        self.result_text.insert(1.0, result)
        self.result_text.config(state=tk.DISABLED)
        self.result_text.see(1.0)

    def show_similar_images(self, similar_images):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        if not similar_images:
            tk.Label(
                self.scrollable_frame,
                text="Похожие изображения не найдены",
                font=('Arial', 9),
                fg="#95a5a6",
                bg="white",
                justify=tk.CENTER
            ).pack(pady=40)
            return

        grid_frame = tk.Frame(self.scrollable_frame, bg='white')
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for i, img_info in enumerate(similar_images[:10]):
            try:
                img_path = img_info['path']
                if not Path(img_path).exists():
                    filename = Path(img_path).name
                    img_path = f"data/raw/{img_info['class_name']}/{filename}"

                if Path(img_path).exists():
                    img = Image.open(img_path)

                    img.thumbnail((120, 120), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)

                    img_frame = tk.Frame(grid_frame, bg='white', relief=tk.RAISED, borderwidth=1)

                    row = i // 2
                    col = i % 2
                    img_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

                    img_label = tk.Label(img_frame, image=photo, bg='white')
                    img_label.image = photo  # Сохраняем ссылку
                    img_label.pack(padx=5, pady=5)

                    class_label = tk.Label(
                        img_frame,
                        text=img_info['class_name'],
                        font=('Arial', 9, 'bold'),
                        bg='white',
                        fg='#2c3e50'
                    )
                    class_label.pack(pady=(0, 5))

            except Exception as e:
                print(f"Ошибка загрузки похожего изображения: {e}")
                continue

        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)

        if not grid_frame.winfo_children():
            tk.Label(
                self.scrollable_frame,
                text="Не удалось загрузить похожие изображения",
                font=('Arial', 9),
                fg="#95a5a6",
                bg="white",
                justify=tk.CENTER
            ).pack(pady=40)

    def clear_results(self):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        tk.Label(
            self.scrollable_frame,
            text="Похожие изображения появятся здесь",
            font=('Arial', 9),
            fg="#95a5a6",
            bg="white",
            justify=tk.CENTER
        ).pack(pady=40)

    def run(self):
        self.root.mainloop()


def main():
    app = MinimalBirdClassifier()
    app.run()


if __name__ == "__main__":
    main()
