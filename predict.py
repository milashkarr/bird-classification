import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
import os
from pathlib import Path

class MinimalBirdClassifier:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Классификатор птиц")
        self.root.geometry("900x650")
        self.root.resizable(True, True)
        
        # Центрируем окно
        self.center_window()
        
        # Загрузка модели
        self.class_names = ['Зяблик', 'Пеночка', 'Щегол', 'Поползень', 'Ласточка']
        self.model = self.load_model()
        
        # Трансформации
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Инициализация
        self.current_image = None
        self.photo = None
        
        self.setup_ui()
    
    def center_window(self):
        """Центрирование окна"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def load_model(self):
        """Загрузка модели"""
        model = models.mobilenet_v2()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
        
        try:
            if Path("models/best_model.pth").exists():
                checkpoint = torch.load("models/best_model.pth", 
                                      map_location='cpu',
                                      weights_only=False)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
        
        model.eval()
        return model
    
    def setup_ui(self):
        """Создание минималистичного интерфейса"""
        # Главный контейнер
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
        
        # Кнопки
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
        
        # Левая часть - изображение
        left_panel = ttk.Frame(content_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Правая часть - результаты
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
        
        # Контейнер для изображения
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
        
        # Контейнер для текста
        result_text_frame = tk.Frame(result_frame, height=220)
        result_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        result_text_frame.pack_propagate(False)
        
        # Многострочный текст
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
        
        # Начальный текст
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
        
        # Canvas с прокруткой
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
        
        # Заглушка
        tk.Label(
            self.scrollable_frame,
            text="Похожие изображения появятся здесь",
            font=('Arial', 9),
            fg="#95a5a6",
            bg="white",
            justify=tk.CENTER
        ).pack(pady=40)
        
        # 4. СТАТУС БАР
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
        """Выбор изображения"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                # Загрузка и масштабирование
                image = Image.open(file_path)
                preview = image.copy()
                preview.thumbnail((320, 240), Image.Resampling.LANCZOS)
                
                # Конвертируем
                self.photo = ImageTk.PhotoImage(preview)
                self.image_label.config(
                    image=self.photo,
                    text="",
                    bg="white"
                )
                
                # Сохраняем оригинал
                self.current_image = image.convert('RGB')
                
                # Активируем кнопку
                self.classify_btn.config(state=tk.NORMAL, bg="#27ae60")
                
                # Обновляем статус
                self.status_bar.config(text="Изображение загружено")
                
                # Очищаем результаты
                self.clear_results()
                
            except Exception as e:
                self.status_bar.config(text=f"Ошибка загрузки")
    
    def classify_image(self):
        """Классификация изображения"""
        if self.current_image is None:
            return
        
        try:
            self.status_bar.config(text="Классификация...")
            self.classify_btn.config(state=tk.DISABLED, bg="#95a5a6")
            self.root.update()
            
            # Преобразуем изображение
            img_tensor = self.transform(self.current_image).unsqueeze(0)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                prob, idx = torch.max(probs, dim=1)
            
            # Результат
            bird_class = self.class_names[idx.item()]
            confidence = prob.item() * 100
            
            # Отображаем результаты
            self.show_results(bird_class, confidence, probs)
            self.show_similar_placeholder(bird_class)
            
            self.status_bar.config(text="Готов")
            self.classify_btn.config(state=tk.NORMAL, bg="#2ecc71")
            
        except Exception as e:
            self.status_bar.config(text="Ошибка классификации")
            self.classify_btn.config(state=tk.NORMAL, bg="#2ecc71")
    
    def show_results(self, bird_class, confidence, probs):
        """Отображение результатов"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        # Форматируем результат
        result = "Основной класс:\n"
        result += f"  {bird_class}\n\n"
        
        result += "Уверенность:\n"
        result += f"  {confidence:.1f}%\n\n"
        
        result += "Вероятности по другим классам:\n"
        
        # Сортируем по вероятности (исключая основной класс)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        for i in range(len(self.class_names)):
            if i == 0:
                continue  # Пропускаем основной класс
                
            class_idx = sorted_indices[0][i].item()
            class_name = self.class_names[class_idx]
            prob_value = sorted_probs[0][i].item() * 100
            
            result += f"{class_name:10} - {prob_value:5.1f}%\n"
        
        # Вставляем текст
        self.result_text.insert(1.0, result)
        self.result_text.config(state=tk.DISABLED)
        self.result_text.see(1.0)
    
    def show_similar_placeholder(self, bird_class):
        """Заглушка для похожих изображений"""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Заглушка
        tk.Label(
            self.scrollable_frame,
            text="Похожие изображения появятся здесь",
            font=('Arial', 9),
            fg="#95a5a6",
            bg="white",
            justify=tk.CENTER
        ).pack(pady=40)
    
    def clear_results(self):
        """Очистка результатов"""
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
        """Запуск приложения"""
        self.root.mainloop()

def main():
    """Точка входа"""
    app = MinimalBirdClassifier()
    app.run()

if __name__ == "__main__":
    # Скрываем консоль на Windows
    import sys
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
        except:
            pass
    
    main()
    