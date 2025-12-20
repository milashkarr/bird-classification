import json
from datetime import datetime
from pathlib import Path


def create_report():
    report_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_stages": 4,
        "stages": ["prepare", "train", "embeddings", "report"]
    }

    try:
        with open("metrics/data_metrics.json", "r", encoding="utf-8") as f:
            data_metrics = json.load(f)
            report_data["data_metrics"] = data_metrics
    except:
        report_data["data_metrics"] = "N/A"

    try:
        with open("metrics/train_metrics.json", "r", encoding="utf-8") as f:
            train_metrics = json.load(f)
            report_data["train_metrics"] = train_metrics
    except:
        report_data["train_metrics"] = "N/A"

    try:
        with open("chroma_db_info.json", "r", encoding="utf-8") as f:
            db_info = json.load(f)
            report_data["vector_db"] = db_info
    except:
        report_data["vector_db"] = "N/A"

    report_data["models_exist"] = {
        "classification_model": Path("models/best_model.pth").exists(),
        "vector_database": Path("chroma_db").exists()
    }

    with open("metrics/summary.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("DVC PROJECT SUMMARY")
    print("=" * 60)
    print(f"Date: {report_data['date']}")
    print(f"Pipeline stages: {report_data['pipeline_stages']}")
    print(f"Total images: {data_metrics.get('total_images', 'N/A') if isinstance(data_metrics, dict) else 'N/A'}")
    print(f"Classes: {data_metrics.get('classes', 'N/A') if isinstance(data_metrics, dict) else 'N/A'}")
    print(f"Model accuracy: {train_metrics.get('best_val_acc', 'N/A') if isinstance(train_metrics, dict) else 'N/A'}%")
    print(f"Vector DB images: {db_info.get('total_images', 'N/A') if isinstance(db_info, dict) else 'N/A'}")
    print("=" * 60)


if __name__ == "__main__":
    create_report()