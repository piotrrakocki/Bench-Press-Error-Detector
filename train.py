import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from data_augmentation.video_augmentation import VideoAugmenter
from data_processing.dataset import create_dataset
from data_processing.video_utils import VideoProcessor
from models.metrics import F1Score
from models.model_3d_cnn import build_3d_cnn_multilabel

import config

def main():
    # ================================
    # Ustawienia z config.py
    # ================================
    num_frames = config.NUM_FRAMES
    img_size = config.IMG_SIZE
    batch_size = config.BATCH_SIZE
    epochs = config.EPOCHS
    error_mapping = config.ERROR_MAPPING
    video_dir = config.VIDEO_DIR
    frames_base_dir = config.FRAMES_BASE_DIR
    num_errors = len(error_mapping)

    # ================================
    # (Opcjonalnie) Augmentacja wideo
    # ================================
    augmenter = VideoAugmenter(output_folder="augmented_videos")
    # Przykład: przetworzenie wszystkich wideo w folderze
    # augmenter.process_input_path(video_dir)

    # ================================
    # Ekstrakcja klatek z wideo (jeśli nie są jeszcze zapisane)
    # ================================
    processor = VideoProcessor(num_frames=num_frames, img_size=img_size, error_mapping=error_mapping)
    video_frames_info = []  # Lista krotek: (ścieżka_do_folderu_z_klatkami, wektor_etykiet)

    if not os.path.exists(frames_base_dir):
        os.makedirs(frames_base_dir, exist_ok=True)
        video_files = glob.glob(os.path.join(video_dir, "*.mp4")) + glob.glob(os.path.join(video_dir, "*.avi"))
        video_files = sorted(video_files)

        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(frames_base_dir, video_name)
            try:
                processor.save_video_frames_to_disk(video_path, output_dir)
                # Zakładamy, że nazwa pliku (lub folderu) zawiera etykiety
                label_vector = VideoProcessor.extract_labels_from_filename(video_name, error_mapping)
                video_frames_info.append((output_dir, label_vector))
            except Exception as e:
                print(f"Błąd przy przetwarzaniu {video_path}: {e}")
                continue
    else:
        for folder_name in os.listdir(frames_base_dir):
            full_path = os.path.join(frames_base_dir, folder_name)
            if os.path.isdir(full_path):
                label_vector = VideoProcessor.extract_labels_from_filename(folder_name, error_mapping)
                video_frames_info.append((full_path, label_vector))

    # ================================
    # Podział na zbiór treningowy i walidacyjny
    # ================================
    np.random.shuffle(video_frames_info)
    total_samples = len(video_frames_info)
    train_samples = int(0.8 * total_samples)
    train_info = video_frames_info[:train_samples]
    val_info = video_frames_info[train_samples:]

    # ================================
    # Przygotowanie tf.data.Dataset
    # ================================
    train_dataset = create_dataset(
        info_list=train_info,
        video_processor=processor,
        num_frames=num_frames,
        img_size=img_size,
        num_errors=num_errors,
        batch_size=batch_size,
        shuffle_buffer=train_samples,
        repeat=True
    )

    val_dataset = create_dataset(
        info_list=val_info,
        video_processor=processor,
        num_frames=num_frames,
        img_size=img_size,
        num_errors=num_errors,
        batch_size=batch_size
    )

    steps_per_epoch = train_samples // batch_size
    validation_steps = (total_samples - train_samples) // batch_size
    print(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

    # ================================
    # Budowanie i kompilacja modelu 3D CNN
    # ================================
    input_shape = (num_frames, img_size[1], img_size[0], 3)
    model = build_3d_cnn_multilabel(input_shape, num_errors)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            F1Score(name='f1_score')
        ]
    )

    # ================================
    # Trenowanie modelu
    # ================================
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps
    )

    # Zapis modelu
    model.save("model_3d_cnn_multilabel_tfV11.h5")
    print("Model został zapisany.")

    # ================================
    # Wykres strat (loss) i dokładności (binary accuracy)
    # ================================
    plt.figure(figsize=(12, 5))

    # Wykres strat
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Strata - trening')
    plt.plot(history.history['val_loss'], label='Strata - walidacja')
    plt.title('Strata w kolejnych epokach')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()

    # Wykres dokładności
    plt.subplot(1, 2, 2)
    plt.plot(history.history['binary_accuracy'], label='Dokładność - trening')
    plt.plot(history.history['val_binary_accuracy'], label='Dokładność - walidacja')
    plt.title('Dokładność w kolejnych epokach')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()

    plt.tight_layout()

    # Zapis wykresu do pliku
    plt.savefig("loss_accuracy_plot.png")
    plt.show()

    # ================================
    # Wykres metryk: Precision, Recall, F1 Score
    # ================================
    plt.figure(figsize=(18, 5))

    # Precision
    plt.subplot(1, 3, 1)
    plt.plot(history.history['precision'], label='Precision - trening')
    plt.plot(history.history['val_precision'], label='Precision - walidacja')
    plt.title('Precision')
    plt.xlabel('Epoka')
    plt.ylabel('Precision')
    plt.legend()

    # Recall
    plt.subplot(1, 3, 2)
    plt.plot(history.history['recall'], label='Recall - trening')
    plt.plot(history.history['val_recall'], label='Recall - walidacja')
    plt.title('Recall')
    plt.xlabel('Epoka')
    plt.ylabel('Recall')
    plt.legend()

    # F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(history.history['f1_score'], label='F1 - trening')
    plt.plot(history.history['val_f1_score'], label='F1 - walidacja')
    plt.title('F1 Score')
    plt.xlabel('Epoka')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()

    # Zapis wykresu do pliku
    plt.savefig("metrics_plot.png")
    plt.show()

    # Pobierz etykiety i predykcje ze zbioru walidacyjnego
    y_true = []
    y_pred = []
    for batch in val_dataset:
        X, y = batch
        preds = model.predict(X)
        y_true.extend(y.numpy())
        y_pred.extend(preds)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_classes = y_true.shape[1]
    plt.figure(figsize=(12, 8))

    # Oblicz i wykreśl ROC dla każdej klasy
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Klasa {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywe ROC dla poszczególnych etykiet')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curves.png")
    plt.show()

    from sklearn.metrics import precision_recall_curve

    plt.figure(figsize=(12, 8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        plt.plot(recall, precision, lw=2, label=f'Klasa {i}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Krzywe Precision-Recall dla poszczególnych etykiet')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("pr_curves.png")
    plt.show()

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Ustal próg, np. 0.5
    threshold = 0.5
    y_pred_binary = (y_pred[:, 0] > threshold).astype(int)
    cm = confusion_matrix(y_true[:, 0], y_pred_binary)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywistość")
    plt.title("Macierz pomyłek dla etykiety 0")
    plt.tight_layout()
    plt.savefig("confusion_matrix_label0.png")
    plt.show()


if __name__ == "__main__":
    main()
