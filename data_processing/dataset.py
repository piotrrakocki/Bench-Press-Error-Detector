import tensorflow as tf

def data_generator(info_list, video_processor):
    for folder_path, label_vector in info_list:
        try:
            frames = video_processor.load_frames_from_folder(folder_path)
            yield frames, label_vector
        except Exception as e:
            print(f"Błąd podczas wczytywania {folder_path}: {e}")
            continue

def create_dataset(info_list, video_processor, num_frames, img_size, num_errors, batch_size, shuffle_buffer=None, repeat=False):
    output_signature = (
        tf.TensorSpec(shape=(num_frames, img_size[1], img_size[0], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(num_errors,), dtype=tf.int32)
    )
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(info_list, video_processor),
        output_signature=output_signature
    )
    if shuffle_buffer:
        ds = ds.shuffle(buffer_size=shuffle_buffer)
    ds = ds.batch(batch_size)
    if repeat:
        ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
