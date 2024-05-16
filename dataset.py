import os
import numpy as np
import cv2
from skimage.transform import resize
from collections import Counter


def load_dataset_train(csv_file, data_dir, image_size, crop_size_percent=0.3, batch_name=None):
    if batch_name is not None and  os.path.exists(f"cached_image_sets/{batch_name}_images.npy") and os.path.exists(f"cached_image_sets/{batch_name}_labels.npy"):
        images = np.load(f"cached_image_sets/{batch_name}_images.npy")
        labels = np.load(f"cached_image_sets/{batch_name}_labels.npy")
        return images, labels
    
    target_labels = [x for x in range(1, 7)]
    total_target = 1113
    rows = np.genfromtxt(csv_file, delimiter=",", skip_header=1, dtype=str)
    labels = [np.argmax(row[1:].astype(float).astype(np.uint8)) for row in rows]
    labels = np.array([1 if x > 0 else 0 for x in labels], dtype=np.uint8)

    mel_indexes = np.where(labels == 0)[0]
    mel_selected_indexes = np.random.choice(mel_indexes, total_target, replace=False)

    other_selected_indexes, other_labels = get_indexes(labels, target_labels, total_target)
    selected_indexes = np.concatenate([mel_selected_indexes, other_selected_indexes])

    out_images = np.zeros((total_target * 2 * 3, image_size[0], image_size[1], 3), dtype=np.float32)

    new_labels = []
    total_images = 0

    for count, i in enumerate(selected_indexes):
        row = rows[i]
        label = np.argmax(row[1:].astype(float).astype(np.uint8))
        label = 1 if label > 0 else 0
        image_id = str(row[0])
        image_file = image_id + ".jpg"
        image_path = os.path.join(data_dir, image_file)

        images = load_image_train(image_path, image_size, crop_size_percent=crop_size_percent)

        for j in images:
            new_labels.append(label)
            out_images[total_images] = j
            total_images += 1
            
    if batch_name is not None:
        np.save(f"cached_image_sets/{batch_name}_images", out_images)
        np.save(f"cached_image_sets/{batch_name}_labels", np.array(new_labels))

    return out_images, np.array(new_labels)


def load_dataset_normal(csv_file, data_dir, image_size, crop_size_percent=0.3, batch_name=None):
    if  batch_name is not None and  os.path.exists(f"cached_test_image_sets/{batch_name}_images.npy") and os.path.exists(f"cached_test_image_sets/{batch_name}_labels.npy"):
        images = np.load(f"cached_test_image_sets/{batch_name}_images.npy")
        labels = np.load(f"cached_test_image_sets/{batch_name}_labels.npy")
        return images, labels
    data = np.genfromtxt(csv_file, delimiter=",", skip_header=1, dtype=str)
    images = np.zeros((len(data), image_size[0], image_size[1], 3), dtype=np.float32)
    labels = []

    for count, row in enumerate(data):
        label = np.argmax(row[1:].astype(float).astype(np.uint8))
        label = 1 if label > 0 else 0
        image_id = str(row[0])
        image_file = image_id + ".jpg"
        image_path = os.path.join(data_dir, image_file)
        image = load_image_test(image_path, image_size, crop_size_percent=crop_size_percent)
        images[count] = image
        labels.append(label)

    labels = np.array(labels, dtype=np.uint8)
    
    if batch_name is not None:
        np.save(f"cached_test_image_sets/{batch_name}_images", images)
        np.save(f"cached_test_image_sets/{batch_name}_labels", labels)

    return images, labels


def get_indexes(labels, target_labels, target):
    outputs = {label: [] for label in target_labels}
    counts = Counter(labels)
    target_labels = sorted(target_labels, key=lambda x: counts[x])
    average = target // len(target_labels)
    current_total = 0

    for count, i in enumerate(target_labels):
        current_total = sum([len(outputs[x]) for x in outputs])
        remaining_classes_count = len(target_labels) - count

        average = (target - current_total) // remaining_classes_count
        if counts[i] <= average:
            indexes = np.where(labels == i)[0]
            selected = np.random.choice(indexes, counts[i], replace=False)
            outputs[i] = selected

        else:
            indexes = np.where(labels == i)[0]
            selected = np.random.choice(indexes, average, replace=False)
            outputs[i] = selected

    indexes = [item for sublist in outputs.values() for item in sublist]
    new_labels = [labels[i] for i in indexes]

    return indexes, new_labels


def load_image_train(image_path, image_size, crop_size_percent=0.3) -> np.ndarray[np.ndarray]:
    image = cv2.imread(image_path)
    image = np.array(image)
    image = preprocess(image, image_size, crop_size_percent=crop_size_percent)
    images = augment(image)
    return images


def load_image_test(image_path, image_size, crop_size_percent=0.3):
    image = cv2.imread(image_path)
    image = np.array(image)
    image = preprocess(image, image_size, crop_size_percent=crop_size_percent)
    return image


def preprocess(image: np.ndarray, image_size, crop_size_percent=0.3):
    image = resize_and_crop(image, (image_size[0], image_size[1]), crop_size_percent=crop_size_percent)
    return image


def resize_and_crop(image, image_size, crop_size_percent=0.3):
    resized_height, resized_width = image_size
    height, width, channels = image.shape

    width_crop = int((width * crop_size_percent))
    height_crop = int((height * crop_size_percent))
    cropped = image[height_crop // 2 : height - height_crop // 2, width_crop // 2 : width - width_crop // 2 :]
    image = resize(cropped, (resized_height, resized_width))

    return image


def augment(image: np.ndarray):
    image_1 = np.fliplr(image)
    image_2 = np.flipud(image)

    return np.array([image, image_1, image_2])