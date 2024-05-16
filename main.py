import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import json
import os
from pprint import pprint
import matplotlib.pyplot as plt

from dataset import load_dataset_train, load_dataset_normal
from models import get_model_1, get_model_2, get_model_3
# set random seed
seed = 2390098
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)
tf.config.experimental.enable_op_determinism()


def get_best_results(batch_names):
    best_results = []

    for batch_name in batch_names:
        with open(f"{output_dir}/{batch_name}.json", "r") as f:
            json_data = json.load(f)
            mel_acc = json_data["mel"][1]
            other_acc = json_data["other"][1]

            avg_acc = (mel_acc + other_acc) / 2

            best_results.append((batch_name, avg_acc))

    sorted_results = sorted(best_results, key=lambda x: x[1], reverse=True)
    print(sorted_results[:5])


train_data_dir = r"data\ISIC2018_Task3_Training_Input"
train_csv_file = r"data\ISIC2018_Task3_Training_GroundTruth.csv"

test_data_dir = r"data\ISIC2018_Task3_Test_Input"
test_csv_file = r"data\ISIC2018_Task3_Test_GroundTruth.csv"

output_dir = r"./output_test"

def get_best_model_cb(save_path):
    return tf.keras.callbacks.ModelCheckpoint(f"{save_path}.keras", monitor="val_loss", save_best_only=True)


def get_early_stopping(start, patience):
    return tf.keras.callbacks.EarlyStopping(
        patience=patience, restore_best_weights=True, monitor="val_loss", start_from_epoch=start
    )

sizes = [32, 64, 128, 256]

image_sizes = [(45, 60), (60, 80)]
lrs = [0.001, 0.00075, 0.0005]
models = [get_model_1, get_model_2, get_model_3]
crop_sizes = [0.25, 0.3, 0.35]
batch_sizes = [32, 64]

# create all permutations of the above parameters
setup_configs = []
for image_size in image_sizes:
    for lr in lrs:
        for crop_size_percent in crop_sizes:
            for batch_size in batch_sizes:
                setup_configs.append(
                    {
                        "image_size": image_size,
                        "lr": lr,
                        "model": get_model_1,
                        "crop_size_percent": crop_size_percent,
                        "batch_size": batch_size,
                    }
                )
                
                
def load_and_evaluate_models() -> list[dict]:
    model_files = os.listdir(output_dir)
    model_files = [f for f in model_files if f.endswith(".keras")]
    
    output_data = []

    for i in model_files:
        lr, image_size, crop_size_percent, batch_size = get_configuration_data_from_filename(f"{output_dir}/{i}")
        
        model = keras.models.load_model(f"{output_dir}/{i}")
        model_name = i.rstrip(".keras")
        
        test_images, test_labels = load_dataset_normal(
            test_csv_file, test_data_dir, image_size, crop_size_percent=crop_size_percent, batch_name=get_image_batch_name(image_size, crop_size_percent)
        )
        
        model_data = evaluate_model(model, model_name, test_images, test_labels, lr, image_size, crop_size_percent, batch_size)
        output_data.append(model_data)

        
    # sort the models by overall_score
    sorted_models = sorted(output_data, key=lambda x: x["overall_score"], reverse=True)

    with open(f"experiment_results.json", "w") as f:
        json.dump(sorted_models, f)
    
    pprint(sorted_models[:10])
    
def evaluate_model(model, model_name, test_images, test_labels, lr, image_size, crop_size_percent, batch_size):
    
    predictions = model.predict(test_images)
    
    # convert predictions to binary
    predictions = np.round(predictions).astype(int)
    print(predictions)
    mel_predictions = predictions[test_labels == 0]
    other_predictions = predictions[test_labels == 1]

    
    mel_acc = accuracy_score(test_labels[test_labels == 0], mel_predictions)
    other_acc = accuracy_score(test_labels[test_labels == 1], other_predictions)
    overall_accuracy = (mel_acc + other_acc) / 2
    
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    
    conf_matrix = confusion_matrix(test_labels, predictions)
    
    fpr, tpr, _ = roc_curve(test_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    overall_score = overall_accuracy + precision + recall + f1 + roc_auc
    
    model_data = {
        "model_name":model_name,
        "lr": lr,
        "image_size": image_size,
        "crop_size_percent": crop_size_percent,
        "batch_size": batch_size,  
        "overall_accuracy": overall_accuracy, 
        "mel_accuracy": mel_acc, 
        "other_accuracy": other_acc,
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "confusion_matrix": conf_matrix.tolist(), 
        "roc_auc": roc_auc, 
        "overall_score": overall_score
        }
    
    return model_data

def param_search():


    names = []
    for count, config in enumerate(setup_configs):
        model = config["model"]
        model_name = model.__name__
        lr = config["lr"]
        image_size = config["image_size"]
        crop_size_percent = config["crop_size_percent"]
        batch_size = config["batch_size"]
        batch_name = f"batch_{model_name}__{lr}__{str(image_size)}__{crop_size_percent}__{batch_size}"
        
        # the batch name only including the preprocessing parameters, not the model parameters
        image_batch_name = get_image_batch_name(image_size, crop_size_percent)
        names.append(batch_name)
        
        train_images, train_labels = load_dataset_train(
            train_csv_file, train_data_dir, image_size, crop_size_percent=crop_size_percent, batch_name=image_batch_name
        )

        model = model(sizes, image_size)

        # shuffle the images
        indices = np.arange(len(train_images))
        np.random.shuffle(indices)
        images = train_images[indices]
        labels = train_labels[indices]

        adamax = keras.optimizers.Adamax(learning_rate=lr)
        model.compile(optimizer=adamax, loss="binary_crossentropy", metrics=["accuracy"])

        early_stopping = get_early_stopping(15, 20)
        best_model = get_best_model_cb(f"{output_dir}/{batch_name}")
        model.fit(
            images,
            labels,
            epochs=300,
            validation_split=0.2,
            batch_size=batch_size,
            callbacks=[early_stopping, best_model],
        )

        test_images, test_labels = load_dataset_normal(
            test_csv_file, test_data_dir, image_size, crop_size_percent=crop_size_percent, batch_name=image_batch_name
        )

        overall_eval = model.evaluate(test_images, test_labels)

        mel_eval = model.evaluate(test_images[test_labels == 0], test_labels[test_labels == 0])

        other_eval = model.evaluate(test_images[test_labels == 1], test_labels[test_labels == 1])

        out_data = {"overall": overall_eval, "mel": mel_eval, "other": other_eval}

        with open(f"{output_dir}/{batch_name}.json", "w") as f:
            json.dump(out_data, f)
            
def get_image_batch_name(image_size, crop_size_percent):
    return f"batch_{str(image_size)}__{crop_size_percent}"

def get_configuration_data_from_filename(filename):
    filename = filename.split("/")[-1].rstrip(".keras")
    parts = filename.split("__")
    grouped_parts = [x.split("_") for x in parts]

    flat_parts = [part for group in grouped_parts for part in group]
    flat_parts = flat_parts[4:]
    print(flat_parts)
    lr = float(flat_parts[0])
    image_size = tuple([int(i) for i in flat_parts[1].strip("()").split(",")])
    crop_size_percent = float(flat_parts[2])
    
    return lr, image_size, crop_size_percent, batch_size
    
def train_other_models_on_best_config():
    with open("experiment_results.json", "r") as f:
        data = json.load(f)
        
    best_configs = data[:5]
    models_to_test = [get_model_2, get_model_3]
    
    output_data = []
    
    for model_builder in models_to_test:
        for config in best_configs:
            lr = config["lr"]
            image_size = config["image_size"]
            crop_size_percent = config["crop_size_percent"]
            batch_size = config["batch_size"]
            
            model_name = f"{model_builder.__name__}__{lr}__{image_size}__{crop_size_percent}__{batch_size}"
            
            
            train_images, train_labels = load_dataset_train(
                train_csv_file, train_data_dir, image_size, crop_size_percent=crop_size_percent, batch_name=get_image_batch_name(image_size, crop_size_percent)
            )
            
            model = model_builder([64, 128, 256, 512], image_size)
            
            # shuffle the images
            indices = np.arange(len(train_images))
            np.random.shuffle(indices)
            images = train_images[indices]
            labels = train_labels[indices]
            
            adamax = keras.optimizers.Adamax(learning_rate=lr)
            model.compile(optimizer=adamax, loss="binary_crossentropy", metrics=["accuracy"])
            
            early_stopping = get_early_stopping(14, 12)
            best_model = get_best_model_cb(f"final_comparison/{model_name}")
            model.fit(
                images,
                labels,
                epochs=300,
                validation_split=0.2,
                batch_size=batch_size,
                callbacks=[early_stopping, best_model],
            )
            
            test_images, test_labels = load_dataset_normal(
                test_csv_file, test_data_dir, image_size, crop_size_percent=crop_size_percent, batch_name=get_image_batch_name(image_size, crop_size_percent)
            )
            
            
            model_data = evaluate_model(model, model_name, test_images, test_labels, lr, image_size, crop_size_percent, batch_size)
            output_data.append(model_data)

    
    with open("experiment_results.json", "r") as f:
        data = json.load(f)
    
    output_data.extend(data[:5])
    
    sorted_models = sorted(output_data, key=lambda x: x["overall_score"], reverse=True)
    
    # also include the worst model
    best_models = sorted_models[:3]
    best_models.append(sorted_models[-1])
    pprint(best_models)
    
    with open("./final_results.json", "w") as f:
        json.dump(best_models, f)
        
    with open("./all_final_results.json", "r") as f:
        json.dump(sorted_models, f)
        
    for i in best_models:
        create_graphics_from_results(f"final_comparison/{i['model_name']}.keras", i["model_name"], i)
        print(i["model_name"])
    
    # create some graphics for the best models

def create_final_graphics(final_results_path, models_path):
    with open(final_results_path, "r") as f:
        final_results = json.load(f)
        
    for count, model_data in enumerate(final_results):
        model_name = model_data["model_name"]
        lr = model_data["lr"]
        image_size = model_data["image_size"]
        crop_size_percent = model_data["crop_size_percent"]
        batch_size = model_data["batch_size"]
        
        model_path = f"{models_path}/{model_name}.keras"
        
        create_graphics_from_results(model_path, f"{count}_{model_name}", model_data)
        
        print(f"Model {count + 1} done")
    
            
def create_graphics_from_results(model_path, model_name, results_dict):
    image_size = results_dict["image_size"]
    crop_size_percent = results_dict["crop_size_percent"]
    
    image_batch_name = get_image_batch_name(image_size, crop_size_percent)
    print(model_path)
    try:
        # split the path and filename into separate variables
        model = keras.models.load_model(f"{model_path}/{filename}")
    except KeyboardInterrupt as e:
        raise
    except BaseException as e:
        filename = model_path.split("/")[-1]
        model = keras.models.load_model(f"output_test/{filename}")
    
    
    test_images, test_labels = load_dataset_normal(
        test_csv_file, test_data_dir, image_size, crop_size_percent=crop_size_percent, batch_name=image_batch_name
    )
    
    predictions = model.predict(test_images)
    
    # convert predictions to binary
    predictions = np.round(predictions).astype(int)
    
    
    
    
    # create auc curve graph
    fpr, tpr, _ = roc_curve(test_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    create_roc_graph(f"{model_name}_roc.png", fpr, tpr, roc_auc)
    create_confusion_matrix_graph(f"{model_name}_confusion_matrix.png", confusion_matrix(test_labels, predictions))
    
def create_roc_graph(output_path, fpr, tpr, auc):
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    
    
def create_confusion_matrix_graph(output_path, conf_matrix):
    conf_matrix = np.array(conf_matrix)
    plt.figure()
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Melanoma", "Other"])
    plt.yticks([0, 1], ["Melanoma", "Other"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_path)


#param_search()
#improve_best_model()
#load_and_evaluate_models()

train_other_models_on_best_config()

#create_final_graphics(r"./final_results.json", r"./final_models")

