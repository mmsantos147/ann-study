import cv2
import tensorflow as tf
import numpy as np
import os


def read_images(path, hw):
    """
    Lê e processa imagens de uma pasta.
    hw -> altura e largura para redimensionalizar as imagens
    """
    if not os.path.isdir(path):
        print(f"The folder '{path}' does not exist or is not valid.")
        return []

    images = []
    for file in os.listdir(path):
        if file.lower().endswith('.jpg'):
            try:
                image = cv2.imread(os.path.join(path, file))
                if image is None:
                    print(f"Error loading image '{file}'.")
                    continue
                image = cv2.resize(image, (hw, hw)) / 255.0
                images.append(image)
            except Exception as e:
                print(f"Error processing image '{file}': {e}")
    return images


def randomize(data_a, data_na, sample_ratio=0.3):
    """
    Randomiza e divide os dados em treino e validação.
    """
    np.random.seed(42)

    def split_data(data, ratio):
        size = int(len(data) * ratio)
        indices = np.random.choice(len(data), size, replace=False)
        return np.array(data)[indices], np.array(data)[np.setdiff1d(np.arange(len(data)), indices)]

    sample_a, train_a = split_data(data_a, sample_ratio)
    sample_na, train_na = split_data(data_na, sample_ratio)

    train_all = np.concatenate([train_a, train_na])
    targets_all = np.array([1] * len(train_a) + [0] * len(train_na))

    indices = np.arange(len(train_all))
    np.random.shuffle(indices)

    valid_x = np.concatenate([sample_a, sample_na])
    valid_y = np.array([1] * len(sample_a) + [0] * len(sample_na))

    return train_all[indices], targets_all[indices], valid_x, valid_y


def build_model(hw):
    """
    Cria e retorna um modelo Keras.
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(hw, hw, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model


def evaluate_images(path, model, hw, threshold=0.5):
    """
    Avalia imagens de uma pasta usando o modelo.
    """
    images = read_images(path, hw)  # Usa a função read_images para carregar as imagens.
    if not images:
        print(f"No images were loaded from the folder '{path}'.")
        return

    total = len(images)
    correct, incorrect = 0, 0
    expected_label = 'non_autistic' if 'non_autistic' in path.lower() else 'autistic'

    for idx, image in enumerate(images):
        prediction = model.predict(image.reshape(1, hw, hw, 3))[0][0]
        classification = 'non_autistic' if prediction < threshold else 'autistic'
        is_correct = classification == expected_label

        if is_correct:
            correct += 1
        else:
            incorrect += 1

        print(f"Image {idx + 1}: Predicted: {prediction:.2f}, Classification: {classification}")

    accuracy = correct / total * 100
    error_rate = incorrect / total * 100

    print(f"\nTotal Images: {total}, Correct Predictions: {correct}, Errors: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}%, Error Rate: {error_rate:.2f}%")


def main():
    hw = 224
    model = build_model(hw)

    train_a = read_images("autism/images/train/autistic", hw)
    train_na = read_images("autism/images/train/non_autistic", hw)
    train_all, targets, valid_x, valid_y = randomize(train_a, train_na)

    while True:
        command = input("'1' Train the model \n'2' Evaluate autistic images \n'3' Evaluate non-autistic images \n'4' Save the model \n'5' Load a model \n'0' Exit: ")

        if command == '1':
            print("Training the model...")
            model.fit(train_all, targets, epochs=10, batch_size=8, validation_data=(valid_x, valid_y))

        elif command == '2':
            evaluate_images('autism/images/valid/autistic', model, hw)

        elif command == '3':
            evaluate_images('autism/images/valid/non_autistic', model, hw)

        elif command == '4':
            model.save('autism/neural_networks/model.keras')
            print("Model saved.")

        elif command == '5':
            models_path = 'autism/neural_networks'
            try:
                models = [f for f in os.listdir(models_path) if f.endswith('.keras')]
                if not models:
                    print("No models found.")
                    continue
                for i, m in enumerate(models, 1):
                    print(f"{i}. {m}")
                choice = int(input("Choose a model to load: ")) - 1
                model = tf.keras.models.load_model(os.path.join(models_path, models[choice]))
                print(f"Model '{models[choice]}' loaded.")
            except Exception as e:
                print(f"Error loading the model: {e}")
        elif command == '0':
            print("Exiting...")
            break
        else:
            print("Invalid command.")


if __name__ == "__main__":
    main()
