import os
import shutil
import gdown
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from skimage import morphology, io
from scipy.ndimage import convolve
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


def erosionar(image):

    # Operaciones morfológica
    kernel = np.ones((100, 100), np.uint8)
    # Dilatación
    # Erosión
    imgE = cv2.erode(image, kernel, iterations=1)

    return imgE


def dilatar(image):
    # Operaciones morfológica
    kernel = np.ones((100, 100), np.uint8)
    # Dilatación
    imgD = cv2.dilate(image, kernel, iterations=1)

    return imgD


def count_skeleton_endpoints(skeleton):
    """
    Cuenta las esquinas en la imagen del esqueleto utilizando Harris Corner Detection.

    Args:
        skeleton (numpy.ndarray): Imagen del esqueleto con valores binarios (0 y 1).

    Returns:
        int: Número de esquinas detectadas.
    """

    # Verifica si la imagen está en RGB y conviértela a escala de grises
    if len(skeleton.shape) == 3 and skeleton.shape[2] == 3:
        gray = cv2.cvtColor(skeleton, cv2.COLOR_RGB2GRAY)
    else:

        gray = skeleton

    # Reduce ruido con un suavizado leve
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Asegúrate de convertir la imagen a un formato adecuado para Harris (grayscale con valores 0-255)
    gray = (gray * 255).astype(np.uint8)

    # Aplica cornerHarris
    blockSize = 2  # Tamaño del vecindario considerado para la detección
    ksize = 3  # Apertura del kernel de Sobel usado
    k = 0.04  # Parámetro libre en la ecuación de Harris
    dst = cv2.cornerHarris(gray, blockSize, ksize, k)

    # Normalizar la respuesta para obtener una mejor visualización
    dst = cv2.dilate(dst, None)  # Dilatar para marcar las esquinas
    corners_detected = (
        dst > 0.01 * dst.max()
    )  # Umbral para considerar un punto como esquina

    # Contar esquinas detectadas
    num_corners = np.sum(corners_detected)

    return num_corners


def remove_spurious_endpoints(skeleton, endpoints):
    """
    Filtra puntos espurios del mapa de endpoints.
    :param skeleton: Imagen binaria del esqueleto.
    :param endpoints: Mapa binario de endpoints detectados.
    :return: Mapa binario de endpoints filtrados.
    """
    # Filtrar puntos aislados que no pertenecen a ninguna rama
    valid_endpoints = np.zeros_like(endpoints)
    for y, x in zip(*np.where(endpoints)):
        # Vecindad 3x3 del punto actual
        neighbors = skeleton[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
        # Contar cuántos vecinos tiene el punto (excluyendo el propio)
        if (
            np.sum(neighbors) > 1
        ):  # Más de un vecino significa que pertenece a una rama válida
            valid_endpoints[y, x] = 1

    return valid_endpoints


def getSkeleton(image):

    # Convert the image to grayscale

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = image

    # Apply a threshold to get a binary image (First check background color)
    if gray[0, 0] == 0:
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # plt.imshow(image)
    # plt.show()

    def binn01(image):
        (n, m) = image.shape
        for f in range(0, n):
            for c in range(0, m):
                if image[f, c] == 255:
                    image[f, c] = 1
                else:
                    image[f, c]
        return image

    newImage = binn01(
        binary
    )  ##La función recibe imagenes con valores de 0 y 1, por tanto debemos proceder a reemplazar 255 por 1
    skeletonn = morphology.skeletonize(newImage)
    return skeletonn


def processImage(image):

    # if len(image.shape) == 3:  # Si tiene tres canales (imagen en color)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    erosion = erosionar(image)

    # plt.imshow(erosion)
    # plt.show()

    dilatacion = dilatar(erosion)

    # plt.imshow(dilatacion)
    # plt.show()

    esqueleto = getSkeleton(dilatacion)
    esquinas = count_skeleton_endpoints(esqueleto)

    return esqueleto, esquinas


# Extraer características y etiquetas de imágenes
def extract_features(image):

    cropped_img = crop_white_background(image)
    cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(cropped_gray, (64, 64))  # Tamaño estándar de 64x64
    flattened_img = resized_img.flatten()  # Aplanado en un vector

    vertical_profile = np.sum(resized_img, axis=1)
    horizontal_profile = np.sum(resized_img, axis=0)

    esqueleto, esquinas = processImage(resized_img)

    features = np.concatenate([flattened_img, vertical_profile, horizontal_profile])
    # features  = np.append(features, esquinas)
    # features = np.concatenate([flattened_img, vertical_profile, horizontal_profile])

    return features


def crop_white_background(image):

    # print("Cropping the image... with initial shape: ", image.shape)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to get a binary image (First check background color)
    if gray[0, 0] == 0:
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(
            gray, 200, 255, cv2.THRESH_BINARY_INV
        )  # It has to be inverted bebcause it has issues finding obbjects in a white background

    # Find the contours of the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No object detected")
        return image

    # ***************** This method works ok but can lead to errors
    # If the clothes are light so the person will end up being cropped too in different contours in some cases

    # for i, contour in enumerate(contours):
    #     print(f"Contour {i} area: ", cv2.contourArea(contour))

    # # Get the bounding box of the largest contour by area
    # largest_contour = max(contours, key=cv2.contourArea)

    # # Get the bounding box of the largest contour
    # x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image using the bounding box
    # cropped_img = image[y : y + h, x : x + w]

    # ************ If multiple contours are detected, we will crop the image using the bounding box that covers all contours
    # This way we can avoid cropping the person in the image

    # Get a bounding box that contains all contours
    x_min, y_min, w_max, h_max = float("inf"), float("inf"), 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min, y_min = min(x_min, x), min(y_min, y)
        w_max, h_max = max(w_max, x + w), max(h_max, y + h)

    # Crop the image using the bounding box that covers all contours
    cropped_img = image[y_min:h_max, x_min:w_max]

    # print("Cropped image shape: ", cropped_img.shape)

    return cropped_img


def getImages(folder_path):
    images = []

    # get all the images from each subfolder in the folder path for TEST and TRAINING
    for folder in os.listdir(folder_path):
        for image in os.listdir(folder_path + folder):
            img = cv2.imread(folder_path + folder + "/" + image)
            images.append(img)

    # return the images for training and testing
    return images


def load_images_and_labels(folder_path):
    images = []
    labels = []

    for pose_folder in os.listdir(folder_path):
        pose_path = os.path.join(folder_path, pose_folder)

        if not os.path.isdir(pose_path):
            continue

        for image_name in os.listdir(pose_path):
            image_path = os.path.join(pose_path, image_name)
            image = cv2.imread(image_path)

            if image is not None:
                images.append(image)
                labels.append(
                    pose_folder
                )  # Usa el nombre de la subcarpeta como etiqueta
            else:
                print(f"Error al cargar la imagen en {image_path}")

    return images, labels


def build_ann_model(input_dim):
    model = Sequential()
    # Capa de entrada
    model.add(Dense(128, input_dim=input_dim, activation="relu"))  # Primera capa
    model.add(Dense(64, input_dim=input_dim, activation="relu"))  # segunda capa
    model.add(Dense(32, input_dim=input_dim, activation="relu"))  # tercera capa

    # Capa de salida
    model.add(
        Dense(len(set(y_train)), activation="softmax")
    )  # Número de clases en y_train
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# create a machine learning model to classify and identify the yoga poses
# Cargar y preparar los datos de entrenamiento y prueba
print("STFJADHAS")
# print(
#     """
# ░██████╗░███████╗████████╗████████╗██╗███╗░░██╗░██████╗░  ██╗███╗░░░███╗░█████╗░░██████╗░███████╗░██████╗
# ██╔════╝░██╔════╝╚══██╔══╝╚══██╔══╝██║████╗░██║██╔════╝░  ██║████╗░████║██╔══██╗██╔════╝░██╔════╝██╔════╝
# ██║░░██╗░█████╗░░░░░██║░░░░░░██║░░░██║██╔██╗██║██║░░██╗░  ██║██╔████╔██║███████║██║░░██╗░█████╗░░╚█████╗░
# ██║░░╚██╗██╔══╝░░░░░██║░░░░░░██║░░░██║██║╚████║██║░░╚██╗  ██║██║╚██╔╝██║██╔══██║██║░░╚██╗██╔══╝░░░╚═══██╗
# ╚██████╔╝███████╗░░░██║░░░░░░██║░░░██║██║░╚███║╚██████╔╝  ██║██║░╚═╝░██║██║░░██║╚██████╔╝███████╗██████╔╝
# ░╚═════╝░╚══════╝░░░╚═╝░░░░░░╚═╝░░░╚═╝╚═╝░░╚══╝░╚═════╝░  ╚═╝╚═╝░░░░░╚═╝╚═╝░░╚═╝░╚═════╝░╚══════╝╚═════╝░"""
# )
label_encoder = LabelEncoder()

train_images, train_labels = load_images_and_labels("./data/TRAIN/")
train_labels = label_encoder.fit_transform(train_labels)
test_images, test_labels = load_images_and_labels("./data/TEST/")
test_labels = label_encoder.fit_transform(test_labels)
labels = list(set(train_labels).union(set(test_labels)))

# Codificador para las etiquetas, ya que tensorflow no acepta text

print("labels", labels)
X_train, y_train = [], []
X_test, y_test = [], []

print("Training images: ", len(train_images))
print("Test images: ", len(test_images))

for image in train_images:
    features = extract_features(image)
    X_train.append(features)

for image in test_images:
    features = extract_features(image)
    X_test.append(features)

X_train, y_train = np.array(X_train), np.array(train_labels)
X_test, y_test = np.array(X_test), np.array(test_labels)

# Aplicar PCA para reducción de dimensionalidad
pca = PCA(n_components=0.95)  # Retiene el 95% de la varianza
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# ------------------------------------- RF
# Entrenar el modelo con las características reducidas
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train_pca, y_train)
clf.fit(X_train, y_train)

# Evaluar el modelo
# y_pred = clf.predict(X_test_pca)
y_pred_rf = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy:.2f}")

# -------------------------------------- ANN
input_dim = X_train_pca.shape[1]  # Dimensión de entrada tras PCA
ann_model = build_ann_model(input_dim)
accs = []
ann_y_preds = []
for i in range(50):
    history = ann_model.fit(
        X_train_pca,
        y_train,
        epochs=100,
        batch_size=16,
        verbose=0,
        validation_data=(X_test_pca, y_test),
    )
    y_pred_ann = np.argmax(ann_model.predict(X_test_pca), axis=1)
    acc = accuracy_score(y_test, y_pred_ann)
    accs.append(acc)
    ann_y_preds.append(y_pred_ann)


y_pred_ann = ann_y_preds[accs.index(max(accs))]


# -------------------- Comparación de Resultados --------------------
# 1. Accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_ann = max(accs)

# 2. Kappa
kappa_rf = cohen_kappa_score(y_test, y_pred_rf)
kappa_ann = cohen_kappa_score(y_test, y_pred_ann)

# 3. Matriz de Confusión
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_ann = confusion_matrix(y_test, y_pred_ann)

# ------------------------ Resultados -------------------------
print("Resultados del Random Forest:")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Kappa: {kappa_rf:.2f}")
print("Matriz de Confusión:")
print(conf_matrix_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix_rf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)
plt.title(f"Matriz de Confusión - RF")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Verdadera")
plt.show()

print("\nResultados del ANN:")
print(f"Accuracy: {accuracy_ann:.2f}")
print(f"Kappa: {kappa_ann:.2f}")
print("Matriz de Confusión:")
print(conf_matrix_ann)
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix_ann,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)
plt.title(f"Matriz de Confusión - AN")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Verdadera")
plt.show()
