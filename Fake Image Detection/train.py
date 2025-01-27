import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers

# Importation des données
train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = "Data/Train",
    image_size = (256,256),
    batch_size = 32
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = "Data/Validation",
    image_size = (256,256),
    batch_size = 32
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = "Data/Test",
    image_size = (256,256),
    batch_size = 32
)

# Prétraitement
class_names = train_dataset.class_names
"""
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))
"""
train_dataset = train_dataset.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)




taille_image = 256
model = models.Sequential([
    layers.Input(shape=(taille_image,taille_image,3)),
    layers.Conv2D(32,(3,3),activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPool2D((2,2)),

    layers.Flatten(),
    layers.Dense(64,activation="relu"),
    layers.Dense(2,activation="softmax")
])

model.summary()

model.compile(
    optimizer="adam",
    loss= "sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5,
    verbose=1
)
Image = 1
model.save(f"Model_{Image}.keras")
"""
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[tf.argmax(labels[i])])
        plt.axis("off")

plt.show()       
"""