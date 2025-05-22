
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
tf.keras.backend.clear_session()


TRAIN_DIR = 'classification_data/train_data'
VAL_DIR = 'classification_data/val_data'
TEST_DIR = 'classification_data/test_data'
VERIF_FILE = 'verification_data/verification_pairs_val.txt'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
CHECKPOINT_PATH = 'resnet_face_class.h5'


def get_classification_gens(directory, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Returns training and validation generators from a directory with subfolders per class and a 20% validation split.
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen


def build_classification_model(num_classes, input_shape=(*IMG_SIZE, 3)):
    """
    Builts a ResNet50-based model for face classification. Embeddings can be extracted from the penultimate Dense layer
    """
    base = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    x = layers.Dense(512, activation='relu', name='embed_dense')(base.output)
    outputs = layers.Dense(num_classes, activation='softmax', name='softmax')(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    return model


def train_classification_model(model, train_gen, val_gen, epochs=EPOCHS):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    ckpt = callbacks.ModelCheckpoint(
        CHECKPOINT_PATH,
        save_best_only=True,
        monitor='val_loss'
    )

    early = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[ckpt, early]
    )

    return history


def load_verification_pairs(txt_path):
    """
    Reads verification trial pairs and labels (0/1) from text file.
    """

    pairs, labels = [], []
    with open(txt_path, 'r') as f:
        for line in f:
            p1, p2, lbl = line.strip().split()
            pairs.append((p1, p2))
            labels.append(int(lbl))
    
    return pairs, np.array(labels, dtype=int)


def get_embedding_model(model):
    """
    Returns a model up to the embedding layer.
    """
    return tf.keras.Model(inputs=model.input,
                          outputs=model.get_layer('embed_dense').output)


def extract_embedding(embed_model, img_path, img_size=IMG_SIZE):
    """
    Loads an image, preprocesses, and returns its embedding vector.
    """
    img = image.load_img(img_path, target_size=img_size)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return embed_model.predict(arr)[0]


def evaluate_verification(model, pairs, labels):
    """
    Computes embeddings for each pair, then ROC/AUC/ for both Euclidean distance and cosine similarity
    """
    embed_model = get_embedding_model(model)
    embs1, embs2 = [], []
    for p1, p2 in pairs:
        embs1.append(extract_embedding(embed_model, p1))
        embs2.append(extract_embedding(embed_model, p2))
    embs1, embs2 = np.array(embs1), np.array(embs2)

    # Euclidean distances: similarity by negation
    dists = np.linalg.norm(embs1 - embs2, axis=1)
    fpr_e, tpr_e, _ = roc_curve(labels, -dists)
    auc_e = auc(fpr_e, tpr_e)

    # Cosine similarity
    cos_sims = np.array([cosine_similarity([e1], [e2])[0][0]
                         for e1, e2 in zip(embs1, embs2)])
    fpr_c, tpr_c, _ = roc_curve(labels, cos_sims)
    auc_c = auc(fpr_c, tpr_c)

    # Plot ROC curves
    plt.figure()
    plt.plot(fpr_e, tpr_e, label=f'Euclidean AUC={auc_e:.3f}')
    plt.plot(fpr_c, tpr_c, label=f'Cosine AUC={auc_c:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('roc_curve.png')

    print(f'Euclidean AUC: {auc_e:.3f}')
    print(f'Cosine AUC: {auc_c:.3f}')


if __name__ == '__main__':
    # Train classification model
    train_gen, val_gen = get_classification_gens(TRAIN_DIR)
    num_classes = len(train_gen.class_indices)
    model = build_classification_model(num_classes)
    history = train_classification_model(model, train_gen, val_gen)

    # Evaluate on verification pairs
    model.load_weights(CHECKPOINT_PATH)
    pairs, labels = load_verification_pairs(VERIF_FILE)
    evaluate_verification(model, pairs, labels)