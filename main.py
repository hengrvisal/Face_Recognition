from supervised_learning import SupervisedTrainer
from metric_learning import MetricLearningTrainer
from data_module import DataModule

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import models # type: ignore
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity


# _____
# configuration
# -----

TRAIN_DIR = 'classification_data/train_data'
VERIF_FILE = 'verification_pairs_val.txt' # CHECK IN 'verification_data' IF ERROR
IMG_SIZE = (160, 160)
BATCH_SIZE = 64
EPOCHS = 50
CHECKPOINT_SUP = 'supervised_learning_face_class.keras'
CHECKPOINT_MET = 'metrics_learning_face_class.keras'

tf.keras.backend.clear_session()


# ------
# verification evaluation
# ------

def load_verificaiton_pairs(fp):
    pairs, lbls = [], []
    with open(fp, 'r') as f:
        for line in f:
            p1, p2, l = line.strip().split()
            pairs.append((p1, p2)); lbls.append(int(l))
    
    return pairs, np.array(lbls, dtype=int)


def get_embedding_model(trained_model):
    return models.Model(inputs=trained_model.input,
                        outputs=trained_model.get_layer('embed_sense').output)


def evaluate(model, pairs, labels, name):
    emb_model = get_embedding_model(model)
    embs1 = np.array([emb_model.predict(
        tf.expand_dims(tf.image.resize(
            tf.image.decode_jpeg(tf.io.read_file(p1), 3),
            IMG_SIZE)/255.0,0))[0] for p1,_ in pairs])
    
    embs2 = np.array([emb_model.predict(
        tf.expand_dims(tf.image.resize(
            tf.image.decode_jpeg(tf.io.read_file(p2), 3),
            IMG_SIZE)/255.0,0))[0] for _,p2 in pairs])
        
    dists = np.linalg.norm(embs1 - embs2, axis=1)
    cos_sims = np.array([cosine_similarity([e1],[e2])[0,0]
                         for e1, e2 in zip(embs1, embs2)])
    
    fpr_e, tpr_e, _ = roc_curve(labels, -dists)
    fpr_c, tpr_c, _ = roc_curve(labels, cos_sims)
    auc_e, auc_c = auc(fpr_e, tpr_e), auc(fpr_c, tpr_c)
    print(f"{name} -> Euclid AUC={auc_e:.3f}, Cosine AUD={auc_c:.3f}")


if __name__ == '__main__':
    dm = DataModule(TRAIN_DIR, IMG_SIZE, BATCH_SIZE)
    train_ds, val_ds, steps, vsteps = dm.get_supervised_datasets()
    triplet_ds = dm.get_triplet_dataset()

    # supervised branch
    sup = SupervisedTrainer(IMG_SIZE, tf.keras.applications.MobileNetV3, dm.num_classes)
    sup.compile(lr=1e-4)
    sup.train(train_ds, val_ds, steps, vsteps, EPOCHS, CHECKPOINT_SUP)
    sup.model.load_weights(CHECKPOINT_SUP)

    # metric learning branch
    met = MetricLearningTrainer(IMG_SIZE, tf.keras.applications.MobileNetV3, margin=0.2)
    met.compile(lr=1e-4)
    met.train(triplet_ds, steps, EPOCHS, CHECKPOINT_MET)
    met.model.load_weights(CHECKPOINT_MET)

    # evaluation
    pairs, labels = load_verificaiton_pairs(VERIF_FILE)
    evaluate(sup.model, pairs, labels, name='Supervised-Learning')
    evaluate(sup.model, pairs, labels, name='Metrics-Learning')