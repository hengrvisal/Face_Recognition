import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models #type: ignore

class BaseEmbedder:
    def __init__(self, img_size, backbone_fn):
        self.img_size = img_size
        self.backbone = backbone_fn(input_shape=(*img_size, 3), pooling='avg')
        self.build_embedding_head()

    def build_embedding_head(self):
        """
        Attach Dense layer to embedding layer
        """

        x = layers.Dense(128, activation='relu', name='embed_dense')(self.backbone.output)
        self.embedding_model = models.Model(self.backbone.input, x)

    def get_embedding(self, img):
        """
        Run one image (preprocessed through the embedder)
        """
        return self.embedding_model.predict(tf.expand_dims(img, 0))[0]