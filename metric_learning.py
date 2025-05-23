from embedder import BaseEmbedder

import tensorflow as tf
from tensorflow.keras import layers, callbacks, models #type: ignore

class MetricLearningTrainer(BaseEmbedder):
    def __init__(self, img_size, backbone_fn, margin=0.2):
        super().__init__(img_size, backbone_fn)
        self.margin = margin
        self.built_triplet_model()
    
    def build_triplet_model(self):
        inp_a = layers.Input((*self.img_size,3), name='anchor')
        inp_p = layers.Input((*self.img_size,3), name='positive')
        inp_n = layers.Input((*self.img_size,3), name='negative')
        ae = self.embedding_model(inp_a)
        pe = self.embedding_model(inp_p)
        ne = self.embedding_model(inp_n)
        self.triplet_model = models.Model(inputs=[inp_a, inp_p, inp_n],
                                          outputs=[ae, pe, ne])
        
    def triplet_loss(self, y_true, y_pred):
        # y_pred is a list [ae, pe, ne]
        ae, pe, ne = y_pred
        pos = tf.reduce_sum(tf.square(ae - pe), axis=1)
        neg = tf.reduce_sum(tf.square(ae - ne), axis=1)
        return tf.reduce_sum(tf.maximum(pos - neg + self.margin, 0.0))
    
    def compile(self, lr=1e-4):
        # since keras expect one per output, we need a dummy loss
        self.triplet_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=self.triplet_loss)

    def train(self, triplet_ds, steps_per_epoch, epochs, ckpt_path):
        ckpt = callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, save_best_only=True, monitor='loss')
        return self.triplet_model.fit(
            triplet_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[ckpt]
        )