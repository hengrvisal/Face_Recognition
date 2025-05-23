from embedder import BaseEmbedder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks #type: ignore

class SupervisedTrainer(BaseEmbedder):
    def __init__(self, img_size, backbone_fn, n_classes, **kwargs):
        super().__init__(img_size, backbone_fn)
        self.n_classes = n_classes
        self.build_classification_head()

    def build_classification_head(self):
        x = self.embedding_model.output
        out = layers.Dense(self.n_classes, activation='softmax', name='softmax')(x)
        self.model = models.Model(self.embedding_model.input, out)
        
    def compile(self, lr=1e-4):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_ds, val_ds, steps_per_epoch, val_steps, epochs, ckpt_path):
        ckpt = callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss')
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        return self.model.fit(
            train_ds, validation_data=val_ds,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=[ckpt, early_stopping]
        )