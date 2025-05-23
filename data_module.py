import os
import glob
import random
import tensorflow as tf

class DataModule:
    def __init__(self, train_dir, img_size=(160, 160), batch_size=64, split=0.8):
        self.train_dir = train_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.split = split

        # discover classes
        self.class_names = sorted(
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        )

        # number of classes
        self.num_classes = len(self.class_names)
        # cerate a stringlookup layer for labels
        self.label_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.class_names,
            output_mode='int'
        )

        # map class to list of file paths
        all_paths = glob.glob(os.path.join(train_dir, '*', '*'))
        random.shuffle(all_paths)
        split_i = int(split * len(all_paths))
        self.train_paths = all_paths[:split_i]
        self.val_paths = all_paths[split_i:]


    def _parse(self, path):
        # label
        label_str = tf.strings.split(path, os.path.sep)[-2]
        idx = self.label_lookup(label_str) - 1
        # one-hot encode
        label = tf.one_hot(idx, depth=self.num_classes)
        # image
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        img = img / 255.0
        return img, label
    
    
    def get_supervised_datasets(self):
        # train
        train_ds = tf.data.Dataset.from_tensor_slices(self.train_paths)
        train_ds = (train_ds
                    .shuffle(10_000)
                    .map(self._parse, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(self.batch_size)
                    .prefetch(tf.data.AUTOTUNE))
        
        # val
        val_ds = tf.data.Dataset.from_tensor_slices(self.val_paths)
        val_ds = (val_ds
                  .map(self._parse, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(self.batch_size)
                  .prefetch(tf.data.AUTOTUNE))
        
        steps_per_epoch = len(self.train_paths) // self.batch_size
        validation_steps = len(self.val_paths) // self.batch_size
        return train_ds, val_ds, steps_per_epoch, validation_steps
    

    def get_triplet_dataset(self):
        # build class to paths mapping
        cls2paths = {}
        for p in self.train_paths:
            cls = os.path.basename(os.path.dirname(p))
            cls2paths.setdefault(cls, []).append(p)
        classes = list(cls2paths.keys())

        def gen():
            while True:
                a_cls = random.choice(classes)
                n_cls = random.choice(classes)
                while n_cls == a_cls:
                    n_cls = random.choice(classes)
                a = random.choice(cls2paths[a_cls])
                p = random.choice(cls2paths[a_cls])
                n = random.choice(cls2paths[n_cls])
                yield a, p, n

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
            )
        )

        def _parse_triplet(a, p, n):
            ia, _ = self._parse(a)
            ip, _ = self._parse(p)
            in_, _ = self._parse(n)

            return (ia, ip, in_)
        
        ds = (ds
              .map(_parse_triplet, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(self.batch_size)
              .prefetch(tf.data.AUTOTUNE))
        
        return ds