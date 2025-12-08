import tensorflow as tf
from pathlib import Path

class BeverageClassifier:
    def __init__(self, img_height: int = 160, img_width: int = 160, batch_size: int = 32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.class_names = ['cola', 'orange_juice', 'water']

    def create_dataset(self, data_dir: Path):
        """Loads and configures the dataset from the directory."""
        return tf.keras.utils.image_dataset_from_directory(
            data_dir,
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

    def _get_augmentation_layer(self):
        """Creates a data augmentation sequential model to reduce overfitting."""
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ])

    def build_model(self):
        """Builds the MobileNetV2 model with transfer learning and data augmentation."""
        # Download MobileNetV2 pre-trained on ImageNet
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Freeze base model to preserve learned features

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        
        # Initialize data augmentation
        data_augmentation = self._get_augmentation_layer()

        inputs = tf.keras.Input(shape=(self.img_height, self.img_width, 3))
        
        # 1. Apply Data Augmentation (Only active during training phase)
        x = data_augmentation(inputs)
        
        # 2. Preprocess input (Specific to MobileNetV2)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        
        # 3. Pass through the base model and classifier
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)

        self.model = tf.keras.Model(inputs, outputs)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # Slightly adjusted LR
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

    def train(self, train_ds, val_ds, epochs: int = 25):
        """Trains the model with performance optimizations."""
        if self.model is None:
            self.build_model()
        
        # Performance optimization: cache and prefetch data
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

    def save_model(self, filepath: Path):
        """Saves the trained model to the specified path."""
        if self.model:
            self.model.save(filepath)