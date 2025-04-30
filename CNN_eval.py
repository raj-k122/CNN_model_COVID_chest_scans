from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

model = load_model("covid_classifier_cnn.h5")

img_size = (128, 128)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

pred_probs = model.predict(test_generator)
pred_labels = (pred_probs > 0.5).astype(int).ravel()

true_labels = test_generator.classes
idx_to_class = {v: k for k, v in test_generator.class_indices.items()}
target_names = [idx_to_class[i] for i in sorted(idx_to_class)]

report = classification_report(true_labels, pred_labels, target_names=target_names)
print(report)


cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Confusion Matrix')
plt.tight_layout()
plt.savefig('results/test_confusion_matrix.png')
plt.close()
