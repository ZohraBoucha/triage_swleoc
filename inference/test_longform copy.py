import pandas as pd
import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np


# Set max token length
MAX_LENGTH = 4096

# Load the test dataset
df = pd.read_csv('/home/swleocresearch/Desktop/triage-ai/datasets/triage_dataset_test.csv')

# Filter the DataFrame to include only the desired labels (if necessary)
df = df[df['output'].isin(['surgery', 'discharge', 'injection', 'fu', 'physio'])]

# Load the saved model and tokenizer
model_path = '/home/swleocresearch/Desktop/triage-ai/train/medical_longformer'
model = LongformerForSequenceClassification.from_pretrained(model_path)
tokenizer = LongformerTokenizer.from_pretrained(model_path)

# Prepare the test data
texts = df.iloc[:, 1].tolist()  # Assuming the text is in the second column
labels = df.iloc[:, 2].tolist()  # Assuming the label is in the third column

# # Create a label mapping (string to integer)
# label_mapping = {'discharge': 0, 'surgery': 1}  # Adjust if your mapping is different
# inv_label_mapping = {v: k for k, v in label_mapping.items()}

# Get all unique labels from your dataset
all_labels = sorted(df['output'].unique())

# Create label_mapping dynamically
label_mapping = {label: idx for idx, label in enumerate(all_labels)}
inv_label_mapping = {v: k for k, v in label_mapping.items()}

# Convert string labels to integer labels
y_true = [label_mapping[label] for label in labels]

# Tokenization function
def tokenize_function(texts):
    return tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

# Tokenize test data
encodings = tokenize_function(texts)

# Set global attention on CLS token
global_attention_mask = torch.zeros_like(encodings['attention_mask'])
global_attention_mask[:, 0] = 1

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=encodings['input_ids'],
        attention_mask=encodings['attention_mask'],
        global_attention_mask=global_attention_mask
    )
    predictions = torch.argmax(outputs.logits, dim=-1)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

y_pred = predictions.tolist()
# y_pred_proba = probabilities[:, 1].tolist()
y_pred_proba = probabilities.tolist() 

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
try:
    # roc_auc = roc_auc_score(y_true, outputs.logits[:, 1].tolist())
    # roc_auc = roc_auc_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
except ValueError:
    roc_auc = 0  # In case of single class prediction

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Create a confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=all_labels,
            yticklabels=all_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('test_confusion_matrix.png')
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=all_labels))

# Calculate and print the distribution of predictions
pred_distribution = pd.Series(y_pred).map(inv_label_mapping).value_counts()
true_distribution = pd.Series(y_true).map(inv_label_mapping).value_counts()

print("\nPrediction Distribution:")
print(pred_distribution)
print("\nTrue Distribution:")
print(true_distribution)

# # Calculate ROC curve and AUC
# fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.figure(figsize=(10, 7))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.savefig('roc_curve.png')
# plt.close()

# Binarize the output
y_true_bin = label_binarize(y_true, classes=range(len(all_labels)))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(all_labels)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(y_pred_proba)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 7))
colors = ['blue', 'red', 'green', 'yellow', 'purple']
for i, color in zip(range(len(all_labels)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {all_labels[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()