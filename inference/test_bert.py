import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import os

# Load the trained model and tokenizer
model_path = "/home/swleocresearch/Desktop/triage-ai/train/best_model_ger"  # Update this path if needed
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model directory '{model_path}' does not exist.")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    raise

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Function to preprocess the input text
def preprocess_text(text):
    # Tokenize and encode the text
    encoding = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    return encoding

# Function to make predictions with custom thresholds
def predict_with_threshold(text, discharge_threshold=0.5, surgery_threshold=0.3):
    inputs = preprocess_text(text)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        discharge_prob = probabilities[0][0].item()
        surgery_prob = probabilities[0][1].item()

    # Print probabilities to debug
    print(f"Discharge prob: {discharge_prob:.4f}, Surgery prob: {surgery_prob:.4f}")

    # Apply the thresholds
    if surgery_prob > surgery_threshold:
        return "surgery", surgery_prob
    else:
        return "discharge", discharge_prob  # Default to discharge if surgery is not confident enough

# Example usage
if __name__ == "__main__":
    # Load the test set from CSV file
    df = pd.read_csv('test_set.csv')

    # Ensure that 'text' and 'label' columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("The 'text' and 'label' columns are required in the test set CSV file.")

    test_texts = df['text'].tolist()
    test_labels = df['label'].tolist()

    # Assuming you have a mapping between labels and their string representations
    label_mapping = {0: "discharge", 1: "surgery"}

    # Convert actual_labels to strings using the label mapping
    actual_labels = [label_mapping[label] if isinstance(label, int) else label for label in test_labels]

    # Continue with predictions, which should already be in string form
    predictions = []

    for text, actual_label in zip(test_texts, actual_labels):
        predicted_label, confidence = predict_with_threshold(text)
        predictions.append(predicted_label)
        
        print(f"Text: {text[:100]}...")  # Print first 100 characters of text
        print(f"Actual: {actual_label}")
        print(f"Predicted: {predicted_label}")
        print(f"Confidence: {confidence:.4f}")
        print("---")

    # Now both actual_labels and predictions should be strings and you can safely generate the classification report
    print("\nClassification Report:")
    print(classification_report(actual_labels, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual_labels, predictions))


    # Calculate overall accuracy
    accuracy = sum([1 for a, p in zip(actual_labels, predictions) if a == p]) / len(actual_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
