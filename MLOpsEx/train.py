# train.py
# MLOps Workshop Starter Code
# Sentiment Analysis Model using DistilBERT

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import time


class SentimentModel:
    """Simple sentiment analysis model wrapper"""

    def __init__(self):
        # Using a small pretrained model from Hugging Face
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def predict(self, text):
        """Make a prediction on input text"""
        start_time = time.time()
        result = self.pipeline(text)[0]
        latency = time.time() - start_time

        return {
            "text": text,
            "label": result["label"],
            "score": result["score"],
            "latency": latency,
            "timestamp": datetime.now().isoformat()
        }

    def predict_batch(self, texts):
        """Batch prediction"""
        results = self.pipeline(texts)
        return results


# Sample test data
TEST_SAMPLES = [
    {"text": "This movie was absolutely fantastic!", "expected_label": "POSITIVE"},
    {"text": "I loved every minute of it", "expected_label": "POSITIVE"},
    {"text": "Best film I've seen this year", "expected_label": "POSITIVE"},
    {"text": "This was terrible and boring", "expected_label": "NEGATIVE"},
    {"text": "Waste of time, don't watch", "expected_label": "NEGATIVE"},
    {"text": "Awful experience, very disappointed", "expected_label": "NEGATIVE"},
    {"text": "Pretty good, enjoyed it", "expected_label": "POSITIVE"},
    {"text": "Not bad, worth watching", "expected_label": "POSITIVE"},
]


def basic_evaluation(model, test_samples):
    """Basic evaluation function"""
    print("Running basic evaluation...")
    correct = 0
    total = len(test_samples)

    for sample in test_samples:
        prediction = model.predict(sample["text"])
        if prediction["label"] == sample["expected_label"]:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy


def main():
    """Main execution function"""
    print("=" * 50)
    print("MLOps Workshop - Sentiment Analysis Model")
    print("=" * 50)

    # Initialize model
    print("\nLoading model...")
    model = SentimentModel()
    print("Model loaded successfully!")

    # Run basic evaluation
    print("\n" + "=" * 50)
    accuracy = basic_evaluation(model, TEST_SAMPLES)

    # Example predictions
    print("\n" + "=" * 50)
    print("Example Predictions:")
    print("=" * 50)

    example_texts = [
        "This is amazing!",
        "I hate this so much",
        "It's okay, nothing special"
    ]

    for text in example_texts:
        result = model.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['label']} (confidence: {result['score']:.3f})")
        print(f"Latency: {result['latency']:.3f}s")

    print("\n" + "=" * 50)
    print("Basic model working! Now implement MLOps practices:")
    print("1. Add experiment tracking (MLflow/W&B)")
    print("2. Version the model and data (DVC/Git)")
    print("3. Containerize with Docker")
    print("4. Create CI/CD pipeline (GitHub Actions)")
    print("5. Deploy as API (FastAPI/Flask)")
    print("6. Add monitoring and logging")
    print("=" * 50)

    # Save test data
    import json
    import os
    os.makedirs('data', exist_ok=True)
    with open('data/test_samples.json', 'w') as f:
        json.dump(TEST_SAMPLES, f, indent=2)

    print("\n" + "=" * 50)
    print("Basic model working! Now implement MLOps practices:")


if __name__ == "__main__":
    main()
