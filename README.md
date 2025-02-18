# CNN-Based Spam Detection

## Overview
This project implements a **Spam Detection System** using **1D Convolutional Neural Networks (CNNs)**. The system classifies SMS messages as **spam** or **ham (not spam)** using deep learning techniques.

## Features
- Preprocessing and tokenization of text messages.
- CNN-based neural network for text classification.
- Training and evaluation of the model using accuracy metrics.
- Performance visualization with **matplotlib**.
- Spam prediction on new messages.

## Dataset
The project uses the **SMS Spam Collection** dataset. It contains:
- `v1`: Label (`ham` or `spam`)
- `v2`: SMS message text

You can download it from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

## Installation
Ensure you have Python installed (>=3.8). To install dependencies, run:
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
```

## Usage
### Running the Jupyter Notebook
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cnn-spam-detection.git
   cd cnn-spam-detection
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `cnn_spam_detection.ipynb` and run all cells sequentially.

### Training the Model
The model is trained using an **80-20 train-test split**. You can modify hyperparameters such as:
- `max_words = 10000` (Vocabulary size)
- `max_len = 100` (Max sequence length)
- `epochs = 5` (Training epochs)
- `batch_size = 32`

## Model Architecture
- **Embedding Layer**: Converts words into dense vectors.
- **Conv1D + ReLU**: Extracts features from text sequences.
- **Global Max Pooling**: Reduces dimensionality.
- **Dense + Dropout Layers**: Enhance model generalization.
- **Output Layer (Sigmoid Activation)**: Predicts spam probability.

## Evaluation
Evaluate performance using:
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

## Predicting on New Messages
You can test the model with new messages:
```python
def predict_spam(message):
    sequence = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    return "Spam" if prediction > 0.5 else "Ham"
```
Example usage:
```python
messages = ["Congratulations! You've won a free ticket to Bahamas. Call now!",
            "Hey, are we still meeting for lunch today?"]
for msg in messages:
    print(f"Message: {msg} -> Prediction: {predict_spam(msg)}")
```

## Results & Visualization
The training progress is visualized using **matplotlib**:
```python
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## Contributing
Feel free to contribute! Fork the repository, create a branch, and submit a **Pull Request**.

## License
This project is open-source under the **MIT License**.

## Contact
For questions, reach out via [GitHub Issues](https://github.com/yourusername/cnn-spam-detection/issues).

