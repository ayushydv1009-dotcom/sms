import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re, io, zipfile, requests, pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── Hyperparameters ──────────────────────────────────────────────
MAX_VOCAB  = 10000   # Vocabulary size
MAX_LEN    = 100     # Pad/truncate all sequences to this length
EMBED_DIM  = 64
LSTM_UNITS = 64
EPOCHS     = 8
BATCH_SIZE = 64

def clean_text(text):
    """Lowercase, strip special characters and extra whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    # ── 1. Data Loading ─────────────────────────────────────────
    print("Downloading SMS Spam Collection dataset …")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    with z.open('SMSSpamCollection') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=['label', 'message'])

    print(f'Dataset shape : {df.shape}')
    print(f'Label counts  :\n{df["label"].value_counts()}')

    # ── 2. Preprocessing ────────────────────────────────────────
    df['label_enc'] = (df['label'] == 'spam').astype(int)
    df['clean_msg'] = df['message'].apply(clean_text)

    print('\nSample cleaned messages:')
    print(df[['message', 'clean_msg', 'label_enc']].head(4))

    # ── 3. Tokenisation & Padding ────────────────────────────────
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['clean_msg'])

    sequences = tokenizer.texts_to_sequences(df['clean_msg'])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    y = df['label_enc'].values

    print(f'\nVocabulary size : {len(tokenizer.word_index)}')
    print(f'X shape         : {X.shape}')

    # ── 4. Train–Test Split ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')

    # ── 5. Model Architecture — Bidirectional LSTM ───────────────
    print("\nBuilding model architecture …")
    model = keras.Sequential([
        layers.Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN, name='embedding'),
        layers.Bidirectional(
            layers.LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
            name='bilstm'
        ),
        layers.Dense(32, activation='relu', name='dense1'),
        layers.Dropout(0.4, name='dropout'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='RNN_SpamDetector')

    model.summary()

    # ── 6. Compilation & Training ────────────────────────────────
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    print("\nTraining the model …")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # ── 7. Save Model & Tokenizer ────────────────────────────────
    model.save('spam_model.h5')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("\nModel saved  → spam_model.h5")
    print("Tokenizer saved → tokenizer.pkl")

    # ── 8. Evaluation ────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest Loss     : {test_loss:.4f}')
    print(f'Test Accuracy : {test_acc * 100:.2f}%')

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # ── 9. Visualisation — Training Curves ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(axes,
                                   [('accuracy', 'val_accuracy'), ('loss', 'val_loss')],
                                   ['Accuracy', 'Loss']):
        ax.plot(history.history[metric[0]], label='Train', linewidth=2)
        ax.plot(history.history[metric[1]], label='Val',   linewidth=2, linestyle='--')
        ax.set_title(f'RNN LSTM — {title}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', bbox_inches='tight')
    plt.close()

    # ── 10. Visualisation — Confusion Matrix ─────────────────────
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Ham', 'Spam'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap='Reds', colorbar=False)
    ax.set_title('RNN — Confusion Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()

    print("\nVisualizations saved: 'training_curves.png', 'confusion_matrix.png'")

if __name__ == '__main__':
    main()
