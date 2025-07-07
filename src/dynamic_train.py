import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

def load_data(dataset_path='dynamic_dataset'):
    """
    Load and process all sequence files from the dataset
    """
    # Load actions from file if it exists
    actions_file = os.path.join(dataset_path, 'actions.txt')
    if os.path.exists(actions_file):
        with open(actions_file, 'r') as f:
            actions = f.read().strip().split('\n')
    else:
        # Fallback to loading from filenames
        sequence_files = [f for f in os.listdir(dataset_path) if f.startswith('seq_')]
        actions = sorted(list(set([f.split('_')[1] for f in sequence_files])))
    
    print(f"Found {len(actions)} actions: {actions}")
    
    # Load all sequence files
    data = []
    labels = []
    sequence_files = [f for f in os.listdir(dataset_path) if f.startswith('seq_')]
    
    for seq_file in sequence_files:
        action = seq_file.split('_')[1]
        action_idx = actions.index(action)
        
        # Load sequence data
        seq_data = np.load(os.path.join(dataset_path, seq_file))
        print(f"Loaded {seq_file}: {seq_data.shape}")
        
        # Extract features and correct labels
        features = seq_data[:, :, :-1]  # All frames, all features except the last one
        data.append(features)
        
        # Create correct labels
        n_sequences = len(seq_data)
        labels.extend([action_idx] * n_sequences)
    
    # Combine all data
    X = np.concatenate(data, axis=0)
    y = np.array(labels)
    
    print("\nFinal data shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y, actions

def create_model(input_shape, num_classes):
    """
    Create and compile the LSTM model
    """
    model = Sequential([
        LSTM(64, activation='tanh',
             return_sequences=True,
             input_shape=input_shape,
             kernel_regularizer=l2(0.01),
             recurrent_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, activation='tanh',
             kernel_regularizer=l2(0.01),
             recurrent_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu',
              kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X, y, actions, model_dir='models'):
    """
    Train the model with proper validation and callbacks
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Convert labels to categorical
    y_categorical = to_categorical(y, num_classes=len(actions))
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical,
        test_size=0.2,
        random_state=42,
        stratify=y  # Ensure balanced split
    )
    
    # Print class distribution
    print("\nClass distribution in training set:")
    for i, action in enumerate(actions):
        count = np.sum(np.argmax(y_train, axis=1) == i)
        print(f"{action}: {count}")
    
    # Create model
    model = create_model(input_shape=(X.shape[1], X.shape[2]), num_classes=len(actions))
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def main():
    # Load and process data
    X, y, actions = load_data()
    
    # Train model
    model, history = train_model(X, y, actions)
    
    # Final evaluation
    print("\nEvaluating best model on validation set:")
    best_model = tf.keras.models.load_model('models/best_model.keras')
    
    # Split data again to get the same validation set
    y_categorical = to_categorical(y, num_classes=len(actions))
    _, X_val, _, y_val = train_test_split(
        X, y_categorical,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    evaluation = best_model.evaluate(X_val, y_val, verbose=1)
    print(f"\nValidation Loss: {evaluation[0]:.4f}")
    print(f"Validation Accuracy: {evaluation[1]:.4f}")

if __name__ == "__main__":
    main()