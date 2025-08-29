"""
Deep Learning Models for Customer Churn Prediction
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from config import *
from data_preprocessing import DataPreprocessor

class ChurnNeuralNetwork(nn.Module):
    """Feedforward Neural Network for churn prediction"""
    
    def __init__(self, input_size, hidden_layers, dropout_rate=0.3):
        super(ChurnNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ChurnLSTM(nn.Module):
    """LSTM Model for sequential churn prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.3):
        super(ChurnLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out

class DeepLearningChurnPredictor:
    """Deep Learning models for churn prediction"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.neural_net = None
        self.lstm_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_data(self):
        """Load and prepare data for deep learning"""
        print("Loading data for deep learning...")
        
        # Load preprocessor
        self.preprocessor.load_preprocessor()
        
        # Load processed data
        df = pd.read_csv(PROCESSED_DATA_PATH)
        
        # Split features and target
        X = df[self.preprocessor.feature_columns].values
        y = df[self.preprocessor.target_column].values
        
        # Split train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_data_loaders(self, X_train, X_test, y_train, y_test, batch_size=32):
        """Create PyTorch data loaders"""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train feedforward neural network"""
        print("\nTraining Neural Network...")
        
        # Model configuration
        input_size = X_train.shape[1]
        hidden_layers = NEURAL_NET_CONFIG['hidden_layers']
        dropout_rate = NEURAL_NET_CONFIG['dropout_rate']
        learning_rate = NEURAL_NET_CONFIG['learning_rate']
        epochs = NEURAL_NET_CONFIG['epochs']
        batch_size = NEURAL_NET_CONFIG['batch_size']
        
        # Create model
        self.neural_net = ChurnNeuralNetwork(input_size, hidden_layers, dropout_rate)
        self.neural_net.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.neural_net.parameters(), lr=learning_rate)
        
        # Create data loaders
        train_loader, test_loader, _, X_test_tensor, _, y_test_tensor = self.create_data_loaders(
            X_train, X_test, y_train, y_test, batch_size
        )
        
        # Training loop
        train_losses = []
        test_losses = []
        test_aucs = []
        
        for epoch in range(epochs):
            # Training
            self.neural_net.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.neural_net(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.neural_net.eval()
            test_loss = 0.0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.neural_net(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
                    
                    all_predictions.extend(outputs.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            test_auc = roc_auc_score(all_targets, all_predictions)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_aucs.append(test_auc)
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}')
        
        # Save model
        torch.save(self.neural_net.state_dict(), NEURAL_NET_PATH)
        print(f"Neural Network model saved to {NEURAL_NET_PATH}")
        
        # Plot training history
        self.plot_training_history(train_losses, test_losses, test_aucs, 'Neural Network')
        
        return train_losses, test_losses, test_aucs
    
    def create_sequential_data(self, X, y, sequence_length=12):
        """Create sequential data for LSTM"""
        sequences = []
        targets = []
        
        # Group by customer (assuming customers are in order)
        # For simplicity, we'll create sequences by sliding window
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i+sequence_length])
            targets.append(y[i+sequence_length-1])
        
        return np.array(sequences), np.array(targets)
    
    def train_lstm_model(self, X_train, y_train, X_test, y_test):
        """Train LSTM model for sequential churn prediction"""
        print("\nTraining LSTM Model...")
        
        # Model configuration
        sequence_length = LSTM_CONFIG['sequence_length']
        hidden_size = LSTM_CONFIG['hidden_size']
        num_layers = LSTM_CONFIG['num_layers']
        dropout_rate = LSTM_CONFIG['dropout_rate']
        learning_rate = LSTM_CONFIG['learning_rate']
        epochs = LSTM_CONFIG['epochs']
        batch_size = LSTM_CONFIG['batch_size']
        
        # Create sequential data
        print("Creating sequential data...")
        X_train_seq, y_train_seq = self.create_sequential_data(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = self.create_sequential_data(X_test, y_test, sequence_length)
        
        print(f"Sequential training data: {X_train_seq.shape}")
        print(f"Sequential test data: {X_test_seq.shape}")
        
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print("Not enough data for sequential modeling. Skipping LSTM training.")
            return None, None, None
        
        # Create model
        input_size = X_train_seq.shape[2]
        self.lstm_model = ChurnLSTM(input_size, hidden_size, num_layers, dropout_rate)
        self.lstm_model.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=learning_rate)
        
        # Create data loaders
        train_loader, test_loader, _, _, _, _ = self.create_data_loaders(
            X_train_seq, X_test_seq, y_train_seq, y_test_seq, batch_size
        )
        
        # Training loop
        train_losses = []
        test_losses = []
        test_aucs = []
        
        for epoch in range(epochs):
            # Training
            self.lstm_model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.lstm_model.eval()
            test_loss = 0.0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.lstm_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
                    
                    all_predictions.extend(outputs.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            test_auc = roc_auc_score(all_targets, all_predictions)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_aucs.append(test_auc)
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}')
        
        # Save model
        torch.save(self.lstm_model.state_dict(), LSTM_MODEL_PATH)
        print(f"LSTM model saved to {LSTM_MODEL_PATH}")
        
        # Plot training history
        self.plot_training_history(train_losses, test_losses, test_aucs, 'LSTM')
        
        return train_losses, test_losses, test_aucs
    
    def plot_training_history(self, train_losses, test_losses, test_aucs, model_name):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(train_losses, label='Training Loss')
        axes[0].plot(test_losses, label='Test Loss')
        axes[0].set_title(f'{model_name} - Loss History')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC plot
        axes[1].plot(test_aucs, label='Test AUC', color='green')
        axes[1].set_title(f'{model_name} - AUC History')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'{model_name.lower().replace(" ", "_")}_training_history.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_deep_learning_models(self, X_test, y_test):
        """Evaluate deep learning models"""
        print("\n" + "=" * 60)
        print("EVALUATING DEEP LEARNING MODELS")
        print("=" * 60)
        
        results = {}
        
        # Evaluate Neural Network
        if self.neural_net is not None:
            print("\nEvaluating Neural Network...")
            self.neural_net.eval()
            
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            
            with torch.no_grad():
                nn_predictions = self.neural_net(X_test_tensor).cpu().numpy()
            
            nn_pred_binary = (nn_predictions > 0.5).astype(int)
            nn_auc = roc_auc_score(y_test, nn_predictions)
            nn_f1 = f1_score(y_test, nn_pred_binary)
            
            print(f"Neural Network - AUC: {nn_auc:.4f}, F1: {nn_f1:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, nn_pred_binary))
            
            results['neural_network'] = {
                'auc': nn_auc,
                'f1': nn_f1,
                'predictions': nn_predictions
            }
        
        # Evaluate LSTM
        if self.lstm_model is not None:
            print("\nEvaluating LSTM Model...")
            
            # Create sequential test data
            sequence_length = LSTM_CONFIG['sequence_length']
            X_test_seq, y_test_seq = self.create_sequential_data(X_test, y_test, sequence_length)
            
            if len(X_test_seq) > 0:
                self.lstm_model.eval()
                
                X_test_seq_tensor = torch.FloatTensor(X_test_seq).to(self.device)
                
                with torch.no_grad():
                    lstm_predictions = self.lstm_model(X_test_seq_tensor).cpu().numpy()
                
                lstm_pred_binary = (lstm_predictions > 0.5).astype(int)
                lstm_auc = roc_auc_score(y_test_seq, lstm_predictions)
                lstm_f1 = f1_score(y_test_seq, lstm_pred_binary)
                
                print(f"LSTM Model - AUC: {lstm_auc:.4f}, F1: {lstm_f1:.4f}")
                print("Classification Report:")
                print(classification_report(y_test_seq, lstm_pred_binary))
                
                results['lstm'] = {
                    'auc': lstm_auc,
                    'f1': lstm_f1,
                    'predictions': lstm_predictions
                }
        
        return results
    
    def load_trained_models(self):
        """Load pre-trained deep learning models"""
        print("Loading trained deep learning models...")
        
        # Load Neural Network
        if NEURAL_NET_PATH.exists():
            input_size = len(self.preprocessor.feature_columns)
            hidden_layers = NEURAL_NET_CONFIG['hidden_layers']
            dropout_rate = NEURAL_NET_CONFIG['dropout_rate']
            
            self.neural_net = ChurnNeuralNetwork(input_size, hidden_layers, dropout_rate)
            self.neural_net.load_state_dict(torch.load(NEURAL_NET_PATH, map_location=self.device))
            self.neural_net.to(self.device)
            self.neural_net.eval()
            print("Neural Network model loaded")
        
        # Load LSTM
        if LSTM_MODEL_PATH.exists():
            input_size = len(self.preprocessor.feature_columns)
            hidden_size = LSTM_CONFIG['hidden_size']
            num_layers = LSTM_CONFIG['num_layers']
            dropout_rate = LSTM_CONFIG['dropout_rate']
            
            self.lstm_model = ChurnLSTM(input_size, hidden_size, num_layers, dropout_rate)
            self.lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=self.device))
            self.lstm_model.to(self.device)
            self.lstm_model.eval()
            print("LSTM model loaded")
    
    def predict_churn_deep_learning(self, customer_data):
        """Predict churn using deep learning models"""
        if self.neural_net is None and self.lstm_model is None:
            self.load_trained_models()
        
        # Preprocess the data
        processed_data = self.preprocessor.transform_new_data(customer_data).values
        
        predictions = {}
        
        # Neural Network prediction
        if self.neural_net is not None:
            self.neural_net.eval()
            data_tensor = torch.FloatTensor(processed_data).to(self.device)
            
            with torch.no_grad():
                nn_pred = self.neural_net(data_tensor).cpu().numpy()
            
            predictions['neural_network'] = nn_pred.flatten()
        
        # LSTM prediction (simplified - using last sequence_length samples)
        if self.lstm_model is not None and len(processed_data) >= LSTM_CONFIG['sequence_length']:
            self.lstm_model.eval()
            
            # Take last sequence_length samples
            seq_data = processed_data[-LSTM_CONFIG['sequence_length']:]
            seq_tensor = torch.FloatTensor(seq_data).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                lstm_pred = self.lstm_model(seq_tensor).cpu().numpy()
            
            predictions['lstm'] = lstm_pred.flatten()
        
        return predictions
    
    def train_all_deep_learning_models(self):
        """Train all deep learning models"""
        print("=" * 60)
        print("TRAINING DEEP LEARNING MODELS")
        print("=" * 60)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Train Neural Network
        nn_history = self.train_neural_network(X_train, y_train, X_test, y_test)
        
        # Train LSTM
        lstm_history = self.train_lstm_model(X_train, y_train, X_test, y_test)
        
        # Evaluate models
        results = self.evaluate_deep_learning_models(X_test, y_test)
        
        print("\n" + "=" * 60)
        print("DEEP LEARNING TRAINING COMPLETED")
        print("=" * 60)
        
        return results

def main():
    """Main function to train deep learning models"""
    dl_predictor = DeepLearningChurnPredictor()
    results = dl_predictor.train_all_deep_learning_models()
    MODEL PERFORMANCE
    print("\nDeep Learning pipeline completed successfully!")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Plots saved in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()




