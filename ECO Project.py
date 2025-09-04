import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class CorrectedStockPredictor_3Neurons:
    def _init_(self):
        self.scaler = StandardScaler()
        self.ann_model = None
        self.rf_model = None
        
    def fetch_data(self, symbol, start_date='2009-04-05', end_date='2019-04-05'):
        """Fetch stock data with CORRECTED date range"""
        try:
            stock = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Handle MultiIndex columns if present
            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = stock.columns.droplevel(1)
            
            # Convert to simple DataFrame structure
            if len(stock.columns.names) > 1:
                stock.columns = [col[0] if isinstance(col, tuple) else col for col in stock.columns]
            
            # Ensure we have a clean DataFrame with date index
            stock = pd.DataFrame(stock)
            if 'Date' not in stock.columns and stock.index.name != 'Date':
                stock.index.name = 'Date'
            
            return stock
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def create_new_variables(self, data):
        """Create the 6 new variables with CORRECTED rolling calculations"""
        df = data.copy()
        
        # Ensure we have the required columns
        required_columns = ['High', 'Low', 'Close', 'Open', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 1. Stock High minus Low price (H-L)
        df['H-L'] = df['High'] - df['Low']
        
        # 2. CORRECTED: Stock Close minus Open price (C-O) - NOT O-C
        df['C-O'] = df['Close'] - df['Open']
        
        # 3-6. CORRECTED: Remove min_periods to match paper's approach
        df['7_DAYS_MA'] = df['Close'].rolling(window=7).mean()
        df['14_DAYS_MA'] = df['Close'].rolling(window=14).mean()
        df['21_DAYS_MA'] = df['Close'].rolling(window=21).mean()
        df['7_DAYS_STD_DEV'] = df['Close'].rolling(window=7).std()
        
        return df
    
    def prepare_features(self, data):
        """Prepare features for training"""
        # Create new variables
        df = self.create_new_variables(data)
        
        # CORRECTED: Use C-O instead of O-C
        features = ['H-L', 'C-O', '7_DAYS_MA', '14_DAYS_MA', '21_DAYS_MA', '7_DAYS_STD_DEV', 'Volume']
        
        # Check if all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Create feature matrix
        X = df[features].copy()
        
        # Target variable (next day closing price)
        y = df['Close'].shift(-1)  # Shift to get next day's closing price
        
        # Drop rows with NaN values
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y
    
    def chronological_split(self, X, y, split_date='2017-04-04'):
        """CORRECTED: Chronological data splitting as per paper methodology"""
        split_date = pd.to_datetime(split_date)
        
        # Training: 2009-04-06 to 2017-04-03
        # Testing: 2017-04-04 to 2019-04-05
        train_mask = X.index < split_date
        test_mask = X.index >= split_date
        
        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X, y):
        """Train both ANN and Random Forest models with CORRECTED data splitting"""
        # CORRECTED: Use chronological split instead of random split
        X_train, X_test, y_train, y_test = self.chronological_split(X, y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ANN model - Architecture UNCHANGED as requested (3 neurons ONLY)
        self.ann_model = MLPRegressor(
            hidden_layer_sizes=(3,),  # UNCHANGED: Keep 3 neurons as you requested
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            learning_rate_init=0.001
        )
        self.ann_model.fit(X_train_scaled, y_train)
        
        # Train Random Forest model
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def evaluate_models(self, X_test, y_test, X_test_scaled):
        """Evaluate both models using RMSE, MAPE, and MBE"""
        # ANN predictions
        ann_predictions = self.ann_model.predict(X_test_scaled)
        
        # RF predictions
        rf_predictions = self.rf_model.predict(X_test)
        
        # Calculate metrics exactly as per paper
        def calculate_metrics(y_true, y_pred):
            # RMSE
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # MAPE with safe division
            epsilon = 1e-8
            y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
            mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
            
            # MBE (Mean Bias Error)
            mbe = np.mean(y_true - y_pred)
            
            return rmse, mape, mbe
        
        ann_rmse, ann_mape, ann_mbe = calculate_metrics(y_test, ann_predictions)
        rf_rmse, rf_mape, rf_mbe = calculate_metrics(y_test, rf_predictions)
        
        return {
            'ANN': {'RMSE': ann_rmse, 'MAPE': ann_mape, 'MBE': ann_mbe, 'predictions': ann_predictions},
            'RF': {'RMSE': rf_rmse, 'MAPE': rf_mape, 'MBE': rf_mbe, 'predictions': rf_predictions}
        }
    
    def visualize_ann_architecture(self):
        """Visualize the ANN architecture - UNCHANGED (3 neurons as requested)"""
        plt.figure(figsize=(12, 8))
        
        # Input layer positions
        input_features = ['H-L', 'C-O', '7 DAYS MA', '14 DAYS MA', '21 DAYS MA', '7 DAYS STD DEV', 'Volume']
        input_y = np.linspace(0.8, 0.2, len(input_features))
        
        # Hidden layer positions (3 neurons as you requested - UNCHANGED)
        hidden_y = np.linspace(0.7, 0.3, 3)
        
        # Output layer position
        output_y = [0.5]
        
        # Plot input layer
        for i, feature in enumerate(input_features):
            plt.text(0.1, input_y[i], feature, fontsize=10, ha='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
            plt.scatter(0.3, input_y[i], s=800, c='steelblue', alpha=0.7)
        
        # Plot hidden layer (3 neurons as you requested)
        for i in range(3):
            plt.scatter(0.6, hidden_y[i], s=800, c='steelblue', alpha=0.7)
        
        # Plot output layer
        plt.scatter(0.9, output_y[0], s=800, c='steelblue', alpha=0.7)
        plt.text(1.0, output_y[0], 'Output\n(Predicted\nClosing Price)', fontsize=10, ha='left', va='center')
        
        # Draw connections
        for i in range(len(input_features)):
            for j in range(3):
                plt.plot([0.3, 0.6], [input_y[i], hidden_y[j]], 'k-', alpha=0.3, linewidth=0.5)
        
        for i in range(3):
            plt.plot([0.6, 0.9], [hidden_y[i], output_y[0]], 'k-', alpha=0.3, linewidth=0.5)
        
        plt.xlim(0, 1.2)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('ANN Architecture - 3 Hidden Neurons (As You Requested)', fontsize=14, fontweight='bold')
        plt.text(0.3, 0.05, 'Input Layer\n(7 neurons)', ha='center', fontsize=12, fontweight='bold')
        plt.text(0.6, 0.05, 'Hidden Layer\n(3 neurons)', ha='center', fontsize=12, fontweight='bold')
        plt.text(0.9, 0.05, 'Output Layer\n(1 neuron)', ha='center', fontsize=12, fontweight='bold')
        plt.show()
        
    def plot_results(self, y_test, results, company_name):
        """Plot predicted vs actual prices"""
        plt.figure(figsize=(15, 10))
        
        # ANN plot
        plt.subplot(2, 1, 1)
        plt.plot(y_test.values, label='Actual', color='blue', linewidth=2)
        plt.plot(results['ANN']['predictions'], label='ANN Predicted', color='red', linewidth=2)
        plt.title(f'{company_name} - ANN Prediction vs Actual (3 Hidden Neurons)')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # RF plot
        plt.subplot(2, 1, 2)
        plt.plot(y_test.values, label='Actual', color='blue', linewidth=2)
        plt.plot(results['RF']['predictions'], label='RF Predicted', color='green', linewidth=2)
        plt.title(f'{company_name} - Random Forest Prediction vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("ANN Architecture Summary (AS YOU REQUESTED):")
        print("="*40)
        print("Input Layer: 7 neurons")
        print("  - H-L (High minus Low)")
        print("  - C-O (Close minus Open) [CORRECTED]")
        print("  - 7 DAYS MA (7-day moving average)")
        print("  - 14 DAYS MA (14-day moving average)")
        print("  - 21 DAYS MA (21-day moving average)")
        print("  - 7 DAYS STD DEV (7-day standard deviation)")
        print("  - Volume")
        print("Hidden Layer: 3 neurons (fully connected) [UNCHANGED AS REQUESTED]")
        print("Output Layer: 1 neuron (predicted closing price)")
        print("Total Parameters: 7×3 + 3×1 + 3 + 1 = 28 parameters")
        print("="*40)
    
    def predict_single_company(self, symbol, company_name):
        """Predict stock prices for a single company with CORRECTED methodology"""
        print(f"\n{'='*50}")
        print(f"Processing {company_name} ({symbol}) - 3 Hidden Neurons")
        print(f"{'='*50}")
        
        # CORRECTED: Use exact date range from paper
        data = self.fetch_data(symbol, start_date='2009-04-05', end_date='2019-04-05')
        if data is None:
            return None
        
        # Check if we have enough data
        if len(data) < 30:
            print(f"Insufficient data for {symbol}. Need at least 30 days.")
            return None
        
        # Prepare features with corrections
        try:
            X, y = self.prepare_features(data)
        except Exception as e:
            print(f"Error preparing features for {symbol}: {e}")
            return None
        
        # Check if we have enough samples after feature engineering
        if len(X) < 100:
            print(f"Insufficient samples after feature engineering for {symbol}.")
            return None
        
        # Train models with corrected methodology
        try:
            X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = self.train_models(X, y)
        except Exception as e:
            print(f"Error training models for {symbol}: {e}")
            return None
        
        # Evaluate models
        try:
            results = self.evaluate_models(X_test, y_test, X_test_scaled)
        except Exception as e:
            print(f"Error evaluating models for {symbol}: {e}")
            return None
        
        # Display results
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(X)}")
        print(f"Training samples: {len(X_train)} (2009-04-05 to 2017-04-03)")
        print(f"Testing samples: {len(X_test)} (2017-04-04 to 2019-04-05)")
        
        print(f"\nModel Performance (3 Hidden Neurons):")
        print(f"{'Model':<15} {'RMSE':<10} {'MAPE':<10} {'MBE':<10}")
        print(f"{'-'*45}")
        print(f"{'ANN':<15} {results['ANN']['RMSE']:<10.2f} {results['ANN']['MAPE']:<10.2f}% {results['ANN']['MBE']:<10.4f}")
        print(f"{'Random Forest':<15} {results['RF']['RMSE']:<10.2f} {results['RF']['MAPE']:<10.2f}% {results['RF']['MBE']:<10.4f}")
        
        # Plot results
        self.plot_results(y_test, results, company_name)
        
        return results


def main():
    """Main function with corrections but keeping 3 hidden neurons"""
    # Initialize corrected predictor with 3 neurons
    predictor = CorrectedStockPredictor_3Neurons()
    
    print("CORRECTED Stock Closing Price Prediction")
    print("ANN Architecture: 7 → 3 → 1 (3 Hidden Neurons AS REQUESTED)")
    print("All Other Corrections Applied")
    print("="*70)
    
    # Show the ANN architecture with 3 neurons
    print("\nDisplaying ANN Architecture (3 Hidden Neurons)...")
    predictor.visualize_ann_architecture()
    
    # Define companies as mentioned in the paper
    companies = [
        ('NKE', 'Nike'),
        ('GS', 'Goldman Sachs'),
        ('JNJ', 'Johnson & Johnson'),
        ('PFE', 'Pfizer'),
        ('JPM', 'JP Morgan Chase and Co.')
    ]
    
    # Store results for final comparison
    all_results = {}
    
    # Process each company
    for symbol, company_name in companies:
        result = predictor.predict_single_company(symbol, company_name)
        if result is not None:
            all_results[company_name] = result
    
    # Display final comparative analysis
    if all_results:
        print(f"\n{'='*70}")
        print("FINAL COMPARATIVE ANALYSIS - 3 Hidden Neurons")
        print(f"{'='*70}")
        
        print(f"\n{'Company':<25} {'ANN':<25} {'RF':<25}")
        print(f"{'':25} {'RMSE':<8} {'MAPE':<8} {'MBE':<8} {'RMSE':<8} {'MAPE':<8} {'MBE':<8}")
        print(f"{'-'*75}")
        
        ann_better_count = 0
        for company_name, results in all_results.items():
            ann_results = results['ANN']
            rf_results = results['RF']
            
            if ann_results['RMSE'] < rf_results['RMSE']:
                ann_better_count += 1
                
            print(f"{company_name:<25} {ann_results['RMSE']:<8.2f} {ann_results['MAPE']:<8.2f} {ann_results['MBE']:<8.4f} "
                  f"{rf_results['RMSE']:<8.2f} {rf_results['MAPE']:<8.2f} {rf_results['MBE']:<8.4f}")
        
        # Summary
        print(f"\n{'='*70}")
        print("IMPLEMENTATION SUMMARY")
        print(f"{'='*70}")
        print("✅ CORRECTIONS APPLIED:")
        print("  - Chronological data splitting (2009-2017 train, 2017-2019 test)")
        print("  - Correct date range (2009-04-05 to 2019-04-05)")
        print("  - Fixed feature C-O (Close minus Open)")
        print("  - Removed min_periods from rolling calculations")
        print()
        print("🔒 UNCHANGED (as you specifically requested):")
        print("  - ANN architecture: 7 input → 3 hidden → 1 output neurons")
        print("  - Total parameters: 28 (7×3 + 3×1 + 3 + 1)")
        print()
        print(f"📊 RESULTS:")
        print(f"  - ANN outperformed RF in {ann_better_count}/{len(all_results)} companies")
        if ann_better_count >= len(all_results) * 0.8:
            print("  ✅ Mostly consistent with paper findings")
        else:
            print("  ⚠  Mixed results - may need further optimization")
    else:
        print("No results to display. Please check the data availability.")


if _name_ == "_main_":
    main()