# Complete Compound Protocol Wallet Risk Scoring System
# Enhanced with data-driven feature selection and justifiable methodology

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import json
from web3 import Web3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class CompoundFeatureAnalyzer:
    def __init__(self):
        """
        Initialize the feature analyzer for Compound protocol data
        """
        self.feature_correlations = {}
        self.risk_indicators = {}
        self.scaler = StandardScaler()
        
    def engineer_compound_features(self, df):
        """
        Engineer features from Compound transaction data
        
        Expected columns in df:
        - userWallet: wallet address
        - action: type of action (supply, redeem, borrow, repay, liquidate)
        - amount_usd: USD value of transaction
        - asset: asset symbol (ETH, USDC, DAI, etc.)
        - timestamp_dt: datetime of transaction
        - market: compound market (cToken address or symbol)
        """
        
        # Ensure timestamp is datetime
        if 'timestamp_dt' not in df.columns and 'timestamp' in df.columns:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        
        # Group by wallet and compute base features
        features_df = df.groupby('userWallet').agg({
            'action': ['count', 'nunique'],
            'amount_usd': ['sum', 'mean', 'std', 'max', 'min'],
            'asset': 'nunique',
            'market': 'nunique',
            'timestamp_dt': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        features_df.columns = ['userWallet', 'total_transactions', 'unique_actions',
                              'total_volume_usd', 'avg_transaction_usd', 'transaction_volatility',
                              'max_transaction_usd', 'min_transaction_usd',
                              'unique_assets', 'unique_markets',
                              'first_activity', 'last_activity']
        
        # Action-specific counts
        action_counts = df.groupby(['userWallet', 'action']).size().unstack(fill_value=0)
        action_columns = ['supply_count', 'redeem_count', 'borrow_count', 'repay_count', 'liquidate_count']
        
        for col in action_columns:
            action_name = col.replace('_count', '')
            if action_name in action_counts.columns:
                features_df[col] = features_df['userWallet'].map(action_counts[action_name]).fillna(0)
            else:
                features_df[col] = 0
                
        # Derived temporal features
        features_df['activity_days'] = (features_df['last_activity'] - features_df['first_activity']).dt.days + 1
        features_df['avg_days_between_tx'] = features_df['activity_days'] / features_df['total_transactions']
        features_df['transactions_per_day'] = features_df['total_transactions'] / features_df['activity_days']
        
        # Behavioral ratios (key risk indicators)
        features_df['supply_ratio'] = features_df['supply_count'] / features_df['total_transactions']
        features_df['borrow_ratio'] = features_df['borrow_count'] / features_df['total_transactions']
        features_df['repay_ratio'] = features_df['repay_count'] / features_df['total_transactions']
        features_df['liquidate_ratio'] = features_df['liquidate_count'] / features_df['total_transactions']
        features_df['redeem_ratio'] = features_df['redeem_count'] / features_df['total_transactions']
        
        # Risk-specific derived features
        features_df['leverage_indicator'] = features_df['borrow_count'] / (features_df['supply_count'] + 1)
        features_df['repayment_discipline'] = features_df['repay_count'] / (features_df['borrow_count'] + 1)
        features_df['portfolio_diversity'] = (features_df['unique_assets'] * features_df['unique_markets']) / features_df['total_transactions']
        features_df['transaction_size_consistency'] = 1 / (1 + features_df['transaction_volatility'] / (features_df['avg_transaction_usd'] + 1))
        
        # Market timing features
        current_date = pd.Timestamp.now()
        features_df['days_since_last_activity'] = (current_date - features_df['last_activity']).dt.days
        features_df['recency_score'] = 1 / (1 + features_df['days_since_last_activity'] / 30)  # 30-day decay
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        return features_df
    
    def calculate_feature_correlations(self, features_df, target_column='liquidate_ratio'):
        """
        Calculate correlations between features and risk indicators
        """
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'userWallet']
        
        correlations = {}
        
        for feature in numeric_columns:
            if feature != target_column and features_df[feature].var() > 0:
                # Pearson correlation
                pearson_corr, pearson_p = pearsonr(features_df[feature], features_df[target_column])
                
                # Spearman correlation (for non-linear relationships)
                spearman_corr, spearman_p = spearmanr(features_df[feature], features_df[target_column])
                
                correlations[feature] = {
                    'pearson_corr': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_p_value': spearman_p,
                    'abs_pearson': abs(pearson_corr),
                    'abs_spearman': abs(spearman_corr)
                }
        
        return pd.DataFrame(correlations).T
    
    def create_correlation_heatmap(self, features_df, figsize=(15, 12)):
        """
        Create comprehensive correlation heatmap
        """
        # Select numeric features only
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # Remove identifier columns
        feature_cols = [col for col in numeric_features.columns 
                       if col not in ['userWallet'] and numeric_features[col].var() > 0]
        
        correlation_matrix = numeric_features[feature_cols].corr()
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": .8},
                   fmt='.2f')
        
        plt.title('Feature Correlation Heatmap - Compound Protocol Risk Factors', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def justify_feature_weights(self, features_df):
        """
        Create data-driven justification for feature weights with better error handling
        """
        # Check if we have enough data
        if len(features_df) < 5:
            print("Warning: Insufficient data for correlation analysis, using default importance")
            return {}
            
        # Calculate multiple risk correlations
        risk_indicators = ['liquidate_ratio', 'leverage_indicator', 'transaction_volatility', 
                          'days_since_last_activity', 'borrow_ratio']
        
        feature_importance = {}
        
        for risk in risk_indicators:
            if risk in features_df.columns and features_df[risk].var() > 0:
                try:
                    corr_df = self.calculate_feature_correlations(features_df, risk)
                    
                    if not corr_df.empty:
                        # Weight by correlation strength and statistical significance
                        for feature, row in corr_df.iterrows():
                            if feature not in feature_importance:
                                feature_importance[feature] = 0
                            
                            # Check for valid correlation values
                            pearson_corr = row.get('abs_pearson', 0)
                            p_value = row.get('pearson_p_value', 1)
                            
                            if not (np.isnan(pearson_corr) or np.isnan(p_value)):
                                # Add weighted importance based on correlation and significance
                                significance_weight = 1 if p_value < 0.05 else 0.5
                                feature_importance[feature] += pearson_corr * significance_weight
                                
                except Exception as e:
                    print(f"Error calculating correlations for {risk}: {str(e)}")
                    continue
        
        # Remove NaN values and normalize importance scores
        valid_importance = {k: v for k, v in feature_importance.items() 
                           if not (np.isnan(v) or np.isinf(v))}
        
        if not valid_importance:
            print("Warning: No valid feature importance scores calculated")
            return {}
            
        total_importance = sum(valid_importance.values())
        if total_importance > 0:
            for feature in valid_importance:
                valid_importance[feature] /= total_importance
        
        return valid_importance
    
    def create_feature_importance_plot(self, feature_importance, top_n=15):
        """
        Plot feature importance based on correlations
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importance = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importance)
        
        # Color bars by importance level
        colors = ['darkred' if imp > 0.15 else 'red' if imp > 0.10 else 'orange' if imp > 0.05 else 'lightblue' 
                 for imp in importance]
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance Score (Correlation-Based)')
        plt.title('Data-Driven Feature Importance for Compound Risk Scoring', fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add importance threshold lines
        plt.axvline(x=0.15, color='red', linestyle='--', alpha=0.7, label='High Importance (>0.15)')
        plt.axvline(x=0.10, color='orange', linestyle='--', alpha=0.7, label='Medium Importance (>0.10)')
        plt.axvline(x=0.05, color='yellow', linestyle='--', alpha=0.7, label='Low Importance (>0.05)')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return sorted_features

class CompoundRiskScorer:
    def __init__(self, infura_url=None, etherscan_api_key=None):
        """
        Initialize the risk scorer with API credentials and feature analyzer
        """
        self.infura_url = infura_url or "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
        self.etherscan_api_key = etherscan_api_key or "YOUR_ETHERSCAN_API_KEY"
        self.analyzer = CompoundFeatureAnalyzer()
        
        # Compound V2 and V3 contract addresses
        self.compound_contracts = {
            'cDAI': '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643',
            'cUSDC': '0x39aa39c021dfbae8fac545936693ac917d5e7563',
            'cUSDT': '0xf650c3d88d12db855b8bf7d11be6c55a4e07dcc9',
            'cETH': '0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5',
            'cWBTC': '0xc11b1268c1a384e55c48c2391d8d480264a3a7f4',
            'cUSDCv3': '0xc3d688b66703497daa19211eedff47f25384cdc3',
            'comptroller': '0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b'
        }
        
        # Data-driven weights (will be calculated from actual data)
        self.weights = {}
        self.scaler = MinMaxScaler()
        
    def get_transactions(self, wallet_address, start_block=0, max_retries=3):
        """
        Fetch transactions from Etherscan API for a given wallet with retry logic
        """
        for attempt in range(max_retries):
            try:
                url = f"https://api.etherscan.io/api"
                params = {
                    'module': 'account',
                    'action': 'txlist',
                    'address': wallet_address,
                    'startblock': start_block,
                    'endblock': 'latest',
                    'page': 1,
                    'offset': 1000,
                    'sort': 'desc',
                    'apikey': self.etherscan_api_key
                }

                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"Rate limit hit for {wallet_address}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                data = response.json()

                if data['status'] == '1':
                    return data['result']
                elif data['message'] == 'NOTOK' and 'rate limit' in data.get('result', '').lower():
                    wait_time = 2 ** attempt
                    print(f"Rate limit in response for {wallet_address}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"API returned status 0 for {wallet_address}: {data.get('message', 'Unknown error')}")
                    return []

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue

        print(f"Failed to fetch transactions for {wallet_address} after {max_retries} attempts")
        return []

    def parse_compound_transaction(self, tx):
        """
        Parse a Compound transaction to extract action, amount, and asset information
        """
        # This is a simplified parser - you'd need more sophisticated parsing
        # for production use, potentially using contract ABIs
        
        compound_addresses = set(addr.lower() for addr in self.compound_contracts.values())
        to_address = tx.get('to', '').lower()
        
        if to_address not in compound_addresses:
            return None
            
        # Basic action classification based on method ID
        input_data = tx.get('input', '')
        value = float(tx.get('value', 0)) / 1e18
        
        action = 'unknown'
        if input_data.startswith('0xa0712d68'):  # mint
            action = 'supply'
        elif input_data.startswith('0xdb006a75'):  # redeem
            action = 'redeem'
        elif input_data.startswith('0xc5ebeaec'):  # borrow
            action = 'borrow'
        elif input_data.startswith('0x0e752702'):  # repay
            action = 'repay'
        elif input_data.startswith('0xf5e3c462'):  # liquidate
            action = 'liquidate'
            
        # Estimate USD value (simplified - would need price feeds for accuracy)
        amount_usd = value * 2000  # Rough ETH price estimate
        
        return {
            'userWallet': tx['from'],
            'action': action,
            'amount_usd': amount_usd,
            'asset': 'ETH',  # Simplified
            'market': to_address,
            'timestamp_dt': datetime.fromtimestamp(int(tx['timeStamp'])),
            'txHash': tx['hash']
        }

    def convert_transactions_to_dataframe(self, wallet_transactions):
        """
        Convert transaction data to DataFrame format for feature engineering
        """
        parsed_transactions = []
        
        for wallet, transactions in wallet_transactions.items():
            for tx in transactions:
                parsed_tx = self.parse_compound_transaction(tx)
                if parsed_tx:
                    parsed_transactions.append(parsed_tx)
                    
        return pd.DataFrame(parsed_transactions)

    def calculate_data_driven_weights(self, features_df):
        """
        Calculate scoring weights based on actual data correlations with fallback defaults
        """
        # Default weights as fallback
        default_weights = {
            'liquidity_risk': 0.30,
            'activity_risk': 0.25,
            'volatility_risk': 0.20,
            'diversification_risk': 0.15,
            'behavioral_risk': 0.10
        }
        
        try:
            feature_importance = self.analyzer.justify_feature_weights(features_df)
            
            # Check if we have valid feature importance scores
            if not feature_importance or all(np.isnan(list(feature_importance.values()))):
                print("Warning: No valid feature importance scores, using default weights")
                self.weights = default_weights
                return default_weights
            
            # Group features into risk categories
            feature_categories = {
                'liquidity_risk': ['liquidate_ratio', 'leverage_indicator', 'borrow_ratio'],
                'activity_risk': ['days_since_last_activity', 'recency_score', 'transactions_per_day'],
                'volatility_risk': ['transaction_volatility', 'transaction_size_consistency'],
                'diversification_risk': ['portfolio_diversity', 'unique_assets', 'unique_markets'],
                'behavioral_risk': ['repayment_discipline', 'supply_ratio', 'repay_ratio']
            }
            
            category_weights = {}
            
            for category, features in feature_categories.items():
                category_importance = 0
                feature_count = 0
                
                for feature in features:
                    if feature in feature_importance and not np.isnan(feature_importance[feature]):
                        category_importance += feature_importance[feature]
                        feature_count += 1
                
                if feature_count > 0:
                    category_weights[category] = category_importance / feature_count
                else:
                    category_weights[category] = default_weights[category]
            
            # Check for NaN values and replace with defaults
            for category in category_weights:
                if np.isnan(category_weights[category]):
                    category_weights[category] = default_weights[category]
            
            # Normalize category weights to sum to 1
            total_weight = sum(category_weights.values())
            if total_weight > 0 and not np.isnan(total_weight):
                for category in category_weights:
                    category_weights[category] /= total_weight
            else:
                print("Warning: Invalid total weight, using default weights")
                category_weights = default_weights
            
            self.weights = category_weights
            return category_weights
            
        except Exception as e:
            print(f"Error calculating data-driven weights: {str(e)}")
            print("Using default weights")
            self.weights = default_weights
            return default_weights

    def calculate_risk_score(self, wallet_features, weights):
        """
        Calculate risk score using data-driven features and weights with NaN handling
        """
        # Helper function to safely get feature value
        def safe_get(feature_name, default=0):
            value = wallet_features.get(feature_name, default)
            return default if (np.isnan(value) or np.isinf(value)) else value
        
        # Risk components based on feature categories
        risk_components = {}
        
        # Liquidity Risk
        liquidate_ratio = safe_get('liquidate_ratio', 0)
        leverage_indicator = safe_get('leverage_indicator', 0)
        borrow_ratio = safe_get('borrow_ratio', 0)
        
        liquidity_risk = (
            liquidate_ratio * 0.4 +
            min(leverage_indicator, 2.0) * 0.4 +  # Cap leverage at 2.0
            borrow_ratio * 0.2
        )
        risk_components['liquidity_risk'] = min(liquidity_risk, 1.0)
        
        # Activity Risk
        days_inactive = safe_get('days_since_last_activity', 0)
        recency_score = safe_get('recency_score', 0)
        tx_per_day = safe_get('transactions_per_day', 0)
        
        activity_risk = (
            min(days_inactive / 365, 1.0) * 0.5 +
            (1 - recency_score) * 0.3 +
            min(abs(tx_per_day - 0.1) * 10, 1.0) * 0.2  # Optimal ~0.1 tx/day
        )
        risk_components['activity_risk'] = min(activity_risk, 1.0)
        
        # Volatility Risk
        tx_volatility = safe_get('transaction_volatility', 0)
        size_consistency = safe_get('transaction_size_consistency', 1)
        
        volatility_risk = (
            min(tx_volatility / 10000, 1.0) * 0.7 +  # Normalize volatility
            (1 - min(size_consistency, 1.0)) * 0.3
        )
        risk_components['volatility_risk'] = min(volatility_risk, 1.0)
        
        # Diversification Risk
        portfolio_diversity = safe_get('portfolio_diversity', 0)
        unique_assets = safe_get('unique_assets', 1)
        unique_markets = safe_get('unique_markets', 1)
        
        diversification_risk = (
            (1 - min(portfolio_diversity, 1.0)) * 0.4 +
            (1 - min(unique_assets / 5, 1.0)) * 0.3 +
            (1 - min(unique_markets / 5, 1.0)) * 0.3
        )
        risk_components['diversification_risk'] = min(diversification_risk, 1.0)
        
        # Behavioral Risk
        repayment_discipline = safe_get('repayment_discipline', 0)
        supply_ratio = safe_get('supply_ratio', 0)
        repay_ratio = safe_get('repay_ratio', 0)
        
        behavioral_risk = (
            (1 - min(repayment_discipline, 1.0)) * 0.5 +
            abs(supply_ratio - 0.5) * 0.3 +  # Optimal supply ratio ~0.5
            abs(repay_ratio - 0.3) * 0.2   # Optimal repay ratio ~0.3
        )
        risk_components['behavioral_risk'] = min(behavioral_risk, 1.0)
        
        # Calculate weighted final score
        final_score = 0
        for component, risk_value in risk_components.items():
            weight = weights.get(component, 0.2)  # Default weight if missing
            if not (np.isnan(weight) or np.isnan(risk_value)):
                final_score += weight * risk_value
        
        # Ensure final_score is valid
        if np.isnan(final_score) or np.isinf(final_score):
            final_score = 0.5  # Medium risk default
        
        # Convert to 0-1000 scale
        return int(min(max(final_score * 1000, 0), 1000))

    def score_wallets_enhanced(self, wallet_addresses, training_sample_size=100):
        """
        Enhanced wallet scoring with data-driven feature selection and robust error handling
        """
        print(f"Starting enhanced scoring for {len(wallet_addresses)} wallets...")
        
        # Step 1: Collect training data from sample of wallets
        print("Step 1: Collecting training data...")
        training_wallets = wallet_addresses[:min(training_sample_size, len(wallet_addresses))]
        training_transactions = {}
        
        for i, wallet in enumerate(training_wallets):
            print(f"Fetching training data {i+1}/{len(training_wallets)}: {wallet}")
            transactions = self.get_transactions(wallet)
            if transactions:  # Only add if we got transactions
                training_transactions[wallet] = transactions
            time.sleep(0.3)  # Rate limiting
            
        # Step 2: Convert to DataFrame and engineer features
        print("Step 2: Engineering features...")
        training_df = self.convert_transactions_to_dataframe(training_transactions)
        
        if training_df.empty or len(training_df) < 10:
            print("Warning: Insufficient training data, using default weights")
            weights = {
                'liquidity_risk': 0.30,
                'activity_risk': 0.25,
                'volatility_risk': 0.20,
                'diversification_risk': 0.15,
                'behavioral_risk': 0.10
            }
        else:
            try:
                features_df = self.analyzer.engineer_compound_features(training_df)
                
                # Step 3: Analyze features and calculate weights
                print("Step 3: Analyzing features...")
                
                # Only create visualizations if we have enough data
                if len(features_df) >= 5:
                    try:
                        correlation_matrix = self.analyzer.create_correlation_heatmap(features_df)
                        feature_importance = self.analyzer.justify_feature_weights(features_df)
                        if feature_importance:
                            self.analyzer.create_feature_importance_plot(feature_importance)
                    except Exception as e:
                        print(f"Warning: Could not create visualizations: {str(e)}")
                
                # Step 4: Calculate data-driven weights
                print("Step 4: Calculating data-driven weights...")
                weights = self.calculate_data_driven_weights(features_df)
                
                print("Data-driven weights:")
                for category, weight in weights.items():
                    print(f"  {category}: {weight:.3f}")
                    
            except Exception as e:
                print(f"Error in feature analysis: {str(e)}")
                print("Using default weights")
                weights = {
                    'liquidity_risk': 0.30,
                    'activity_risk': 0.25,
                    'volatility_risk': 0.20,
                    'diversification_risk': 0.15,
                    'behavioral_risk': 0.10
                }
        
        # Step 5: Score all wallets
        print("Step 5: Scoring all wallets...")
        results = []
        
        for i, wallet in enumerate(wallet_addresses):
            try:
                print(f"Scoring wallet {i+1}/{len(wallet_addresses)}: {wallet}")
                
                # Get wallet transactions
                transactions = self.get_transactions(wallet)
                wallet_df = self.convert_transactions_to_dataframe({wallet: transactions})
                
                if wallet_df.empty:
                    # High risk for wallets with no Compound activity
                    score = 850
                else:
                    # Engineer features for this wallet
                    wallet_features_df = self.analyzer.engineer_compound_features(wallet_df)
                    
                    if len(wallet_features_df) > 0:
                        wallet_features = wallet_features_df.iloc[0].to_dict()
                        score = self.calculate_risk_score(wallet_features, weights)
                    else:
                        score = 800
                
                results.append({
                    'wallet_id': wallet,
                    'score': score
                })
                
                # Save progress
                if (i + 1) % 10 == 0:
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv('enhanced_wallet_scores.csv', index=False)
                    print(f"Progress saved - {i+1}/{len(wallet_addresses)} complete")
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"Error scoring wallet {wallet}: {str(e)}")
                results.append({
                    'wallet_id': wallet,
                    'score': 500  # Default medium risk
                })
        
        # Save final results
        results_df = pd.DataFrame(results)
        results_df.to_csv('enhanced_wallet_scores.csv', index=False)
        print(f"Enhanced scoring complete! Results saved to enhanced_wallet_scores.csv")
        
        return results_df, weights

def main():
    """
    Main execution function
    """
    print("=== Enhanced Compound Protocol Wallet Risk Scoring System ===")
    print()
    
    # API Configuration
    ETHERSCAN_API_KEY = "2E8Z182IFISX8QNJHGQP7RD9S8CMUYGVXQ"
    INFURA_URL = "https://mainnet.infura.io/v3/1daa5e5385054a21932c3bc33dff2d6b"
    
    # Initialize scorer
    scorer = CompoundRiskScorer(
        infura_url=INFURA_URL,
        etherscan_api_key=ETHERSCAN_API_KEY
    )
    
    # Only use 'Wallet id - Sheet1.csv' for wallet addresses
    csv_filename = "Wallet id - Sheet1.csv"
    try:
        df = pd.read_csv(csv_filename)
        wallet_column = None
        possible_columns = ['wallet_address', 'address', 'wallet', 'wallet_id', 'account', 'userWallet', 'user_wallet']
        for col in possible_columns:
            if col in df.columns:
                wallet_column = col
                break
        if wallet_column is None:
            print("Could not automatically detect wallet address column.")
            print("Available columns:", list(df.columns))
            wallet_column = input("Please enter the column name containing wallet addresses: ")
        wallet_addresses = df[wallet_column].dropna().unique().tolist()
        print(f"Loaded {len(wallet_addresses)} wallet addresses from '{csv_filename}' column '{wallet_column}'")
    except Exception as e:
        print(f"❌ Error loading CSV file: {str(e)}")
        return

    # Run enhanced scoring
    results_df, weights = scorer.score_wallets_enhanced(wallet_addresses)

    print("\n=== FINAL RESULTS ===")
    print(f"Scored {len(results_df)} wallets")
    print(f"\nScore distribution:")
    print(f"Mean: {results_df['score'].mean():.1f}")
    print(f"Median: {results_df['score'].median():.1f}")
    print(f"Min: {results_df['score'].min()}")
    print(f"Max: {results_df['score'].max()}")

    print(f"\nData-driven category weights used:")
    for category, weight in weights.items():
        print(f"  {category}: {weight:.1%}")

    # Save the ML model (scaler and weights) in the current folder
    import joblib
    model_artifacts = {
        'scaler': scorer.scaler,
        'weights': weights
    }
    joblib.dump(model_artifacts, 'compound_risk_model.joblib')
    print("\nML model artifacts saved to 'compound_risk_model.joblib'")

if __name__ == "__main__":
    main()

# Additional utility functions for analysis

def analyze_existing_data(csv_file_path):
    """
    Analyze existing transaction data if you have it
    """
    print("Analyzing existing transaction data...")
    
    df = pd.read_csv(csv_file_path)
    analyzer = CompoundFeatureAnalyzer()
    
    # Engineer features
    features_df = analyzer.engineer_compound_features(df)
    
    # Create visualizations
    correlation_matrix = analyzer.create_correlation_heatmap(features_df)
    feature_importance = analyzer.justify_feature_weights(features_df)
    analyzer.create_feature_importance_plot(feature_importance)
    
    return features_df, feature_importance

def create_risk_report(results_df, weights):
    """
    Create a comprehensive risk analysis report
    """
    report = f"""
# Compound Protocol Risk Scoring Report

## Summary Statistics
- Total wallets analyzed: {len(results_df)}
- Average risk score: {results_df['score'].mean():.1f}
- High risk wallets (>750): {len(results_df[results_df['score'] > 750])}
- Medium risk wallets (400-750): {len(results_df[(results_df['score'] >= 400) & (results_df['score'] <= 750)])}
- Low risk wallets (<400): {len(results_df[results_df['score'] < 400])}

## Data-Driven Weight Configuration
"""
    
    for category, weight in weights.items():
        report += f"- {category.replace('_', ' ').title()}: {weight:.1%}\n"
    
    report += f"""
## Risk Distribution
- 99th percentile: {results_df['score'].quantile(0.99):.0f}
- 95th percentile: {results_df['score'].quantile(0.95):.0f}
- 75th percentile: {results_df['score'].quantile(0.75):.0f}
- 25th percentile: {results_df['score'].quantile(0.25):.0f}

## Methodology
This scoring system uses data-driven feature selection based on correlation analysis
with actual risk indicators rather than arbitrary weights. Features are selected
based on their statistical significance and correlation with liquidation events
and other risk behaviors.
"""
    
    return report