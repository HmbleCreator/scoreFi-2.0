<div align="center"> # scoreFi-2.0 </div>



# Compound Protocol Wallet Risk Scoring System

## Project Overview
This project implements a risk scoring system for DeFi wallets interacting with the Compound V2 and V3 protocols. The system analyzes on-chain transaction patterns, behavioral indicators, and risk management practices to assign risk scores between 0-1000, where higher scores indicate higher risk.

## Methodology and Rationale
For a detailed explanation of the data collection, feature selection, scoring method, and justification of risk indicators, please see [approach.md](./approach.md).

## How It Works
1. **Data Collection**: The script fetches transaction data for each wallet address from the Etherscan API, filtering for interactions with Compound protocol contracts.
2. **Feature Engineering**: For each wallet, features are engineered to capture transaction types, volume, activity patterns, portfolio diversity, and behavioral ratios relevant to risk.
3. **Risk Scoring**: Features are grouped into five risk categories (liquidity, activity, volatility, diversification, behavioral). Each category is weighted based on data-driven analysis, and a final risk score (0-1000) is computed for each wallet.
4. **Output**: Results are saved as a CSV file with columns `wallet_id` and `score`. Model artifacts (scaler and weights) are saved as `compound_risk_model.joblib`.


## API Keys: Etherscan & Infura
This project requires access to the Etherscan API (for fetching on-chain transaction data) and Infura (for Ethereum node access if needed). You must set your API keys in the main execution block of `compound_risk_scorer.py`:

```
ETHERSCAN_API_KEY = "YOUR_ETHERSCAN_API_KEY"
INFURA_URL = "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID"
```
Replace the placeholders with your actual API keys before running the script. You can obtain free API keys by registering at [Etherscan](https://etherscan.io/) and [Infura](https://infura.io/).

## Usage

### Prerequisites
```bash
pip install web3 pandas numpy matplotlib seaborn scikit-learn scipy requests joblib
```

### Running the Scorer
1. Place your wallet addresses in a CSV file named `Wallet id - Sheet1.csv` in the project directory. The file should contain a column with wallet addresses (e.g., `wallet_id`).
2. Run `compound_risk_scorer.py`:
   ```bash
   python compound_risk_scorer.py
   ```
3. The script will fetch data, compute features, score wallets, and save results to `enhanced_wallet_scores.csv` and model artifacts to `compound_risk_model.joblib`.

## Output Files
- `enhanced_wallet_scores.csv`: Wallet-level risk scores
- `compound_risk_model.joblib`: Model scaler and weights


## Score Interpretation
- **0-399**: Low risk
- **400-749**: Medium risk
- **750-1000**: High risk

### Score Normalization Example
The following code snippet demonstrates how raw model outputs (if using a machine learning model) can be normalized to a 0-1000 scale:
```python
preds = model.predict(X_scaled)
scores = 1000 * (preds - preds.min()) / (preds.max() - preds.min())
features_df['credit_score'] = scores.astype(int)
```
In this project, the risk scoring logic is implemented directly in the script, but this normalization approach is standard for converting raw scores to a 0-1000 range.

## Customization
- To use a different wallet CSV, update the filename in the script or rename your file accordingly.
- API keys for Etherscan and Infura are set in the script; update them as needed.

## License
Open source under MIT License.
