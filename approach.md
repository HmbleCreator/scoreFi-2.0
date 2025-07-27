# Approach for Compound Protocol Wallet Risk Scoring

## Data Collection Method
We collected transaction data for each wallet address using the Etherscan API, specifically targeting transactions involving Compound V2 and V3 protocol contract addresses. For each wallet, we fetched all on-chain transactions and filtered them to include only those interacting with Compound contracts. This ensured that only relevant lending, borrowing, and liquidation activities were analyzed. The process included robust error handling and retry logic to manage API rate limits and network issues.

## Feature Selection Rationale
Feature engineering focused on capturing wallet behaviors most indicative of risk in lending protocols. We derived features such as:
- **Transaction counts and types** (supply, borrow, repay, redeem, liquidate)
- **Transaction volume statistics** (total, mean, volatility)
- **Temporal activity** (days active, recency, frequency)
- **Portfolio diversity** (number of unique assets and markets)
- **Behavioral ratios** (supply/borrow/repay/liquidate ratios, leverage, repayment discipline)

Features were selected based on their statistical correlation with risk indicators (e.g., liquidation events, leverage) using both Pearson and Spearman correlation analysis. Only features with significant correlation and variance were retained for scoring.

## Scoring Method
Each wallet was assigned a risk score between 0 and 1000. The scoring model grouped features into five risk categories:
- Liquidity Risk
- Activity Risk
- Volatility Risk
- Diversification Risk
- Behavioral Risk

Weights for each category were determined using data-driven analysis of feature importance, normalized to sum to 1. For each wallet, risk components were calculated from the engineered features, weighted, and summed to produce a final risk score. The model was robust to missing or anomalous data, defaulting to medium risk where necessary. The scoring logic was implemented in a modular, scalable Python class.

## Justification of the Risk Indicators Used
- **Liquidity Risk**: High leverage, frequent liquidations, and high borrow ratios are strong indicators of risk in lending protocols, as they reflect aggressive or unsustainable borrowing behavior.
- **Activity Risk**: Wallets with long inactivity, low recency, or abnormal transaction frequency may be abandoned or manipulated, increasing risk.
- **Volatility Risk**: High variance in transaction sizes suggests erratic behavior, which is riskier than consistent activity.
- **Diversification Risk**: Wallets interacting with few assets or markets are more exposed to single-asset failures or market shocks.
- **Behavioral Risk**: Poor repayment discipline and unbalanced supply/repay ratios indicate risky or irresponsible protocol usage.

These indicators are grounded in both protocol mechanics and empirical analysis of historical risk events in DeFi lending platforms.
