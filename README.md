## MA634 Financial Risk Management — Rolling Portfolio Backtest

Constructs and backtests GMV, Tangency (mean–variance), Equal-Weight (EW), and Active (Treynor–Black) portfolios vs NIFTY 50 with a 6M formation / 3M holding window (2009–2022), and backtests 99% historical VaR at a 3‑month horizon.

### Inputs
- `Stocks_data.csv` — daily closing prices (last column is NIFTY 50).
- `market_Factor_risk_Free.csv` — factors: `Date`, `MF` (percentage), `RF` (decimal).

### Option A: With a virtual environment (recommended)
1) Create and activate the environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```
2) Install dependencies
```bash
pip install -r requirements.txt
```
3) Run the script (replace file name with your script)
```bash
python Assignment_2022CSB1202_2022MCB1255_Codes.py
```


### Option B: Without a virtual environment
1) Install dependencies for your user account
```bash
pip3 install --user -r requirements.txt
```
2) Run the script (replace file name with your script)
```bash
python3 Assignment_2022CSB1093_2022CEB1027_Codes.py
```

### Outputs (saved in this folder)
- `portfolio_holding_returns_3m.csv` — n×5 rolling 3M returns: GMV, MV, EW, Active, NIFTY50
- `cumulative_growth_series.png` — cumulative growth of $1 across windows
- `performance_summary.csv` — mean (annualized), window std, Sharpe (annualized), IR vs NIFTY
- `historical_var_vs_realized_returns.png` — 99% VaR (3M) vs realized returns, with violations

### Notes
- Uses daily simple returns; MF is converted to decimal (÷100).
- Dates are intersected across stocks and factors.
- Weights are fixed over each 3M holding window (no rebalancing).




