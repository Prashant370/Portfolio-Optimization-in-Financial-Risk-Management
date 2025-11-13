## MA634 Financial Risk Management — Rolling Portfolio Backtest

Constructs and backtests GMV, Tangency (mean–variance), Equal-Weight (EW), and Active (Treynor–Black) portfolios vs NIFTY 50 with a 6M formation / 3M holding window (2009–2022), and backtests 99% historical VaR at a 3‑month horizon.

### Inputs
- `Stocks_data.csv` — daily closing prices (last column is NIFTY 50).
- `market_Factor_risk_Free.csv` — factors: `Date`, `MF` (percentage), `RF` (decimal).

### Setup (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run
```powershell
python Assignment_2022CSB1202_Codes.py
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




