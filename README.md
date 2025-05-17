# 🧠 Crypto Trade Execution Simulator

A real-time trading simulator for OKX spot markets with advanced execution modeling, AI-based slippage prediction, market impact estimation (Almgren-Chriss), and live Dash UI.

---

## 🚀 Features

- ✅ Live L2 order book data via OKX WebSocket
- ✅ Dash-based real-time UI with inputs/outputs
- ✅ Slippage, market impact, and fee modeling
- ✅ Logistic regression for maker/taker prediction
- ✅ Numba-accelerated volatility estimator
- ✅ Performance metrics including latency tracking
- ✅ Real-time WebSocket-based data stream (OKX L2 Orderbook)

---

## 🛠️ Technologies

- Python 3.9+
- Dash + Plotly + Bootstrap
- websockets, ccxt, numpy, scikit-learn, numba
- GitHub Actions for CI

---

## 📦 Installation

```bash
git clone https://github.com/UdayalChaudhary/crypto-trade-simulator.git
cd crypto-trade-simulator
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/app.py
