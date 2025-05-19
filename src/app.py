import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import logging
import asyncio
import threading
import time
from datetime import datetime
from src.ws_client import OKXWebSocket
from src.orderbook import OrderBookProcessor
from src.models import ExecutionModels

# Initialize components
ob_processor = OrderBookProcessor()
models = ExecutionModels()
ws_client = OKXWebSocket()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [UI] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("ui.log"),
        logging.StreamHandler()
    ]
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Crypto Trade Simulator"

app.layout = dbc.Container([
    # Status Bar
    dbc.Row([
        dbc.Col([
            dbc.Badge(
                id="ws-status",
                color="light",
                className="me-1",
                style={"fontSize": "1rem"}
            ),
            dbc.Badge(
                id="data-freshness",
                color="light",
                className="me-1",
                style={"fontSize": "1rem"}
            ),
            dbc.Badge(
                id="message-rate",
                color="light",
                style={"fontSize": "1rem"}
            )
        ], width=12)
    ], className="mb-3"),

    # Main Content
    dbc.Row([
        # Input Panel
        dbc.Col([
            html.H1("Trade Execution Simulator", className="mb-4"),
            dbc.Card([
                dbc.CardBody([
                    dcc.Dropdown(
                        id="symbol",
                        options=[{"label": s, "value": s} for s in ["BTC-USDT", "ETH-USDT"]],
                        value="BTC-USDT",
                        clearable=False,
                        className="mb-3"
                    ),
                    dbc.Input(
                        id="quantity",
                        type="number",
                        value=100,
                        min=1,
                        max=1_000_000,
                        step=1,
                        placeholder="Quantity (USD)",
                        className="mb-3"
                    ),
                    dbc.RadioItems(
                        id="volatility-source",
                        options=[
                            {"label": "Manual Input", "value": "manual"},
                            {"label": "Auto-Calculate", "value": "auto"}
                        ],
                        value="manual",
                        className="mb-3"
                    ),
                    dbc.Input(
                        id="volatility-input",
                        type="number",
                        value=0.02,
                        min=0.001,
                        max=0.5,
                        step=0.001,
                        disabled=False,
                        className="mb-3"
                    ),
                    dcc.Dropdown(
                        id="fee-tier",
                        options=[
                            {"label": "Tier 1 (0.1%)", "value": 0.001},
                            {"label": "Tier 2 (0.08%)", "value": 0.0008}
                        ],
                        value=0.001,
                        clearable=False,
                        className="mb-3"
                    ),
                    dbc.Button(
                        "Simulate",
                        id="simulate-btn",
                        n_clicks=0,
                        color="primary",
                        className="w-100"
                    )
                ])
            ])
        ], width=4),

        # Output Panel
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(id="output-slippage", className="mb-2"),
                    html.Div(id="output-fees", className="mb-2"),
                    html.Div(id="output-impact", className="mb-2"),
                    html.Div(id="output-net-cost", className="mb-2 h4"),
                    html.Div(id="output-maker-prob", className="mb-2"),
                    html.Div(id="output-latency", className="mb-2 text-muted")
                ])
            ]),
            
            # Log Panel
            dbc.Card([
                dbc.CardHeader("Event Log"),
                dbc.CardBody([
                    dcc.Textarea(
                        id="log-output",
                        style={'width': '100%', 'height': 150},
                        readOnly=True
                    )
                ])
            ], className="mt-3")
        ], width=8)
    ]),

    # Hidden components
    dcc.Store(id="error-store", data=""),
    dcc.Interval(id="update-interval", interval=1000),
    dcc.Interval(id="log-update", interval=2000)
])

# Callbacks
@app.callback(
    [Output("volatility-input", "disabled"),
     Output("volatility-input", "value")],
    [Input("volatility-source", "value")]
)
def toggle_volatility_input(source):
    if source == "auto":
        try:
            if len(ob_processor.price_history) >= 2:
                volatility = models.calculate_volatility(np.array(list(ob_processor.price_history)))
                return True, round(volatility, 4)
        except Exception as e:
            logging.error(f"Volatility calc error: {e}")
    return False, 0.02

@app.callback(
    [Output("ws-status", "children"),
     Output("ws-status", "color"),
     Output("data-freshness", "children"),
     Output("data-freshness", "color"),
     Output("message-rate", "children")],
    [Input("update-interval", "n_intervals")]
)
def update_status(_):
    # WebSocket status
    status = ws_client.status
    ws_color = "success" if status == "connected" else "danger"
    
    # Data freshness
    freshness = "Data: FRESH" if ob_processor.is_fresh else "Data: STALE"
    fresh_color = "success" if ob_processor.is_fresh else "warning"
    
    # Message rate
    rate = f"Rate: {ws_client.message_rate:.1f}/s"
    
    return (
        f"WebSocket: {status.upper()}",
        ws_color,
        freshness,
        fresh_color,
        rate
    )

@app.callback(
    [Output("output-slippage", "children"),
     Output("output-fees", "children"),
     Output("output-impact", "children"),
     Output("output-net-cost", "children"),
     Output("output-maker-prob", "children"),
     Output("output-latency", "children"),
     Output("simulate-btn", "disabled"),
     Output("error-store", "data")],
    [Input("simulate-btn", "n_clicks"),
     Input("update-interval", "n_intervals")],
    [State("quantity", "value"),
     State("volatility-source", "value"),
     State("volatility-input", "value"),
     State("fee-tier", "value"),
     State("error-store", "data")]
)
def run_simulation(n_clicks, n_intervals, quantity, vol_source, vol_manual, fee_tier, prev_error):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Validate and normalize inputs
        quantity = np.clip(float(quantity), 1, 1_000_000)
        fee_tier = 0.001 if fee_tier not in [0.001, 0.0008] else fee_tier
        
        if vol_source == "auto":
            volatility = models.calculate_volatility(np.array(list(ob_processor.price_history)))
        else:
            volatility = np.clip(float(vol_manual), 0.001, 0.5)
        
        # Calculate metrics
        slippage = ob_processor.calculate_slippage(quantity)
        fees = quantity * fee_tier
        impact = models.almgren_chriss_impact(quantity, volatility, 1_000_000)
        net_cost = slippage + fees + impact
        maker_prob = models.predict_maker_probability([quantity, volatility])
        avg_latency = np.mean(ob_processor.latency_log) if ob_processor.latency_log else 0
        
        return [
            f"Slippage: {slippage:.2f}%",
            f"Fees: ${fees:.2f}",
            f"Market Impact: ${impact:.2f}",
            f"Total Cost: ${net_cost:.2f}",
            f"Maker Probability: {maker_prob*100:.1f}%",
            f"Processing Latency: {avg_latency:.1f}ms",
            False,  # Re-enable button
            ""  # Clear error store
        ]
    except Exception as e:
        error_msg = str(e)
        if error_msg != prev_error:
            logging.error(f"Simulation error: {error_msg}")
        return [
            "Error calculating slippage",
            "Error calculating fees",
            "Error calculating impact",
            "Error calculating total cost",
            "Error calculating maker probability",
            "N/A",
            False,
            error_msg
        ]

@app.callback(
    Output("log-output", "value"),
    [Input("log-update", "n_intervals")]
)
def update_logs(_):
    try:
        with open("event.log", "r") as f:
            log_lines = f.readlines()
            # Show only the last 100 lines for clarity
            return "".join(log_lines[-100:])
    except Exception:
        return "Event log not available"

def run_websocket():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ws_client.run(ob_processor))

if __name__ == "__main__":
    # Start WebSocket in background thread
    ws_thread = threading.Thread(target=run_websocket, daemon=True)
    ws_thread.start()
    
    app.run(debug=False, port=8050)