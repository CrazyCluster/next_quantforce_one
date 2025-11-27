from flask import Flask, jsonify
from next_quantforce_one import fetch_info, run_daily
from next_parameter_optimizer_one_mc import run_optimize
import argparse
import logging
import datetime
import sys

# -----------------------------------------------------
# Logging konfigurieren (wichtig für Render Logs)
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# -----------------------------------------------------
# Flask App Setup
# -----------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------
# Health Check / Root Route
# -----------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "alive",
        "message": "🚀 QuantForce One is operational.",
        "time": datetime.datetime.now().isoformat() + "Z"
    })

# -----------------------------------------------------
# Daily Task Endpoint (Triggerbar per curl)
# -----------------------------------------------------
@app.route("/run_daily", methods=["POST"])
def run_daily_trigger():
    logging.info("⚙️  Received weekly run trigger...")
    try:
        result = run_daily(load_paramaters=True)
        logging.info("✅ Daily task completed successfully")
        return jsonify({
            "success": True,
            "message": "Daily trading routine executed successfully.",
            "result": result
        }), 200

    except Exception as e:
        logging.error(f"❌ Error during daily task: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/run_weekly", methods=["POST"])
def run_weekly_trigger():
    logging.info("⚙️  Received weekly run trigger...")
    nasdaq_symbols = fetch_info()
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=nasdaq_symbols, help="Symbols to evaluate")
    p.add_argument("--trials", type=int, default=80)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--period", type=str, default="2y")
    p.add_argument("--storage", type=str, default=None, help="sqlite:///optuna.db")
    p.add_argument("--min_trades", type=int, default=3)
    p.add_argument("--n_mc", type=int, default=1000, help="MC sims per symbol (reduce for speed)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    try:
        study, best, out_json, out_csv = run_optimize(symbols=args.symbols, n_trials=args.trials, period=args.period, workers=args.workers, storage=args.storage, min_trades_required=args.min_trades, n_mc=args.n_mc, verbose=args.verbose)
        logging.info("✅ Weekly task completed successfully")
        return jsonify({
            "success": True,
            "message": "Weekly trading routine executed successfully.",
            "result": best
        }), 200

    except Exception as e:
        logging.error(f"❌ Error during weely task: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -----------------------------------------------------
# Start der App (für Render)
# -----------------------------------------------------
if __name__ == "__main__":
    # Render setzt den PORT über eine Umgebungsvariable
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
