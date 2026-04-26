import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import hashlib

app = Flask(__name__)

SECRET_TOKEN = "MySuperSecretKey123"

# File paths
MODEL_FILE = "setup_predictor_model.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "training_data.csv"
TRADE_LOG_FILE = "trade_log.json"

# Global variables
model = None
scaler = None
trade_history = []

# Load trade history if exists
if os.path.exists(TRADE_LOG_FILE):
    with open(TRADE_LOG_FILE, 'r') as f:
        trade_history = json.load(f)

# ====================================================
# INTELLIGENT ASSISTANT LAYER
# ====================================================

class TradingAssistant:
    """AI-powered trading assistant that explains decisions"""
    
    def __init__(self):
        self.conversation_history = []
        self.active_zones = {}
    
    def analyze_zone(self, zone_data):
        """Comprehensive zone analysis with natural language"""
        
        strength = zone_data.get('strength', 0)
        touches = zone_data.get('touches', 0)
        is_bearish = zone_data.get('is_bearish', False)
        price = zone_data.get('price', 0)
        timeframe = zone_data.get('timeframe', '15')
        volume_score = zone_data.get('volume_score', 50)
        
        # Get zone boundaries (from webhook or estimate)
        zone_high = zone_data.get('zone_high', price * 1.002)
        zone_low = zone_data.get('zone_low', price * 0.998)
        
        # Determine zone status
        if price < zone_low:
            zone_status = "REJECTING"
            status_emoji = "🔴"
            action = "SELL" if is_bearish else "BUY"
        elif price > zone_high:
            zone_status = "ABSORBING"
            status_emoji = "🟡"
            action = "WAIT"
        else:
            zone_status = "INSIDE"
            status_emoji = "⏳"
            action = "MONITOR"
        
        # Calculate risk metrics
        zone_range = zone_high - zone_low
        stop_loss = zone_high + (zone_range * 0.5) if is_bearish else zone_low - (zone_range * 0.5)
        risk = abs(price - stop_loss)
        reward = risk * 2  # Assume 2:1 reward ratio
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Generate natural language explanation
        explanation = self._generate_explanation(
            strength, touches, zone_status, is_bearish, 
            volume_score, timeframe, rr_ratio
        )
        
        # Generate actionable advice
        advice = self._generate_advice(action, zone_status, strength, touches)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(strength, touches, volume_score, zone_status)
        
        return {
            "zone_status": zone_status,
            "status_emoji": status_emoji,
            "action": action,
            "explanation": explanation,
            "advice": advice,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "risk_reward": f"1:{rr_ratio:.1f}",
            "entry_price": price if action == "SELL" or action == "BUY" else None
        }
    
    def _generate_explanation(self, strength, touches, zone_status, is_bearish, volume_score, timeframe, rr_ratio):
        """Generate human-readable explanation"""
        
        direction = "bearish (sell)" if is_bearish else "bullish (buy)"
        
        if zone_status == "REJECTING":
            explanation = f"""
🔍 ZONE ANALYSIS:
   • This is a {strength}% strength {direction} zone on the {timeframe}min timeframe.
   • Price has broken BELOW the zone - this is a REJECTION signal.
   • The zone has been tested {touches} time(s).
   • Volume score is {volume_score}/100, {'confirming the move' if volume_score > 60 else 'low - be cautious'}.
   • Risk/Reward ratio is 1:{rr_ratio:.1f}.
"""
        elif zone_status == "ABSORBING":
            explanation = f"""
🔍 ZONE ANALYSIS:
   • This is a {strength}% strength {direction} zone on the {timeframe}min timeframe.
   • Price is holding ABOVE the zone - this is ABSORPTION, not rejection.
   • Buyers are absorbing seller pressure, invalidating the short setup.
   • The zone has been tested {touches} time(s) but is not breaking.
   • Volume score is {volume_score}/100 - {'confirming absorption' if volume_score > 60 else 'low - unclear'}.
"""
        else:
            explanation = f"""
🔍 ZONE ANALYSIS:
   • This is a {strength}% strength {direction} zone on the {timeframe}min timeframe.
   • Price is currently INSIDE the zone - waiting for a breakout.
   • A confirmed trade requires price to close BELOW {'the zone' if is_bearish else 'above the zone'}.
   • The zone has been tested {touches} time(s).
   • Volume score is {volume_score}/100.
"""
        
        return explanation.strip()
    
    def _generate_advice(self, action, zone_status, strength, touches):
        """Generate actionable trading advice"""
        
        if action == "SELL":
            return f"""
💡 ACTIONABLE ADVICE:
   ✅ SHORT POSITION: Enter now or on retest of zone
   🛑 STOP LOSS: Place stop loss ABOVE the zone
   🎯 TAKE PROFIT: Target the next support level below
   📊 POSITION SIZE: Use {'full' if strength > 70 else 'half'} position size
   ⚠️ NOTE: {'Zone has been tested ' + str(touches) + ' times - may weaken' if touches > 0 else 'Fresh zone - good'}"""
        
        elif action == "BUY":
            return f"""
💡 ACTIONABLE ADVICE:
   ✅ LONG POSITION: Enter now or on retest of zone
   🛑 STOP LOSS: Place stop loss BELOW the zone
   🎯 TAKE PROFIT: Target the next resistance level above
   📊 POSITION SIZE: Use {'full' if strength > 70 else 'half'} position size
   ⚠️ NOTE: {'Zone has been tested ' + str(touches) + ' times - may weaken' if touches > 0 else 'Fresh zone - good'}"""
        
        elif action == "WAIT":
            return f"""
💡 ACTIONABLE ADVICE:
   ⏳ DO NOT ENTER: Price is absorbing, not rejecting
   👀 WHAT TO WATCH: Wait for price to close BELOW the zone
   📊 POSITION SIZE: Prepare reduced position (half size) for confirmation trade
   🔄 REASSESS: Check flow indicator - needs sellers in control"""
        
        else:  # MONITOR
            return f"""
💡 ACTIONABLE ADVICE:
   👀 MONITOR ONLY: No entry yet
   🔍 WATCH FOR: Price to close {'BELOW' if 'bearish' in action else 'ABOVE'} the zone
   📊 PREPARE: Have your stop loss and target levels ready
   ⏰ TIMING: Wait for candle close for confirmation"""
    
    def _calculate_confidence(self, strength, touches, volume_score, zone_status):
        """Calculate confidence score (0-100)"""
        score = 0
        
        # Strength contributes (max 40)
        score += min(strength * 0.4, 40)
        
        # Volume contributes (max 20)
        score += min(volume_score * 0.2, 20)
        
        # Zone status contributes
        if zone_status == "REJECTING":
            score += 25
        elif zone_status == "INSIDE":
            score += 10
        else:
            score += 0
        
        # Touch penalty (zones lose power after multiple touches)
        score -= min(touches * 5, 20)
        
        return min(max(score, 0), 100)
    
    def log_trade(self, zone_data, decision, result=None):
        """Log trade for performance tracking"""
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "zone_strength": zone_data.get('strength'),
            "direction": "BEARISH" if zone_data.get('is_bearish') else "BULLISH",
            "price": zone_data.get('price'),
            "decision": decision.get('action'),
            "confidence": decision.get('confidence'),
            "result": result  # 'win', 'loss', or None
        }
        trade_history.append(trade_record)
        
        # Save to file
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(trade_history, f, indent=2)
        
        return trade_record

# Initialize assistant
assistant = TradingAssistant()

# ====================================================
# TRAINING FUNCTIONS (kept from original)
# ====================================================

def train_setup_predictor():
    """Train AI to predict which setups are the BEST"""
    global model, scaler
    
    print("🧠 Training Setup Predictor AI...")
    
    if not os.path.exists(DATA_FILE):
        create_enhanced_training_data()
    
    df = pd.read_csv(DATA_FILE)
    print(f"📊 Loaded {len(df)} historical setups for training")
    
    feature_columns = [
        'strength', 'touches', 'reaction_size', 'volume_score',
        'timeframe_value', 'hour_of_day', 'day_of_week',
        'is_bullish', 'volatility', 'confluence_count'
    ]
    
    X = df[feature_columns].values
    y = df['setup_quality'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    predictions = model.predict(X_scaled)
    mae = np.mean(np.abs(predictions - y))
    print(f"✅ Model trained! Average prediction error: {mae:.1f} points")
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Model saved to {MODEL_FILE}")

def create_enhanced_training_data():
    """Create realistic training data"""
    np.random.seed(42)
    n_samples = 5000
    
    strength = np.random.randint(0, 100, n_samples)
    touches = np.random.randint(0, 5, n_samples)
    reaction_size = np.random.uniform(0, 0.01, n_samples)
    volume_score = np.random.randint(0, 100, n_samples)
    timeframe_value = np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    hour_of_day = np.random.randint(0, 24, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    is_bullish = np.random.randint(0, 2, n_samples)
    volatility = np.random.uniform(0.001, 0.02, n_samples)
    confluence_count = np.random.randint(1, 6, n_samples)
    
    setup_quality = np.zeros(n_samples)
    
    for i in range(n_samples):
        quality = 0
        quality += strength[i] * 0.4
        quality += min(touches[i] * 10, 20)
        if reaction_size[i] > 0.005:
            quality += 15
        elif reaction_size[i] > 0.002:
            quality += 8
        quality += volume_score[i] * 0.15
        if 8 <= hour_of_day[i] <= 12:
            quality += 10
        quality += min(confluence_count[i] * 5, 15)
        setup_quality[i] = min(100, quality)
    
    df = pd.DataFrame({
        'strength': strength, 'touches': touches, 'reaction_size': reaction_size,
        'volume_score': volume_score, 'timeframe_value': timeframe_value,
        'hour_of_day': hour_of_day, 'day_of_week': day_of_week,
        'is_bullish': is_bullish, 'volatility': volatility,
        'confluence_count': confluence_count, 'setup_quality': setup_quality
    })
    
    df.to_csv(DATA_FILE, index=False)
    print(f"✅ Created enhanced training data: {DATA_FILE}")

def predict_setup_quality(features_dict):
    """Predict quality score"""
    global model, scaler
    
    if model is None:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_FILE, 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Loaded existing setup predictor model")
        else:
            train_setup_predictor()
    
    feature_order = [
        'strength', 'touches', 'reaction_size', 'volume_score',
        'timeframe_value', 'hour_of_day', 'day_of_week',
        'is_bullish', 'volatility', 'confluence_count'
    ]
    
    features = np.array([[features_dict[f] for f in feature_order]])
    features_scaled = scaler.transform(features)
    quality = model.predict(features_scaled)[0]
    
    return min(100, max(0, quality))

# ====================================================
# WEBHOOK ENDPOINT with INTELLIGENT ASSISTANT
# ====================================================

@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive zone data and return AI prediction with intelligent assistant"""
    
    data = request.get_json()
    
    if not data or data.get('token') != SECRET_TOKEN:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    print("\n" + "="*70)
    print("🤖 INTELLIGENT TRADING ASSISTANT")
    print("="*70)
    print(f"📡 Alert received at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Extract features
    strength = data.get('strength', 50)
    touches = data.get('touches', 0)
    reaction_size = data.get('reaction_size', 0.003)
    volume_score = data.get('volume_score', 50)
    candle_range = data.get('candle_range', 0.008)
    is_bearish = data.get('is_bearish', False)
    timeframe = data.get('timeframe', '15')
    price = data.get('price', 0)
    
    # Zone boundaries
    zone_high = data.get('zone_high', price * 1.002)
    zone_low = data.get('zone_low', price * 0.998)
    
    # Time features
    now = datetime.now()
    timeframe_map = {'5': 1, '15': 2, '60': 3, '240': 4}
    timeframe_value = timeframe_map.get(str(timeframe), 2)
    
    # Prepare for AI prediction
    features = {
        'strength': strength, 'touches': touches, 'reaction_size': reaction_size,
        'volume_score': volume_score, 'timeframe_value': timeframe_value,
        'hour_of_day': now.hour, 'day_of_week': now.weekday(),
        'is_bullish': 1 if not is_bearish else 0,
        'volatility': candle_range * 100, 'confluence_count': 1
    }
    
    # Get AI prediction
    setup_quality = predict_setup_quality(features)
    
    # Get assistant analysis
    zone_data_for_assistant = {
        'strength': strength, 'touches': touches, 'is_bearish': is_bearish,
        'price': price, 'timeframe': timeframe, 'volume_score': volume_score,
        'zone_high': zone_high, 'zone_low': zone_low
    }
    
    assistant_decision = assistant.analyze_zone(zone_data_for_assistant)
    
    # Print assistant output
    print(f"""
    {'='*70}
    📊 ZONE SUMMARY
    {'='*70}
    📍 Price: {price}
    📐 Zone: [{zone_low:.5f} - {zone_high:.5f}]
    💪 Strength: {strength}%
    👆 Touches: {touches}
    📈 Direction: {'BEARISH (SELL)' if is_bearish else 'BULLISH (BUY)'}
    """)
    
    print(assistant_decision['explanation'])
    print(assistant_decision['advice'])
    
    print(f"""
    {'='*70}
    🎯 FINAL VERDICT
    {'='*70}
    {assistant_decision['status_emoji']} Action: {assistant_decision['action']}
    📊 Confidence: {assistant_decision['confidence']:.1f}%
    🤖 AI Setup Quality: {setup_quality:.1f}%
    ⚠️ Risk/Reward: {assistant_decision['risk_reward']}
    """)
    
    if assistant_decision['stop_loss']:
        print(f"🛑 Suggested Stop Loss: {assistant_decision['stop_loss']:.5f}")
    
    print("="*70 + "\n")
    
    # Log the trade decision
    assistant.log_trade(zone_data_for_assistant, assistant_decision)
    
    # Return response
    response = {
        "status": "success",
        "setup_quality": round(setup_quality, 1),
        "assistant": {
            "action": assistant_decision['action'],
            "confidence": assistant_decision['confidence'],
            "explanation": assistant_decision['explanation'],
            "advice": assistant_decision['advice'],
            "stop_loss": assistant_decision['stop_loss'],
            "risk_reward": assistant_decision['risk_reward']
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(response), 200

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Simple web dashboard to see trade history"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Assistant Dashboard</title>
        <style>
            body { font-family: Arial; margin: 20px; background: #1a1a2e; color: white; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #333; padding: 8px; text-align: left; }
            th { background: #16213e; }
            .win { color: #00ff00; }
            .loss { color: #ff4444; }
            .pending { color: #ffaa00; }
        </style>
    </head>
    <body>
        <h1>🤖 Trading Assistant Dashboard</h1>
        <h2>Trade History</h2>
        <table>
            <tr>
                <th>Time</th>
                <th>Zone Strength</th>
                <th>Direction</th>
                <th>Decision</th>
                <th>Confidence</th>
                <th>Result</th>
            </tr>
            {% for trade in trades %}
            <tr>
                <td>{{ trade.timestamp[:19] }}</td>
                <td>{{ trade.zone_strength }}%</td>
                <td>{{ trade.direction }}</td>
                <td>{{ trade.decision }}</td>
                <td>{{ trade.confidence }}%</td>
                <td class="{{ trade.result if trade.result else 'pending' }}">{{ trade.result or 'Pending' }}</td>
            </tr>
            {% endfor %}
        </table>
        <p>Total Trades: {{ trades|length }}</p>
    </body>
    </html>
    ''', trades=trade_history[-50:])  # Show last 50 trades

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "trades_logged": len(trade_history),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/train', methods=['POST'])
def train():
    train_setup_predictor()
    return jsonify({"status": "success", "message": "Model retrained"}), 200

if __name__ == '__main__':
    print("="*70)
    print("🤖 MONEY GLITCH AI - INTELLIGENT TRADING ASSISTANT")
    print("="*70)
    print("Features enabled:")
    print("  ✅ Natural language explanations")
    print("  ✅ Actionable trading advice")
    print("  ✅ Trade logging and history")
    print("  ✅ Confidence scoring")
    print("  ✅ Risk/Reward calculation")
    print("="*70)
    
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Loaded existing setup predictor model")
    else:
        train_setup_predictor()
    
    print(f"\n🚀 Starting Flask server on port 5000...")
    print("📡 Webhook endpoint: http://localhost:5000/webhook")
    print("📊 Dashboard: http://localhost:5000/dashboard")
    print("🔒 Security token: " + SECRET_TOKEN)
    print("="*70)
    
    app.run(host='0.0.0.0', port=5000, debug=True)