import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from collections import deque

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
flow_history = deque(maxlen=100)

# Load trade history if exists
if os.path.exists(TRADE_LOG_FILE):
    with open(TRADE_LOG_FILE, 'r') as f:
        trade_history = json.load(f)

# ====================================================
# BUYER/SELLER FLOW DETECTION ENGINE
# ====================================================

class FlowDetector:
    """Detects whether buyers or sellers are in control"""
    
    def __init__(self):
        self.volume_history = deque(maxlen=20)
        self.flow_scores = deque(maxlen=10)
    
    def analyze_flow(self, volume_score, price_change, is_bearish_zone):
        """
        Analyze market flow to determine who is in control
        
        Parameters:
        - volume_score: 0-100 volume intensity
        - price_change: recent price movement (positive = up, negative = down)
        - is_bearish_zone: True for sell zones, False for buy zones
        
        Returns:
        - flow: "BUYERS", "SELLERS", or "NEUTRAL"
        - flow_strength: 0-100 how strong the flow is
        - explanation: text explanation
        """
        
        # Determine price direction
        price_up = price_change > 0
        price_down = price_change < 0
        
        # Calculate flow score
        flow_score = 0
        
        if price_up and volume_score > 50:
            flow_score = volume_score  # Buyers strength
        elif price_down and volume_score > 50:
            flow_score = -volume_score  # Sellers strength
        elif volume_score <= 50:
            flow_score = volume_score * 0.1  # Weak/neutral
        
        # Add to history
        self.flow_scores.append(flow_score)
        
        # Determine average flow
        avg_flow = sum(self.flow_scores) / len(self.flow_scores) if self.flow_scores else 0
        
        # Determine who is in control
        if avg_flow > 20:
            flow = "BUYERS"
            flow_strength = min(abs(avg_flow), 100)
            flow_emoji = "🟢"
        elif avg_flow < -20:
            flow = "SELLERS"
            flow_strength = min(abs(avg_flow), 100)
            flow_emoji = "🔴"
        else:
            flow = "NEUTRAL"
            flow_strength = abs(avg_flow)
            flow_emoji = "🟡"
        
        # Generate explanation
        if flow == "BUYERS":
            explanation = f"Buyers are in control ({flow_strength:.0f}% strength). Volume confirms upward pressure."
            if is_bearish_zone:
                advice = "⚠️ CONFLICT: Bearish zone but buyers are leading - wait for seller confirmation"
            else:
                advice = "✅ CONFIRMATION: Bullish zone with buyer dominance - good setup"
        elif flow == "SELLERS":
            explanation = f"Sellers are in control ({flow_strength:.0f}% strength). Volume confirms downward pressure."
            if is_bearish_zone:
                advice = "✅ CONFIRMATION: Bearish zone with seller dominance - good setup"
            else:
                advice = "⚠️ CONFLICT: Bullish zone but sellers are leading - wait for buyer confirmation"
        else:
            explanation = f"Market is neutral ({flow_strength:.0f}% strength). No clear dominant side."
            advice = "⏳ WAIT: Let the market show direction before entering"
        
        return {
            "flow": flow,
            "flow_emoji": flow_emoji,
            "flow_strength": round(flow_strength, 1),
            "flow_score": round(avg_flow, 1),
            "explanation": explanation,
            "advice": advice,
            "confluence": (flow == "SELLERS" and is_bearish_zone) or (flow == "BUYERS" and not is_bearish_zone)
        }

# Initialize flow detector
flow_detector = FlowDetector()

# ====================================================
# INTELLIGENT ASSISTANT LAYER
# ====================================================

class TradingAssistant:
    """AI-powered trading assistant that explains decisions"""
    
    def __init__(self):
        self.conversation_history = []
        self.active_zones = {}
    
    def analyze_zone(self, zone_data, flow_data):
        """Comprehensive zone analysis with natural language and flow"""
        
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
        reward = risk * 2
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Combine zone analysis with flow analysis
        flow = flow_data['flow']
        flow_strength = flow_data['flow_strength']
        flow_confluence = flow_data['confluence']
        
        # Final decision based on both zone and flow
        if zone_status == "REJECTING" and flow_confluence:
            final_action = "STRONG_ENTRY"
            final_emoji = "🚀"
            confidence = min(100, strength + flow_strength)
        elif zone_status == "REJECTING" and not flow_confluence:
            final_action = "CONFLICT_ENTRY"
            final_emoji = "⚠️"
            confidence = min(100, strength * 0.6)
        elif zone_status == "ABSORBING":
            final_action = "WAIT"
            final_emoji = "⏳"
            confidence = 30
        else:
            final_action = "MONITOR"
            final_emoji = "👀"
            confidence = 50
        
        # Generate explanation with flow data
        explanation = self._generate_explanation(
            strength, touches, zone_status, is_bearish, 
            volume_score, timeframe, rr_ratio, flow, flow_strength
        )
        
        # Generate advice with flow data
        advice = self._generate_advice(final_action, zone_status, strength, touches, flow)
        
        return {
            "zone_status": zone_status,
            "status_emoji": status_emoji,
            "action": final_action,
            "action_emoji": final_emoji,
            "explanation": explanation,
            "advice": advice,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "risk_reward": f"1:{rr_ratio:.1f}",
            "entry_price": price if final_action == "STRONG_ENTRY" else None,
            "flow_confluence": flow_confluence,
            "flow": flow,
            "flow_strength": flow_strength
        }
    
    def _generate_explanation(self, strength, touches, zone_status, is_bearish, volume_score, timeframe, rr_ratio, flow, flow_strength):
        """Generate human-readable explanation with flow data"""
        
        direction = "bearish (sell)" if is_bearish else "bullish (buy)"
        
        if zone_status == "REJECTING":
            explanation = f"""
🔍 ZONE ANALYSIS:
   • This is a {strength}% strength {direction} zone on the {timeframe}min timeframe.
   • Price has broken BELOW the zone - this is a REJECTION signal.
   • The zone has been tested {touches} time(s).
   • Volume score is {volume_score}/100.
   • Risk/Reward ratio is 1:{rr_ratio:.1f}.
   • Market Flow: {flow} ({flow_strength:.0f}% strength)
"""
        elif zone_status == "ABSORBING":
            explanation = f"""
🔍 ZONE ANALYSIS:
   • This is a {strength}% strength {direction} zone on the {timeframe}min timeframe.
   • Price is holding ABOVE the zone - this is ABSORPTION, not rejection.
   • Buyers are absorbing seller pressure, invalidating the short setup.
   • The zone has been tested {touches} time(s) but is not breaking.
   • Volume score is {volume_score}/100.
   • Market Flow: {flow} ({flow_strength:.0f}% strength)
"""
        else:
            explanation = f"""
🔍 ZONE ANALYSIS:
   • This is a {strength}% strength {direction} zone on the {timeframe}min timeframe.
   • Price is currently INSIDE the zone - waiting for a breakout.
   • A confirmed trade requires price to close BELOW {'the zone' if is_bearish else 'above the zone'}.
   • The zone has been tested {touches} time(s).
   • Volume score is {volume_score}/100.
   • Market Flow: {flow} ({flow_strength:.0f}% strength)
"""
        
        return explanation.strip()
    
    def _generate_advice(self, action, zone_status, strength, touches, flow):
        """Generate actionable trading advice with flow"""
        
        if action == "STRONG_ENTRY":
            return f"""
💡 ACTIONABLE ADVICE:
   ✅ ENTRY: {'SHORT' if 'bearish' in str(action) else 'LONG'} position
   🛑 STOP LOSS: Place {'above' if 'bearish' in str(action) else 'below'} the zone
   🎯 TAKE PROFIT: Next {'support' if 'bearish' in str(action) else 'resistance'} level
   📊 SIZE: Full position (high confidence)
   🔄 FLOW: {flow} - aligns with trade direction"""
        
        elif action == "CONFLICT_ENTRY":
            return f"""
💡 ACTIONABLE ADVICE:
   ⚠️ REDUCED ENTRY: Half position only
   🛑 STOP LOSS: Wider than normal (2x zone width)
   📊 SIZE: Half position (conflicting signals)
   🔄 FLOW: {flow} - conflicts with zone direction
   👀 WAIT: Until flow aligns with zone"""
        
        elif action == "WAIT":
            return f"""
💡 ACTIONABLE ADVICE:
   ⏳ DO NOT ENTER: Price absorbing, not rejecting
   👀 WATCH FOR: Close {'BELOW' if 'bearish' in str(action) else 'ABOVE'} zone
   🔄 FLOW: {flow}
   📊 PREPARE: Have stop loss ready for confirmation"""
        
        else:
            return f"""
💡 ACTIONABLE ADVICE:
   👀 MONITOR ONLY: No entry yet
   🔍 WATCH: Price movement and candle closes
   🔄 FLOW: {flow}
   ⏰ TIMING: Wait for breakout confirmation"""
    
    def log_trade(self, zone_data, decision, result=None):
        """Log trade for performance tracking"""
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "zone_strength": zone_data.get('strength'),
            "direction": "BEARISH" if zone_data.get('is_bearish') else "BULLISH",
            "price": zone_data.get('price'),
            "decision": decision.get('action'),
            "confidence": decision.get('confidence'),
            "flow": decision.get('flow', 'UNKNOWN'),
            "flow_confluence": decision.get('flow_confluence', False),
            "result": result
        }
        trade_history.append(trade_record)
        
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(trade_history, f, indent=2)
        
        return trade_record

# Initialize assistant
assistant = TradingAssistant()

# ====================================================
# TRAINING FUNCTIONS
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
# WEBHOOK ENDPOINT with FLOW DETECTION
# ====================================================

@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive zone data and return AI prediction with intelligent assistant and flow detection"""
    
    data = request.get_json()
    
    if not data or data.get('token') != SECRET_TOKEN:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    print("\n" + "="*70)
    print("🤖 INTELLIGENT TRADING ASSISTANT WITH FLOW DETECTION")
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
    
    # Calculate price change (for flow detection)
    price_change = data.get('price_change', 0.0001)
    
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
    
    # Analyze flow
    flow_data = flow_detector.analyze_flow(volume_score, price_change, is_bearish)
    
    # Get assistant analysis with flow data
    zone_data_for_assistant = {
        'strength': strength, 'touches': touches, 'is_bearish': is_bearish,
        'price': price, 'timeframe': timeframe, 'volume_score': volume_score,
        'zone_high': zone_high, 'zone_low': zone_low
    }
    
    assistant_decision = assistant.analyze_zone(zone_data_for_assistant, flow_data)
    
    # Print flow analysis
    print(f"""
    {'='*70}
    🌊 MARKET FLOW ANALYSIS
    {'='*70}
    {flow_data['flow_emoji']} WHO'S IN CONTROL: {flow_data['flow']}
    📊 FLOW STRENGTH: {flow_data['flow_strength']}%
    📝 EXPLANATION: {flow_data['explanation']}
    ⚡ ADVICE: {flow_data['advice']}
    🔄 FLOW ALIGNMENT: {'✅ YES (Confluence)' if flow_data['confluence'] else '❌ NO (Conflict)'}
    """)
    
    # Print zone summary
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
    {assistant_decision['action_emoji']} Action: {assistant_decision['action']}
    📊 Confidence: {assistant_decision['confidence']:.1f}%
    🤖 AI Setup Quality: {setup_quality:.1f}%
    ⚠️ Risk/Reward: {assistant_decision['risk_reward']}
    🔄 Flow Alignment: {'✅ YES' if assistant_decision['flow_confluence'] else '❌ NO'}
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
        "flow_analysis": flow_data,
        "assistant": {
            "action": assistant_decision['action'],
            "confidence": assistant_decision['confidence'],
            "explanation": assistant_decision['explanation'],
            "advice": assistant_decision['advice'],
            "stop_loss": assistant_decision['stop_loss'],
            "risk_reward": assistant_decision['risk_reward'],
            "flow_confluence": assistant_decision['flow_confluence']
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
            .buyers { color: #00ff00; }
            .sellers { color: #ff4444; }
        </style>
    </head>
    <body>
        <h1>🤖 Trading Assistant Dashboard</h1>
        <h2>Trade History with Flow Analysis</h2>
        <table>
            <tr>
                <th>Time</th>
                <th>Zone Strength</th>
                <th>Direction</th>
                <th>Decision</th>
                <th>Confidence</th>
                <th>Flow</th>
                <th>Confluence</th>
                <th>Result</th>
            </tr>
            {% for trade in trades %}
            <tr>
                <td>{{ trade.timestamp[:19] }}</td>
                <td>{{ trade.zone_strength }}%</td>
                <td>{{ trade.direction }}</td>
                <td>{{ trade.decision }}</td>
                <td>{{ trade.confidence }}%</td>
                <td class="{{ 'buyers' if trade.flow == 'BUYERS' else 'sellers' if trade.flow == 'SELLERS' else 'neutral' }}">{{ trade.flow or 'N/A' }}</td>
                <td>{{ '✅' if trade.flow_confluence else '❌' }}</td>
                <td class="{{ trade.result if trade.result else 'pending' }}">{{ trade.result or 'Pending' }}</td>
            </tr>
            {% endfor %}
        </table>
        <p>Total Trades: {{ trades|length }}</p>
    </body>
    </html>
    ''', trades=trade_history[-50:])

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
    print("🤖 MONEY GLITCH AI - WITH BUYER/SELLER FLOW DETECTION")
    print("="*70)
    print("Features enabled:")
    print("  ✅ Buyer/Seller flow detection")
    print("  ✅ Flow strength scoring")
    print("  ✅ Confluence analysis (flow + zone)")
    print("  ✅ Natural language explanations")
    print("  ✅ Actionable trading advice")
    print("  ✅ Trade logging with flow data")
    print("="*70)
    
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Loaded existing setup predictor model")
    else:
        train_setup_predictor()
    
    print("\n🚀 Starting Flask server on port 5000...")
    print("📡 Webhook endpoint: http://localhost:5000/webhook")
    print("📊 Dashboard: http://localhost:5000/dashboard")
    print("🔒 Security token: " + SECRET_TOKEN)
    print("="*70)
    
    app.run(host='0.0.0.0', port=5000, debug=True)