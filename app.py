import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
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
# INSTITUTIONAL DETECTION ENGINE
# ====================================================

class InstitutionalDetector:
    """Detects institutional footprints in market activity"""
    
    def __init__(self):
        self.inst_zones = []
        self.volume_history = deque(maxlen=50)
    
    def analyze_candle(self, volume, high, low, close, open_price, is_bullish, is_bearish):
        """
        Analyze a candle for institutional footprints
        
        Returns:
        - is_institutional: Boolean
        - inst_type: "INSTITUTIONAL_BUY", "INSTITUTIONAL_SELL", or "REGULAR"
        - volume_climax: Boolean
        - pattern: String description
        """
        
        # Calculate average volume
        self.volume_history.append(volume)
        avg_volume = sum(self.volume_history) / len(self.volume_history) if self.volume_history else volume
        volume_multiplier = volume / avg_volume if avg_volume > 0 else 1
        
        # Volume climax detection
        is_climax = volume_multiplier >= 1.5
        
        # Wick analysis
        candle_range = high - low
        if candle_range > 0:
            upper_wick_pct = ((high - max(open_price, close)) / candle_range) * 100 if candle_range > 0 else 0
            lower_wick_pct = ((min(open_price, close) - low) / candle_range) * 100 if candle_range > 0 else 0
            body_pct = 100 - (upper_wick_pct + lower_wick_pct)
        else:
            upper_wick_pct = 0
            lower_wick_pct = 0
            body_pct = 100
        
        # Rejection detection
        has_rejection = upper_wick_pct > 60 or lower_wick_pct > 60
        
        # Body dominance detection (institutional control)
        is_imbalance = body_pct > 60
        
        # Calculate institutional score
        score = 0
        if is_climax:
            score += 1
        if has_rejection:
            score += 1
        if is_imbalance:
            score += 1
        
        is_institutional = score >= 2
        
        # Determine institutional type
        if is_institutional:
            if is_bullish:
                inst_type = "INSTITUTIONAL_BUY"
                pattern = "High-volume bullish rejection + body dominance" if has_rejection and is_imbalance else "Volume climax with bullish structure"
            elif is_bearish:
                inst_type = "INSTITUTIONAL_SELL"
                pattern = "High-volume bearish rejection + body dominance" if has_rejection and is_imbalance else "Volume climax with bearish structure"
            else:
                inst_type = "INSTITUTIONAL_NEUTRAL"
                pattern = "Volume climax but direction unclear"
        else:
            inst_type = "REGULAR"
            pattern = "Normal market activity"
        
        # Create detailed analysis
        analysis = {
            "is_institutional": is_institutional,
            "inst_type": inst_type,
            "inst_type_display": "🏦 INSTITUTIONAL" if is_institutional else "👤 REGULAR",
            "volume_climax": is_climax,
            "volume_multiplier": round(volume_multiplier, 2),
            "pattern": pattern,
            "has_rejection": has_rejection,
            "is_imbalance": is_imbalance,
            "upper_wick_pct": round(upper_wick_pct, 1),
            "lower_wick_pct": round(lower_wick_pct, 1),
            "body_pct": round(body_pct, 1),
            "score": score
        }
        
        # Store institutional zones for future reference
        if is_institutional:
            self.inst_zones.append({
                "timestamp": datetime.now().isoformat(),
                "type": inst_type,
                "volume_multiplier": volume_multiplier
            })
            if len(self.inst_zones) > 100:
                self.inst_zones.pop(0)
        
        return analysis

# Initialize institutional detector
inst_detector = InstitutionalDetector()

# ====================================================
# INTELLIGENT ASSISTANT LAYER WITH INSTITUTIONAL DATA
# ====================================================

class TradingAssistant:
    """AI-powered trading assistant that explains decisions with institutional context"""
    
    def __init__(self):
        self.conversation_history = []
        self.active_zones = {}
        self.inst_zones = []
        self.regular_zones = []
    
    def analyze_zone(self, zone_data, flow_data, inst_data):
        """Comprehensive zone analysis with natural language, flow, and institutional data"""
        
        strength = zone_data.get('strength', 0)
        touches = zone_data.get('touches', 0)
        is_bearish = zone_data.get('is_bearish', False)
        price = zone_data.get('price', 0)
        timeframe = zone_data.get('timeframe', '15')
        volume_score = zone_data.get('volume_score', 50)
        
        # Get zone boundaries
        zone_high = zone_data.get('zone_high', price * 1.002)
        zone_low = zone_data.get('zone_low', price * 0.998)
        zone_range = zone_high - zone_low
        
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
        
        # Risk metrics
        stop_loss = zone_high + (zone_range * 0.5) if is_bearish else zone_low - (zone_range * 0.5)
        risk = abs(price - stop_loss)
        reward = risk * 2
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Entry price suggestions
        if is_bearish:
            primary_entry = zone_low
            aggressive_entry = price if price < zone_high else zone_high
            conservative_entry = zone_low - (zone_range * 0.2)
        else:
            primary_entry = zone_high
            aggressive_entry = price if price > zone_low else zone_low
            conservative_entry = zone_high + (zone_range * 0.2)
        
        # Combine flow and institutional data
        flow = flow_data['flow']
        flow_strength = flow_data['flow_strength']
        flow_confluence = flow_data['confluence']
        
        is_institutional = inst_data['is_institutional']
        inst_type = inst_data['inst_type']
        volume_climax = inst_data['volume_climax']
        volume_multiplier = inst_data['volume_multiplier']
        inst_pattern = inst_data['pattern']
        
        # Calculate conviction boost
        conviction_boost = 15 if is_institutional else 0
        
        # Final decision based on zone, flow, and institutional data
        if zone_status == "REJECTING" and flow_confluence:
            if is_institutional:
                final_action = "STRONG_ENTRY"
                final_emoji = "🚀"
                conviction = "EXTREME"
                confidence = min(100, strength + flow_strength + conviction_boost)
            else:
                final_action = "STRONG_ENTRY"
                final_emoji = "📈"
                conviction = "HIGH"
                confidence = min(100, strength + flow_strength)
            recommended_entry = primary_entry
        elif zone_status == "REJECTING" and not flow_confluence:
            if is_institutional:
                final_action = "CONFLICT_ENTRY"
                final_emoji = "⚠️"
                conviction = "MEDIUM"
                confidence = min(100, strength * 0.6 + conviction_boost)
            else:
                final_action = "CONFLICT_ENTRY"
                final_emoji = "⚠️"
                conviction = "LOW"
                confidence = min(100, strength * 0.6)
            recommended_entry = aggressive_entry
        elif zone_status == "ABSORBING":
            final_action = "WAIT"
            final_emoji = "⏳"
            conviction = "NONE"
            confidence = 30
            recommended_entry = None
        else:
            final_action = "MONITOR"
            final_emoji = "👀"
            conviction = "NONE"
            confidence = 50
            recommended_entry = None
        
        # Generate explanations
        explanation = self._generate_explanation(
            strength, touches, zone_status, is_bearish, 
            volume_score, timeframe, rr_ratio, flow, flow_strength,
            is_institutional, inst_pattern, volume_climax, volume_multiplier
        )
        
        advice = self._generate_advice(
            final_action, zone_status, strength, touches, flow, 
            recommended_entry, primary_entry, aggressive_entry, 
            conservative_entry, stop_loss, is_institutional, conviction
        )
        
        return {
            "zone_status": zone_status,
            "status_emoji": status_emoji,
            "action": final_action,
            "action_emoji": final_emoji,
            "conviction": conviction,
            "explanation": explanation,
            "advice": advice,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "risk_reward": f"1:{rr_ratio:.1f}",
            "entry_price": recommended_entry,
            "primary_entry": primary_entry,
            "aggressive_entry": aggressive_entry,
            "conservative_entry": conservative_entry,
            "flow_confluence": flow_confluence,
            "flow": flow,
            "flow_strength": flow_strength,
            "is_institutional": is_institutional,
            "inst_type": inst_type,
            "volume_climax": volume_climax,
            "volume_multiplier": volume_multiplier,
            "inst_pattern": inst_pattern
        }
    
    def _generate_explanation(self, strength, touches, zone_status, is_bearish, volume_score, timeframe, rr_ratio, flow, flow_strength, is_institutional, inst_pattern, volume_climax, volume_multiplier):
        """Generate human-readable explanation with institutional data"""
        
        direction = "bearish (sell)" if is_bearish else "bullish (buy)"
        inst_prefix = "🏦 INSTITUTIONAL: " if is_institutional else ""
        
        if zone_status == "REJECTING":
            explanation = f"""
🔍 ZONE ANALYSIS:
   • {strength}% strength {direction} zone on {timeframe}min timeframe
   • Price broke BELOW the zone - REJECTION confirmed
   • Tested {touches} time(s)
   • Volume score: {volume_score}/100
   {f'   • VOLUME CLIMAX: {volume_multiplier}x average' if volume_climax else ''}
   • Risk/Reward: 1:{rr_ratio:.1f}
   • Market Flow: {flow} ({flow_strength:.0f}% strength)
   • {inst_prefix}{inst_pattern}
"""
        elif zone_status == "ABSORBING":
            explanation = f"""
🔍 ZONE ANALYSIS:
   • {strength}% strength {direction} zone on {timeframe}min timeframe
   • Price is ABOVE the zone - ABSORPTION happening
   • Buyers absorbing pressure - invalidates setup
   • Volume score: {volume_score}/100
   • Market Flow: {flow} ({flow_strength:.0f}% strength)
   • {inst_prefix}{inst_pattern}
"""
        else:
            explanation = f"""
🔍 ZONE ANALYSIS:
   • {strength}% strength {direction} zone on {timeframe}min timeframe
   • Price INSIDE zone - waiting for breakout
   • Need close {'BELOW' if is_bearish else 'ABOVE'} for entry
   • Volume score: {volume_score}/100
   • Market Flow: {flow} ({flow_strength:.0f}% strength)
   • {inst_prefix}{inst_pattern}
"""
        
        return explanation.strip()
    
    def _generate_advice(self, action, zone_status, strength, touches, flow, recommended_entry, primary_entry, aggressive_entry, conservative_entry, stop_loss, is_institutional, conviction):
        """Generate actionable trading advice with institutional context"""
        
        if action == "STRONG_ENTRY":
            return f"""
💡 ACTIONABLE ADVICE:
   ✅ ENTRY: {'SHORT' if 'bearish' in str(action) else 'LONG'} position
   📊 CONVICTION: {conviction}
   📍 ENTRY PRICE: {recommended_entry:.5f}
   
   🎯 ENTRY OPTIONS:
      • Primary Entry: {primary_entry:.5f} (breakout confirmation)
      • Aggressive Entry: {aggressive_entry:.5f} (early entry)
      • Conservative Entry: {conservative_entry:.5f} (wait for follow-through)
   
   🛑 STOP LOSS: {stop_loss:.5f}
   📊 POSITION SIZE: Full ({conviction} conviction)
   🔄 FLOW: {flow} - aligns with trade direction
   {f'🏦 INSTITUTIONAL: YES - High confidence trade' if is_institutional else ''}"""
        
        elif action == "CONFLICT_ENTRY":
            return f"""
💡 ACTIONABLE ADVICE:
   ⚠️ REDUCED ENTRY: Half position only
   📊 CONVICTION: {conviction}
   📍 ENTRY PRICE: {recommended_entry:.5f} (if taken)
   
   🎯 ENTRY OPTIONS:
      • Aggressive Entry: {aggressive_entry:.5f}
      • Conservative Entry: {conservative_entry:.5f}
   
   🛑 STOP LOSS: {stop_loss:.5f}
   📊 SIZE: Half position (conflicting signals)
   🔄 FLOW: {flow} - conflicts with zone direction
   👀 WAIT: For alignment for full position
   {f'🏦 INSTITUTIONAL: YES - But flow conflict' if is_institutional else ''}"""
        
        elif action == "WAIT":
            return f"""
💡 ACTIONABLE ADVICE:
   ⏳ DO NOT ENTER: Price absorbing, not rejecting
   👀 WATCH FOR: Close {'BELOW' if 'bearish' in str(action) else 'ABOVE'} zone
   📍 PRICE LEVELS:
      • Zone Entry: {primary_entry:.5f}
      • Stop Loss: {stop_loss:.5f}
   🔄 FLOW: {flow}
   📊 PREPARE: Have stop loss ready for confirmation"""
        
        else:
            return f"""
💡 ACTIONABLE ADVICE:
   👀 MONITOR ONLY: No entry yet
   🔍 WATCH: Price movement and candle closes
   📍 KEY LEVELS:
      • Zone Low: {primary_entry if 'bearish' else conservative_entry:.5f}
      • Zone High: {conservative_entry if 'bearish' else primary_entry:.5f}
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
            "conviction": decision.get('conviction'),
            "entry_price": decision.get('entry_price'),
            "stop_loss": decision.get('stop_loss'),
            "flow": decision.get('flow', 'UNKNOWN'),
            "flow_confluence": decision.get('flow_confluence', False),
            "is_institutional": decision.get('is_institutional', False),
            "inst_type": decision.get('inst_type', ''),
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
# WEBHOOK ENDPOINT with INSTITUTIONAL DETECTION
# ====================================================

@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive zone data and return AI prediction with institutional detection"""
    
    data = request.get_json()
    
    if not data or data.get('token') != SECRET_TOKEN:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    print("\n" + "="*70)
    print("🤖 INTELLIGENT TRADING ASSISTANT WITH INSTITUTIONAL DETECTION")
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
    
    # Analyze institutional footprint
    simulated_volume = volume_score * 100
    simulated_high = price + (candle_range / 2)
    simulated_low = price - (candle_range / 2)
    is_bullish = not is_bearish
    
    inst_data = inst_detector.analyze_candle(
        simulated_volume, simulated_high, simulated_low, 
        price, price, is_bullish, is_bearish
    )
    
    # Get assistant analysis with flow and institutional data
    zone_data_for_assistant = {
        'strength': strength, 'touches': touches, 'is_bearish': is_bearish,
        'price': price, 'timeframe': timeframe, 'volume_score': volume_score,
        'zone_high': zone_high, 'zone_low': zone_low
    }
    
    assistant_decision = assistant.analyze_zone(zone_data_for_assistant, flow_data, inst_data)
    
    # Print institutional analysis
    print(f"""
    {'='*70}
    🏦 INSTITUTIONAL FOOTPRINT DETECTION
    {'='*70}
    {inst_data['inst_type_display']}
    📊 VOLUME CLIMAX: {'✅ YES' if inst_data['volume_climax'] else '❌ NO'} {f'({inst_data["volume_multiplier"]}x average)' if inst_data['volume_climax'] else ''}
    🔍 PATTERN: {inst_data['pattern']}
    📐 WICK ANALYSIS: Upper: {inst_data['upper_wick_pct']}% | Lower: {inst_data['lower_wick_pct']}% | Body: {inst_data['body_pct']}%
    🏆 INSTITUTIONAL SCORE: {inst_data['score']}/3
    """)
    
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
    📍 Current Price: {price}
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
    📊 Conviction Level: {assistant_decision['conviction']}
    📊 Confidence: {assistant_decision['confidence']:.1f}%
    🤖 AI Setup Quality: {setup_quality:.1f}%
    ⚠️ Risk/Reward: {assistant_decision['risk_reward']}
    🔄 Flow Alignment: {'✅ YES' if assistant_decision['flow_confluence'] else '❌ NO'}
    🏦 Institutional Backing: {'✅ CONFIRMED' if assistant_decision['is_institutional'] else '❌ NOT DETECTED'}
    """)
    
    if assistant_decision['stop_loss']:
        print(f"🛑 Suggested Stop Loss: {assistant_decision['stop_loss']:.5f}")
    
    if assistant_decision['entry_price']:
        print(f"""
    📍 ENTRY PRICE SUGGESTIONS:
       • Primary Entry: {assistant_decision['primary_entry']:.5f}
       • Aggressive Entry: {assistant_decision['aggressive_entry']:.5f}
       • Conservative Entry: {assistant_decision['conservative_entry']:.5f}
    """)
    
    print("="*70 + "\n")
    
    # Log the trade decision
    assistant.log_trade(zone_data_for_assistant, assistant_decision)
    
    # Return response
    response = {
        "status": "success",
        "setup_quality": round(setup_quality, 1),
        "flow_analysis": flow_data,
        "institutional_analysis": inst_data,
        "assistant": {
            "action": assistant_decision['action'],
            "confidence": assistant_decision['confidence'],
            "conviction": assistant_decision['conviction'],
            "explanation": assistant_decision['explanation'],
            "advice": assistant_decision['advice'],
            "stop_loss": assistant_decision['stop_loss'],
            "risk_reward": assistant_decision['risk_reward'],
            "entry_price": assistant_decision['entry_price'],
            "primary_entry": assistant_decision['primary_entry'],
            "aggressive_entry": assistant_decision['aggressive_entry'],
            "conservative_entry": assistant_decision['conservative_entry'],
            "flow_confluence": assistant_decision['flow_confluence'],
            "is_institutional": assistant_decision['is_institutional'],
            "inst_type": assistant_decision['inst_type'],
            "volume_climax": assistant_decision['volume_climax'],
            "volume_multiplier": assistant_decision['volume_multiplier']
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(response), 200

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
    print("🤖 MONEY GLITCH AI - WITH INSTITUTIONAL DETECTION")
    print("="*70)
    print("Features enabled:")
    print("  ✅ Buyer/Seller flow detection")
    print("  ✅ Institutional footprint detection")
    print("  ✅ Volume climax analysis")
    print("  ✅ Entry price suggestions (3 levels)")
    print("  ✅ Stop loss calculation")
    print("  ✅ Conviction scoring (EXTREME/HIGH/MEDIUM/LOW/NONE)")
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
    print("🔒 Security token: " + SECRET_TOKEN)
    print("="*70)
    
    app.run(host='0.0.0.0', port=5000, debug=True)