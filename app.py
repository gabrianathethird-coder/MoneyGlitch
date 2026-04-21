import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

SECRET_TOKEN = "MySuperSecretKey123"

# File paths
MODEL_FILE = "setup_predictor_model.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "training_data.csv"

# Global variables
model = None
scaler = None

# ====================================================
# ADVANCED SETUP PREDICTOR MODEL
# ====================================================

def train_setup_predictor():
    """Train AI to predict which setups are the BEST"""
    global model, scaler
    
    print("🧠 Training Setup Predictor AI...")
    
    # If no training data exists, create enhanced sample data
    if not os.path.exists(DATA_FILE):
        create_enhanced_training_data()
    
    # Load training data
    df = pd.read_csv(DATA_FILE)
    print(f"📊 Loaded {len(df)} historical setups for training")
    
    # Features that predict a "good setup"
    feature_columns = [
        'strength',           # Zone strength (0-100)
        'touches',            # Times touched
        'reaction_size',      # How far price bounced
        'volume_score',       # Volume proxy
        'timeframe_value',    # 1=5min, 2=15min, 3=1h, 4=4h
        'hour_of_day',        # 0-23 trading hour
        'day_of_week',        # 0-6 (Monday=0)
        'is_bullish',         # 1 for bullish, 0 for bearish
        'volatility',         # ATR/candle range
        'confluence_count'    # How many zones overlap
    ]
    
    X = df[feature_columns].values
    y = df['setup_quality'].values  # 0-100 quality score
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest Regressor for quality prediction
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Calculate accuracy metrics
    predictions = model.predict(X_scaled)
    mae = np.mean(np.abs(predictions - y))
    print(f"✅ Model trained! Average prediction error: {mae:.1f} points")
    
    # Save model and scaler
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Model saved to {MODEL_FILE}")

def create_enhanced_training_data():
    """Create realistic training data for setup quality prediction"""
    np.random.seed(42)
    n_samples = 5000
    
    # Generate realistic feature distributions
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
    
    # Calculate setup quality (0-100) based on realistic rules
    setup_quality = np.zeros(n_samples)
    
    for i in range(n_samples):
        quality = 0
        
        # Strength contributes most (40% of score)
        quality += strength[i] * 0.4
        
        # Touches add value (up to 20 points)
        quality += min(touches[i] * 10, 20)
        
        # Good reaction size adds points
        if reaction_size[i] > 0.005:
            quality += 15
        elif reaction_size[i] > 0.002:
            quality += 8
        
        # Volume adds (up to 15 points)
        quality += volume_score[i] * 0.15
        
        # Best hours for trading (London/NY overlap: 8-12 EST)
        if 8 <= hour_of_day[i] <= 12:
            quality += 10
        
        # Confluence adds
        quality += min(confluence_count[i] * 5, 15)
        
        # Cap at 100
        setup_quality[i] = min(100, quality)
    
    df = pd.DataFrame({
        'strength': strength,
        'touches': touches,
        'reaction_size': reaction_size,
        'volume_score': volume_score,
        'timeframe_value': timeframe_value,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_bullish': is_bullish,
        'volatility': volatility,
        'confluence_count': confluence_count,
        'setup_quality': setup_quality
    })
    
    df.to_csv(DATA_FILE, index=False)
    print(f"✅ Created enhanced training data: {DATA_FILE}")
    print(f"   Average setup quality: {df['setup_quality'].mean():.1f}")

def predict_setup_quality(features_dict):
    """Predict quality score for a setup (0-100)"""
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
    
    # Prepare features in correct order
    feature_order = [
        'strength', 'touches', 'reaction_size', 'volume_score',
        'timeframe_value', 'hour_of_day', 'day_of_week',
        'is_bullish', 'volatility', 'confluence_count'
    ]
    
    features = np.array([[features_dict[f] for f in feature_order]])
    features_scaled = scaler.transform(features)
    quality = model.predict(features_scaled)[0]
    
    return min(100, max(0, quality))

def get_trade_recommendation(quality, strength):
    """Generate actionable trade recommendation"""
    if quality >= 80:
        return {
            "action": "🚀 STRONG BUY",
            "confidence": "HIGH",
            "risk_level": "LOW",
            "suggestion": "Take trade immediately when zone is touched"
        }
    elif quality >= 65:
        return {
            "action": "📈 BUY",
            "confidence": "MEDIUM-HIGH",
            "risk_level": "LOW-MEDIUM",
            "suggestion": "Enter on first touch, use normal position size"
        }
    elif quality >= 50:
        return {
            "action": "👀 WATCH",
            "confidence": "MEDIUM",
            "risk_level": "MEDIUM",
            "suggestion": "Wait for confirmation, reduce position size"
        }
    elif quality >= 35:
        return {
            "action": "⚠️ AVOID",
            "confidence": "LOW",
            "risk_level": "HIGH",
            "suggestion": "Look for better setups elsewhere"
        }
    else:
        return {
            "action": "❌ SKIP",
            "confidence": "VERY LOW",
            "risk_level": "VERY HIGH",
            "suggestion": "Do not take this trade"
        }

# ====================================================
# WEBHOOK ENDPOINT
# ====================================================

@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive zone data and return AI prediction"""
    
    data = request.get_json()
    
    # Verify token
    if not data or data.get('token') != SECRET_TOKEN:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    print("\n" + "="*60)
    print(f"📡 Webhook received at {datetime.now()}")
    
    # Extract features
    strength = data.get('strength', 50)
    touches = data.get('touches', 0)
    reaction_size = data.get('reaction_size', 0.003)
    volume_score = data.get('volume_score', 50)
    candle_range = data.get('candle_range', 0.008)
    is_bearish = data.get('is_bearish', False)
    timeframe = data.get('timeframe', '15')
    price = data.get('price', 0)
    
    # Convert timeframe to numeric value
    timeframe_map = {'5': 1, '15': 2, '60': 3, '240': 4}
    timeframe_value = timeframe_map.get(str(timeframe), 2)
    
    # Get current time features
    now = datetime.now()
    hour_of_day = now.hour
    day_of_week = now.weekday()
    
    # Calculate volatility (using candle range)
    volatility = candle_range * 100  # Convert to percentage
    
    # Estimate confluence (simplified)
    confluence_count = 1  # Default, can be enhanced
    
    # Prepare features for prediction
    features = {
        'strength': strength,
        'touches': touches,
        'reaction_size': reaction_size,
        'volume_score': volume_score,
        'timeframe_value': timeframe_value,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_bullish': 1 if not is_bearish else 0,
        'volatility': volatility,
        'confluence_count': confluence_count
    }
    
    # Get AI prediction
    setup_quality = predict_setup_quality(features)
    recommendation = get_trade_recommendation(setup_quality, strength)
    
    # Print detailed analysis
    print(f"📊 ZONE ANALYSIS")
    print(f"   ├─ Strength: {strength}%")
    print(f"   ├─ Touches: {touches}")
    print(f"   ├─ Timeframe: {timeframe}")
    print(f"   ├─ Volume Score: {volume_score}")
    print(f"   └─ Direction: {'BEARISH' if is_bearish else 'BULLISH'}")
    
    print(f"\n🎯 SETUP QUALITY SCORE: {setup_quality:.1f}%")
    print(f"   ├─ Recommendation: {recommendation['action']}")
    print(f"   ├─ Confidence: {recommendation['confidence']}")
    print(f"   ├─ Risk Level: {recommendation['risk_level']}")
    print(f"   └─ Suggestion: {recommendation['suggestion']}")
    
    print("\n" + "="*60)
    
    # Return response
    response = {
        "status": "success",
        "setup_quality": round(setup_quality, 1),
        "recommendation": recommendation,
        "zone_data": {
            "strength": strength,
            "touches": touches,
            "price": price,
            "is_bearish": is_bearish
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(response), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/train', methods=['POST'])
def train():
    """Manually retrain the model with new data"""
    train_setup_predictor()
    return jsonify({"status": "success", "message": "Model retrained"}), 200

if __name__ == '__main__':
    print("="*60)
    print("🤖 MONEY GLITCH AI - ADVANCED SETUP PREDICTOR")
    print("="*60)
    
    # Train or load model
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
    print("\n" + "="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)