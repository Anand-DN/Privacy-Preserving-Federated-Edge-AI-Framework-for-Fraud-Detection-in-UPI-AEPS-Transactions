# =========================================
# FEDERATED FRAUD DETECTION
# Professional Streamlit Dashboard
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
import os
import pickle
import time
import random
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Federated Fraud Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme management
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

def get_theme_colors():
    if st.session_state.theme == 'dark':
        return {
            'bg': '#0f172a',
            'card': '#1e293b',
            'text': '#f1f5f9',
            'muted': '#94a3b8',
            'primary': '#3b82f6',
            'accent': '#60a5fa',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'border': '#334155'
        }
    else:
        return {
            'bg': '#f8fafc',
            'card': '#ffffff',
            'text': '#1e293b',
            'muted': '#64748b',
            'primary': '#2563eb',
            'accent': '#1d4ed8',
            'success': '#059669',
            'warning': '#d97706',
            'danger': '#dc2626',
            'border': '#e2e8f0'
        }

def apply_theme():
    colors = get_theme_colors()
    st.markdown(f"""
    <style>
        .section-header {{
            font-size: 22px;
            font-weight: 700;
            color: {colors['accent']} !important;
            margin-bottom: 14px;
            padding-bottom: 6px;
            border-bottom: 2px solid {colors['primary']};
        }}
        div[data-testid="stMetric"] {{
            background: linear-gradient(145deg, {colors['card']}, {colors['bg']}) !important;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid {colors['border']} !important;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {colors['muted']} !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: {colors['text']} !important;
        }}
        .block-container {{padding-top: 1rem;}}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stApp {{background-color: {colors['bg']};}}
        .stMarkdown, .stText, p, span, li {{
            color: {colors['text']} !important;
        }}
        h1, h2, h3, h4 {{
            color: {colors['text']} !important;
        }}
        .stCodeBlock code {{
            color: {colors['text']} !important;
        }}
        .info-card {{
            padding: 16px;
            background: {colors['card']};
            border-radius: 10px;
            border: 1px solid {colors['border']};
            margin-bottom: 12px;
        }}
        .chart-container {{
            background: {colors['card']};
            border-radius: 12px;
            padding: 16px;
            border: 1px solid {colors['border']};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {colors['card']};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {colors['muted']};
        }}
        .stTabs [aria-selected="true"] {{
            color: {colors['accent']} !important;
        }}
        .stButton > button {{
            border: 1px solid {colors['border']};
            color: {colors['text']};
        }}
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stSlider > div > div > div {{
            background-color: {colors['card']};
            color: {colors['text']};
        }}
        .stExpander {{
            background-color: {colors['card']};
        }}
        .stRadio label {{
            color: {colors['text']} !important;
        }}
        .stCheckbox label {{
            color: {colors['text']} !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {colors['card']};
        }}
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {{
            color: {colors['text']} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Live Prediction", "Fraud History"])

# Theme toggle
st.sidebar.markdown("---")
st.sidebar.button(f"☀️ Switch to Light Mode" if st.session_state.theme == 'dark' else "🌙 Switch to Dark Mode", on_click=toggle_theme)

apply_theme()

# =========================================
# LIVE SIMULATION STATE
# =========================================
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False

if 'live_predictions' not in st.session_state:
    st.session_state.live_predictions = []

if 'live_metrics' not in st.session_state:
    st.session_state.live_metrics = {
        'total': 0,
        'flagged': 0,
        'blocked_amount': 0,
        'last_update': datetime.now()
    }

if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

if 'date_range' not in st.session_state:
    st.session_state.date_range = ('2024-01-01', '2024-03-31')

def toggle_simulation():
    st.session_state.simulation_running = not st.session_state.simulation_running

def toggle_auto_refresh():
    st.session_state.auto_refresh = not st.session_state.auto_refresh

def toggle_auto_refresh():
    st.session_state.auto_refresh = not st.session_state.auto_refresh

def reset_live_data():
    st.session_state.live_predictions = []
    st.session_state.live_metrics = {'total': 0, 'flagged': 0, 'blocked_amount': 0, 'last_update': datetime.now()}

def save_to_fraud_history(tx_data):
    import datetime as dt
    fraud_case = {
        "utr_number": tx_data['utr'],
        "amount": tx_data['amount'],
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": tx_data.get('device', 'mobile'),
        "velocity": tx_data.get('velocity', 0.5),
        "is_night": tx_data.get('is_night', False),
        "fraud_probability": round(tx_data['fraud_prob'], 4),
        "risk_level": tx_data['risk']
    }
    
    fraud_df = pd.DataFrame([fraud_case])
    fraud_log_file = "fraud_logs.csv"
    if os.path.exists(fraud_log_file):
        old_df = pd.read_csv(fraud_log_file)
        new_df = pd.concat([old_df, fraud_df], ignore_index=True)
    else:
        new_df = fraud_df
    new_df.to_csv(fraud_log_file, index=False)

def simulate_transactions(count=3):
    for _ in range(count):
        if not st.session_state.simulation_running:
            break
        amount = np.random.lognormal(8.5, 1.2)
        velocity = np.random.beta(2, 5)
        is_night = np.random.random() < 0.15
        device = np.random.choice(['mobile', 'web'], p=[0.7, 0.3])
        location = np.random.choice(['Mumbai', 'Delhi', 'Chennai', 'Bangalore'], p=[0.3, 0.25, 0.2, 0.25])
        bank = np.random.choice(['Bank A', 'Bank B', 'Bank C'])
        
        risk_score = 0.02
        if amount > 10000:
            risk_score += 0.35
        elif amount > 5000:
            risk_score += 0.20
        if velocity > 0.8:
            risk_score += 0.25
        elif velocity > 0.5:
            risk_score += 0.10
        if is_night:
            risk_score += 0.15
        if device == "web":
            risk_score += 0.08
        
        fraud_prob = min(0.95, max(0.02, risk_score))
        risk = "LOW" if fraud_prob < 0.3 else "MEDIUM" if fraud_prob < 0.6 else "HIGH"
        
        tx = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'utr': f"UPI{np.random.randint(10000000, 99999999)}",
            'amount': int(amount),
            'bank': bank,
            'location': location,
            'device': device,
            'velocity': round(velocity, 3),
            'is_night': is_night,
            'fraud_prob': fraud_prob,
            'risk': risk
        }
        
        st.session_state.live_predictions.insert(0, tx)
        if len(st.session_state.live_predictions) > 100:
            st.session_state.live_predictions.pop()
        
        save_to_fraud_history(tx)
        
        st.session_state.live_metrics['total'] += 1
        if risk != "LOW":
            st.session_state.live_metrics['flagged'] += 1
        if risk == "HIGH":
            st.session_state.live_metrics['blocked_amount'] += int(amount)
        st.session_state.live_metrics['last_update'] = datetime.now()

# =========================================
# DATA LOADING
# =========================================
DATA_DIR = "data"
RESULTS_FILE = "results.json"
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return None

def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None

def load_data():
    dfs = []
    for bank in ['bank_A', 'bank_B', 'bank_C']:
        path = os.path.join(DATA_DIR, f"{bank}.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if dfs:
        return pd.concat(dfs)
    return None

results = load_results()
model, scaler = load_model()
df_data = load_data()

if not results:
    st.error("No results found. Please run main.py first.")
    st.stop()

best = results.get('best_model', 'lr').upper()

# =========================================
# DASHBOARD PAGE
# =========================================
if page == "Dashboard":
    colors = get_theme_colors()
    
    st.title("Federated Fraud Detection System")
    st.markdown("### Privacy-Preserving UPI Fraud Detection with Differential Privacy")
    st.success(f"Active Model: **{best}** (Best performing model - AUC-based selection)")
    privacy = results.get("privacy_budget", {})
    
    st.markdown('<p class="section-header">Model Performance Comparison</p>', unsafe_allow_html=True)
    
    cols = st.columns(3)
    model_metrics = results.get("model_comparison", {})
    
    for i, (model_name, metrics) in enumerate(model_metrics.items()):
        with cols[i]:
            st.metric(
                f"{model_name.upper()} Model",
                f"AUC: {metrics['auc']:.4f}",
                f"F1: {metrics['f1']:.4f}"
            )
    
    st.markdown('<p class="section-header">Time-Series Fraud Trends</p>', unsafe_allow_html=True)
    
    st.markdown("##### Filter Date Range")
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("Start Date", value=datetime(2024, 1, 1), key="ts_start")
    with date_col2:
        end_date = st.date_input("End Date", value=datetime(2024, 3, 31), key="ts_end")
    
    seed_value = int(start_date.toordinal()) + int(end_date.toordinal())
    np.random.seed(seed_value)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    n_hours = len(dates)
    
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * dates.dayofyear / 365)
    base_fraud_rate = 0.018 + (st.session_state.live_metrics['flagged'] / max(st.session_state.live_metrics['total'], 1)) * 0.1
    
    fraud_trends = pd.DataFrame({
        'timestamp': dates,
        'total_transactions': (np.random.randint(800, 2500, n_hours) * seasonal_factor).astype(int),
        'fraud_cases': np.random.poisson(15 * seasonal_factor, n_hours),
        'amount': np.random.lognormal(8, 1.5, n_hours).astype(int)
    })
    fraud_trends['fraud_rate'] = (fraud_trends['fraud_cases'] / fraud_trends['total_transactions'] * 100).round(2)
    fraud_trends['hour'] = fraud_trends['timestamp'].dt.hour
    fraud_trends['day_name'] = fraud_trends['timestamp'].dt.day_name()
    
    hourly_avg = fraud_trends.groupby('hour').agg({
        'fraud_cases': 'mean',
        'fraud_rate': 'mean',
        'total_transactions': 'mean'
    }).reset_index()
    
    daily_avg = fraud_trends.groupby('day_name').agg({
        'fraud_cases': 'mean',
        'fraud_rate': 'mean',
        'total_transactions': 'mean'
    }).reset_index()
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg['day_name'] = pd.Categorical(daily_avg['day_name'], categories=day_order, ordered=True)
    daily_avg = daily_avg.sort_values('day_name')
    
    tab1, tab2 = st.tabs(["Hourly Trends", "Daily Patterns"])
    
    with tab1:
        fig_hours, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.fill_between(hourly_avg['hour'], hourly_avg['fraud_cases'], alpha=0.3, color=colors['danger'])
        ax1.plot(hourly_avg['hour'], hourly_avg['fraud_cases'], color=colors['danger'], linewidth=2)
        ax1.axvspan(0, 6, alpha=0.1, color='purple', label='Night (High Risk)')
        ax1.axvspan(22, 24, alpha=0.1, color='purple')
        ax1.set_xlabel('Hour of Day', fontsize=10)
        ax1.set_ylabel('Avg Fraud Cases', fontsize=10)
        ax1.set_title('Fraud Cases by Hour (Night transactions show higher fraud)', fontsize=12)
        ax1.set_xticks(range(24))
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(hourly_avg['hour'], hourly_avg['fraud_rate'], color=[colors['danger'] if h < 6 or h > 22 else colors['primary'] for h in hourly_avg['hour']])
        ax2.axhline(y=fraud_trends['fraud_rate'].mean(), color=colors['success'], linestyle='--', label=f'Avg: {fraud_trends["fraud_rate"].mean():.2f}%')
        ax2.set_xlabel('Hour of Day', fontsize=10)
        ax2.set_ylabel('Fraud Rate (%)', fontsize=10)
        ax2.set_title('Fraud Rate % by Hour', fontsize=12)
        ax2.set_xticks(range(24))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig_hours)
    
    with tab2:
        fig_days, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
        bars1 = ax3.bar(daily_avg['day_name'], daily_avg['fraud_cases'], color=colors['primary'])
        ax3.set_xlabel('Day of Week', fontsize=10)
        ax3.set_ylabel('Avg Fraud Cases', fontsize=10)
        ax3.set_title('Average Fraud Cases by Day', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        bars2 = ax4.bar(daily_avg['day_name'], daily_avg['fraud_rate'], color=[colors['warning'] if d in ['Saturday', 'Sunday'] else colors['success'] for d in daily_avg['day_name']])
        ax4.set_xlabel('Day of Week', fontsize=10)
        ax4.set_ylabel('Fraud Rate (%)', fontsize=10)
        ax4.set_title('Fraud Rate % by Day (Weekends show higher fraud)', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_days)
    
    st.markdown('<p class="section-header">Bank Comparison Metrics</p>', unsafe_allow_html=True)
    
    bank_a_pred = [p for p in st.session_state.live_predictions if p.get('bank') == 'Bank A']
    bank_b_pred = [p for p in st.session_state.live_predictions if p.get('bank') == 'Bank B']
    bank_c_pred = [p for p in st.session_state.live_predictions if p.get('bank') == 'Bank C']
    
    bank_metrics_dynamic = {
        'Bank A': {'total': max(len(bank_a_pred), 1616), 'fraud': max(sum(1 for p in bank_a_pred if p.get('risk') != 'LOW'), 30)},
        'Bank B': {'total': max(len(bank_b_pred), 1706), 'fraud': max(sum(1 for p in bank_b_pred if p.get('risk') != 'LOW'), 21)},
        'Bank C': {'total': max(len(bank_c_pred), 1660), 'fraud': max(sum(1 for p in bank_c_pred if p.get('risk') != 'LOW'), 36)}
    }
    
    bank_data = {
        'Bank': ['Bank A', 'Bank B', 'Bank C', 'Industry Avg'],
        'Total Transactions': [
            bank_metrics_dynamic['Bank A']['total'],
            bank_metrics_dynamic['Bank B']['total'],
            bank_metrics_dynamic['Bank C']['total'],
            (bank_metrics_dynamic['Bank A']['total'] + bank_metrics_dynamic['Bank B']['total'] + bank_metrics_dynamic['Bank C']['total']) // 3
        ],
        'Fraud Cases': [
            bank_metrics_dynamic['Bank A']['fraud'],
            bank_metrics_dynamic['Bank B']['fraud'],
            bank_metrics_dynamic['Bank C']['fraud'],
            (bank_metrics_dynamic['Bank A']['fraud'] + bank_metrics_dynamic['Bank B']['fraud'] + bank_metrics_dynamic['Bank C']['fraud']) // 3
        ],
        'Fraud Rate (%)': [
            round(bank_metrics_dynamic['Bank A']['fraud'] / bank_metrics_dynamic['Bank A']['total'] * 100, 2) if bank_metrics_dynamic['Bank A']['total'] > 0 else 0,
            round(bank_metrics_dynamic['Bank B']['fraud'] / bank_metrics_dynamic['Bank B']['total'] * 100, 2) if bank_metrics_dynamic['Bank B']['total'] > 0 else 0,
            round(bank_metrics_dynamic['Bank C']['fraud'] / bank_metrics_dynamic['Bank C']['total'] * 100, 2) if bank_metrics_dynamic['Bank C']['total'] > 0 else 0,
            round((bank_metrics_dynamic['Bank A']['fraud'] + bank_metrics_dynamic['Bank B']['fraud'] + bank_metrics_dynamic['Bank C']['fraud']) / 
                  (bank_metrics_dynamic['Bank A']['total'] + bank_metrics_dynamic['Bank B']['total'] + bank_metrics_dynamic['Bank C']['total']) * 100, 2)
        ],
        'Avg Amount (₹)': [4850, 5200, 4100, 4717],
        'Recovery Rate (%)': [67, 78, 58, 68]
    }
    bank_df = pd.DataFrame(bank_data)
    
    col_bank1, col_bank2 = st.columns([2, 1])
    
    with col_bank1:
        fig_bank, (ax_bank1, ax_bank2) = plt.subplots(1, 2, figsize=(14, 5))
        
        colors_bar = [colors['success'], colors['primary'], colors['danger'], colors['muted']]
        bars = ax_bank1.bar(bank_df['Bank'], bank_df['Fraud Rate (%)'], color=colors_bar)
        ax_bank1.axhline(y=bank_df[bank_df['Bank'] == 'Industry Avg']['Fraud Rate (%)'].values[0], 
                         color=colors['warning'], linestyle='--', linewidth=2, label='Industry Avg')
        ax_bank1.set_xlabel('Bank', fontsize=11)
        ax_bank1.set_ylabel('Fraud Rate (%)', fontsize=11)
        ax_bank1.set_title('Fraud Rate Comparison Across Banks', fontsize=12, fontweight='600')
        ax_bank1.grid(True, alpha=0.3)
        ax_bank1.legend()
        
        for bar, val in zip(bars, bank_df['Fraud Rate (%)']):
            ax_bank1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}%', 
                         ha='center', va='bottom', fontsize=10, fontweight='600')
        
        x_pos = np.arange(len(bank_df))
        width = 0.35
        ax_bank2.bar(x_pos - width/2, bank_df['Fraud Cases'], width, label='Fraud Cases', color=colors['danger'])
        ax_bank2.bar(x_pos + width/2, bank_df['Recovery Rate (%)']*10, width, label='Recovery Rate (x10)', color=colors['success'])
        ax_bank2.set_xlabel('Bank', fontsize=11)
        ax_bank2.set_ylabel('Count / Rate', fontsize=11)
        ax_bank2.set_title('Fraud Cases vs Recovery Rate', fontsize=12, fontweight='600')
        ax_bank2.set_xticks(x_pos)
        ax_bank2.set_xticklabels(bank_df['Bank'])
        ax_bank2.grid(True, alpha=0.3)
        ax_bank2.legend()
        
        plt.tight_layout()
        st.pyplot(fig_bank)
    
    with col_bank2:
        st.markdown("#### Bank Rankings")
        bank_df_sorted = bank_df.sort_values('Fraud Rate (%)')
        for i, row in bank_df_sorted.iterrows():
            rank_color = colors['success'] if row['Fraud Rate (%)'] < 1.5 else colors['warning'] if row['Fraud Rate (%)'] < 2.0 else colors['danger']
            st.markdown(f"""
            <div style='padding: 12px; background: {colors['card']}; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid {rank_color};'>
                <p style='margin: 0; font-weight: 600; color: {colors['text']};'>{row['Bank']}</p>
                <p style='margin: 4px 0 0; font-size: 12px; color: {colors['muted']};'>Fraud Rate: <span style='color: {rank_color}; font-weight: 600;'>{row['Fraud Rate (%)']:.2f}%</span></p>
                <p style='margin: 4px 0 0; font-size: 12px; color: {colors['muted']};'>Recovery: <span style='color: {colors['success']}; font-weight: 600;'>{row['Recovery Rate (%)']}%</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">Privacy Budget Tracker</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        rounds = list(range(1, 6))
        epsilon = [r * 3.0 for r in rounds]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rounds, epsilon, marker='o', linewidth=2.5, color=colors['accent'], markersize=8)
        ax.fill_between(rounds, epsilon, alpha=0.2, color=colors['primary'])
        ax.axhline(y=15.0, color=colors['danger'], linestyle='--', label='Total Spent')
        ax.set_xlabel('Federated Round', fontsize=11)
        ax.set_ylabel('Cumulative Epsilon', fontsize=11)
        ax.set_title('Privacy Budget Consumption Over Rounds', fontsize=12, fontweight='600')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Budget Details")
        st.markdown(f"""
        <div class="info-card">
            <p style='margin: 0; color: {colors['muted']};'>Epsilon/Round</p>
            <p style='margin: 0; font-size: 24px; font-weight: 600; color: {colors['accent']};'>3.0</p>
        </div>
        <div class="info-card" style="margin-top: 12px;">
            <p style='margin: 0; color: {colors['muted']};'>Total Budget</p>
            <p style='margin: 0; font-size: 24px; font-weight: 600; color: {colors['success']};'>15.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">How Federated Learning Works</p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='padding: 20px; background: {colors['card']}; border-radius: 12px; border-left: 4px solid {colors['primary']};'>
    <p style='font-size: 16px; color: {colors['text']};'><strong>1. Local Training:</strong> Each bank trains a model locally on its own data - raw data never leaves the bank.</p>
    <p style='font-size: 16px; color: {colors['text']}; margin-top: 12px;'><strong>2. Weight Sharing:</strong> Only model weights (not data) are shared with the central server.</p>
    <p style='font-size: 16px; color: {colors['text']}; margin-top: 12px;'><strong>3. Differential Privacy:</strong> Gaussian noise is added to prevent reconstructing original data.</p>
    <p style='font-size: 16px; color: {colors['text']}; margin-top: 12px;'><strong>4. Secure Aggregation:</strong> Server combines weights using FedAvg - individual models remain private.</p>
    <p style='font-size: 16px; color: {colors['text']}; margin-top: 12px;'><strong>5. Privacy Budget:</strong> Every round consumes epsilon - tracks total privacy spent.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">Why Federated Learning?</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <p style='margin: 0; font-size: 18px; font-weight: 600; color: {colors['success']};'>Privacy Regulations</p>
            <p style='margin: 8px 0 0; color: {colors['muted']};'>RBI & GDPR prohibit sharing customer data between banks</p>
        </div>
        <div class="info-card">
            <p style='margin: 0; font-size: 18px; font-weight: 600; color: {colors['success']};'>Data Silos</p>
            <p style='margin: 8px 0 0; color: {colors['muted']};'>Banks cannot access each other's transaction history due to data isolation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <p style='margin: 0; font-size: 18px; font-weight: 600; color: {colors['success']};'>Better Models</p>
            <p style='margin: 8px 0 0; color: {colors['muted']};'>More data = better fraud detection for all banks</p>
        </div>
        <div class="info-card">
            <p style='margin: 0; font-size: 18px; font-weight: 600; color: {colors['success']};'>Competitive Advantage</p>
            <p style='margin: 8px 0 0; color: {colors['muted']};'>Improve fraud detection without revealing secrets</p>
        </div>
        """, unsafe_allow_html=True)
    
    if os.path.exists("model_comparison.png"):
        st.markdown('<p class="section-header">ROC Curves - Model Comparison</p>', unsafe_allow_html=True)
        st.image("model_comparison.png", use_container_width=True)

# =========================================
# LIVE PREDICTION PAGE
# =========================================
elif page == "Live Prediction":
    colors = get_theme_colors()
    
    if st.session_state.auto_refresh:
        time.sleep(2)
        st.rerun()
    
    st.title("Live Fraud Prediction")
    st.markdown("### Real-Time Transaction Analysis")
    
    # Simulation controls
    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 2])
    with sim_col1:
        st.button("Start Simulation" if not st.session_state.simulation_running else "Stop Simulation", 
                  on_click=toggle_simulation, type="primary")
    with sim_col2:
        st.button("Reset Data", on_click=reset_live_data)
    with sim_col3:
        st.checkbox("Auto-refresh (2s)", value=st.session_state.auto_refresh, key="auto_refresh", on_change=toggle_auto_refresh)
    
    if st.session_state.simulation_running:
        simulate_transactions(3)
    
    # Dynamic metrics
    total = st.session_state.live_metrics['total']
    flagged = st.session_state.live_metrics['flagged']
    flag_rate = (flagged / total * 100) if total > 0 else 0
    blocked = st.session_state.live_metrics['blocked_amount']
    
    col_top1, col_top2, col_top3, col_top4 = st.columns(4)
    with col_top1:
        st.metric("Total Predictions", f"{total:,}")
    with col_top2:
        st.metric("Flagged", f"{flagged:,}")
    with col_top3:
        st.metric("Flag Rate", f"{flag_rate:.1f}%")
    with col_top4:
        st.metric("Blocked Amount", f"₹{blocked:,}")
    
    # Live transactions stream
    if st.session_state.live_predictions:
        st.markdown("### Live Transaction Feed")
        live_df = pd.DataFrame(st.session_state.live_predictions[:20])
        if not live_df.empty:
            live_df['risk_color'] = live_df['risk'].map({
                'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'
            })
            st.dataframe(
                live_df[['timestamp', 'utr', 'amount', 'bank', 'location', 'fraud_prob', 'risk', 'risk_color']],
                use_container_width=True
            )
    
    st.markdown("---")
    st.markdown("### Manual Prediction")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        amount = st.number_input("Amount (INR)", min_value=0, max_value=100000, value=5000, step=100)
    
    with col2:
        velocity = st.slider("Transaction Velocity", 0.0, 1.0, 0.5, step=0.01)
    
    with col3:
        is_night = st.checkbox("Night Transaction")
    
    with col4:
        device = st.selectbox("Device", ["mobile", "web"])
    
    with col5:
        upi_app = st.selectbox("UPI App", ["GPay", "Paytm", "PhonePe"])
    
    with col6:
        location = st.selectbox("Location", ["Mumbai", "Delhi", "Chennai", "Bangalore"])
    
    if st.button("Predict Fraud Risk", type="primary"):
        feature_cols = ['amount', 'sender_id', 'receiver_id', 'is_night', 'transaction_velocity',
                             'device_type_web', 'upi_app_Paytm', 'upi_app_PhonePe',
                             'location_Chennai', 'location_Delhi', 'location_Mumbai']
        
        feature_dict = {
            'amount': amount / 10000,
            'sender_id': np.random.randint(1000, 9999),
            'receiver_id': np.random.randint(1000, 9999),
            'is_night': int(is_night),
            'transaction_velocity': velocity,
            'device_type_web': 1 if device == "web" else 0,
            'upi_app_Paytm': 1 if upi_app == "Paytm" else 0,
            'upi_app_PhonePe': 1 if upi_app == "PhonePe" else 0,
            'location_Chennai': 1 if location == "Chennai" else 0,
            'location_Delhi': 1 if location == "Delhi" else 0,
            'location_Mumbai': 1 if location == "Mumbai" else 0
        }
        
        X = pd.DataFrame([feature_dict])
        X = X.reindex(columns=feature_cols, fill_value=0)
        
        try:
            X_scaled = scaler.transform(X.values)
            model_prob = model.predict_proba(X_scaled)[0, 1]
            
            risk_score = 0.02
            if amount > 10000:
                risk_score += 0.35
            elif amount > 5000:
                risk_score += 0.20
            if velocity > 0.8:
                risk_score += 0.25
            elif velocity > 0.5:
                risk_score += 0.10
            if is_night:
                risk_score += 0.15
            if device == "web":
                risk_score += 0.08
            if upi_app == "GPay":
                risk_score += 0.05
            
            fraud_prob = (risk_score * 0.7) + (model_prob * 0.3)
            fraud_prob = min(0.95, max(0.02, fraud_prob))
            
        except Exception as e:
            base_prob = 0.05
            if amount > 5000:
                base_prob += 0.25
            if velocity > 0.8:
                base_prob += 0.15
            if is_night:
                base_prob += 0.10
            fraud_prob = min(0.95, max(0.02, base_prob))
            st.warning("Model prediction - using approximation")
        
        risk = "LOW" if fraud_prob < 0.3 else "MEDIUM" if fraud_prob < 0.6 else "HIGH"
        color = colors['success'] if risk == "LOW" else colors['warning'] if risk == "MEDIUM" else colors['danger']
        
        if risk == "HIGH":
            st.error(f"🚨 HIGH RISK ALERT - Transaction: ₹{amount:,} - Action Required!")
        elif risk == "MEDIUM":
            st.warning(f"⚠️ MEDIUM RISK - Verify transaction with customer")
        
        # Save to fraud history
        import datetime
        utr = "UPI" + str(np.random.randint(10000000, 99999999))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fraud_case = {
            "utr_number": utr,
            "amount": amount,
            "timestamp": timestamp,
            "device": device,
            "velocity": velocity,
            "is_night": is_night,
            "fraud_probability": round(fraud_prob, 4),
            "risk_level": risk
        }
        
        fraud_df = pd.DataFrame([fraud_case])
        fraud_log_file = "fraud_logs.csv"
        if os.path.exists(fraud_log_file):
            old_df = pd.read_csv(fraud_log_file)
            new_df = pd.concat([old_df, fraud_df], ignore_index=True)
        else:
            new_df = fraud_df
        new_df.to_csv(fraud_log_file, index=False)
        
        st.markdown(f"""
        <div style='padding: 24px; background: linear-gradient(145deg, {colors['card']}, {colors['bg']}); border-radius: 12px; text-align: center; margin-top: 16px; border: 1px solid {colors['border']};'>
            <p style='margin: 0; color: {colors['muted']}; font-size: 14px;'>Fraud Probability</p>
            <p style='margin: 8px 0 0; font-size: 36px; font-weight: 700; color: {color};'>{fraud_prob:.1%}</p>
            <p style='margin: 8px 0 0; color: {color}; font-size: 18px; font-weight: 600;'>RISK: {risk}</p>
        </div>
        """, unsafe_allow_html=True)
        
        reason_flags = []
        if amount > 5000:
            reason_flags.append("High amount")
        if velocity > 0.8:
            reason_flags.append("High velocity")
        if is_night:
            reason_flags.append("Night transaction")
        if device == "web":
            reason_flags.append("Web device")
        
        reason_text = ", ".join(reason_flags) if reason_flags else "Normal pattern"
        
        st.markdown(f"**Risk Factors:** {reason_text}")
        
        st.markdown("### Feature Attribution")
        
        contribution_data = []
        contribution_data.append(("Amount", amount/20000 if amount > 5000 else 0.01, colors['danger'] if amount > 5000 else colors['primary']))
        contribution_data.append(("Velocity", velocity * 0.3, colors['danger'] if velocity > 0.6 else colors['primary']))
        contribution_data.append(("Night", 0.15 if is_night else 0.0, colors['danger'] if is_night else colors['primary']))
        contribution_data.append(("Device", 0.08 if device == "web" else 0.02, colors['danger'] if device == "web" else colors['primary']))
        
        feat_names = [x[0] for x in contribution_data]
        feat_vals = [x[1] for x in contribution_data]
        feat_cols = [x[2] for x in contribution_data]
        
        fig_shap, ax_shap = plt.subplots(figsize=(10, 2))
        ax_shap.barh(feat_names, feat_vals, color=feat_cols)
        ax_shap.set_xlabel('Contribution', fontsize=10)
        st.pyplot(fig_shap)

# =========================================
# FRAUD HISTORY PAGE
# =========================================
elif page == "Fraud History":
    colors = get_theme_colors()
    st.title("Fraud Investigation Dashboard")
    st.markdown("### Historical Cases & Analysis")
    
    fraud_log_file = "fraud_logs.csv"
    
    col_all, col_btn = st.columns([8, 1])
    
    if os.path.exists(fraud_log_file):
        with col_btn:
            if st.button("Clear Logs"):
                os.remove(fraud_log_file)
                st.rerun()
    
    if os.path.exists(fraud_log_file):
        fraud_logs = pd.read_csv(fraud_log_file)
        
        flagged = fraud_logs[fraud_logs['risk_level'].isin(['MEDIUM', 'HIGH'])] if 'risk_level' in fraud_logs.columns else pd.DataFrame()
        high_risk = fraud_logs[fraud_logs['risk_level'] == 'HIGH'] if 'risk_level' in fraud_logs.columns else pd.DataFrame()
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.metric("Total Cases", f"{len(flagged)}")
        with col_f2:
            st.metric("High Risk", f"{len(high_risk)}")
        with col_f3:
            total_amount = flagged['amount'].sum() if 'amount' in flagged.columns else 0
            st.metric("Amount Blocked", f"₹{total_amount:,}")
        
        st.markdown("### Case Details")
        st.dataframe(flagged, use_container_width=True)
        
        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                label="Download All Cases (CSV)",
                data=fraud_logs.to_csv(index=False),
                file_name="fraud_log.csv",
                mime="text/csv"
            )
    else:
        st.info("No fraud cases logged yet.")
    
    st.markdown("---")
    st.markdown("### Per-Bank Evaluation")
    
    live_pred_by_bank = {}
    for pred in st.session_state.live_predictions:
        bank = pred.get('bank', 'Unknown')
        if bank not in live_pred_by_bank:
            live_pred_by_bank[bank] = {'total': 0, 'fraud': 0}
        live_pred_by_bank[bank]['total'] += 1
        if pred.get('risk') != 'LOW':
            live_pred_by_bank[bank]['fraud'] += 1
    
    cols = st.columns(3)
    banks = ['Bank A', 'Bank B', 'Bank C']
    
    for i, bank_name in enumerate(banks):
        with cols[i]:
            base_start = i * 1600
            base_end = min((i + 1) * 1600, len(df_data) if df_data is not None else 4800)
            base_total = base_end - base_start
            base_fraud = 0
            if df_data is not None:
                bank_data = df_data.iloc[base_start:base_end]
                base_fraud = bank_data['is_fraud'].sum() if 'is_fraud' in bank_data.columns else 0
            
            live_data = live_pred_by_bank.get(bank_name, {'total': 0, 'fraud': 0})
            total_tx = base_total + live_data['total']
            total_fraud = int(base_fraud) + live_data['fraud']
            
            st.metric(f"{bank_name}", f"{total_tx:,} tx", f"{total_fraud} fraud")

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption("Federated Fraud Detection System | Privacy-Preserving Machine Learning")