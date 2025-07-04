import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random

# Handle imports with error checking
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.error("PyTorch tidak tersedia. Menggunakan mode simulasi.")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        # Create mock spaces class
        class MockSpaces:
            class Box:
                def __init__(self, low, high, dtype=None):
                    self.low = low
                    self.high = high
                    self.dtype = dtype
            
            class Discrete:
                def __init__(self, n):
                    self.n = n
        
        spaces = MockSpaces()

# Page config
st.set_page_config(
    page_title="AI Workout Scheduler",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Environment Class
class WorkoutSchedulerEnvV2:
    def __init__(self):
        self.observation_space = spaces.Box(
            low=np.array([10, 0, 120, 30, 0, 0, 0, 4, 0, 10, 0, 0], dtype=np.float32),
            high=np.array([80, 1, 220, 150, 4, 180, 1, 12, 1, 40, 2, 1], dtype=np.float32)
        )
        self.action_space = spaces.Discrete(60)
        self.state = None
        self.activity_map = {
            0: "Jalan kaki",
            1: "Jogging", 
            2: "Yoga",
            3: "Senam ringan",
            4: "Sepeda statis"
        }

    def decode_action(self, action):
        waktu = action // 20
        jenis = (action % 20) // 4
        durasi = (action % 20) % 4
        return waktu, jenis, durasi

# PPO Models - Only if torch is available
if TORCH_AVAILABLE:
    class PPOActor(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(PPOActor, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            return self.net(x)

    class PPOCritic(nn.Module):
        def __init__(self, state_dim):
            super(PPOCritic, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            return self.net(x)

    class PPOAgent:
        def __init__(self, state_dim, action_dim, lr=3e-4):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.actor = PPOActor(state_dim, action_dim).to(self.device)
            self.critic = PPOCritic(state_dim).to(self.device)
            
            # Initialize with random weights for demo
            self._initialize_weights()
            
        def _initialize_weights(self):
            for m in self.actor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.1)
            for m in self.critic.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.1)

        def get_recommendation(self, state):
            self.actor.eval()
            self.critic.eval()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_probs = self.actor(state_tensor)
                value = self.critic(state_tensor)
            
            # Get top actions
            top_5 = torch.topk(action_probs[0], 5)
            
            return action_probs[0], value.item(), top_5

else:
    # Fallback Agent without PyTorch
    class PPOAgent:
        def __init__(self, state_dim, action_dim, lr=3e-4):
            self.state_dim = state_dim
            self.action_dim = action_dim
            
        def get_recommendation(self, state):
            # Simple rule-based recommendation as fallback
            np.random.seed(42)  # For consistent results
            
            # Generate pseudo-probabilities based on state
            action_probs = np.random.dirichlet(np.ones(self.action_dim))
            
            # Adjust probabilities based on user profile
            age, gender, height, weight, activity, duration, intensity, sleep, stress, bmi, health, mode = state
            
            # Simple rules to adjust probabilities
            if age > 50:
                # Prefer lighter activities for older users
                for i in range(0, 20):  # Light activities
                    action_probs[i] *= 1.5
            
            if bmi > 25:
                # Prefer longer duration for overweight users
                for i in range(0, 60, 4):  # Longer duration activities
                    action_probs[i+2:i+4] *= 1.3
            
            if stress > 0.7:
                # Prefer yoga for stressed users
                for i in range(8, 60, 20):  # Yoga activities
                    action_probs[i:i+4] *= 1.4
            
            # Normalize probabilities
            action_probs = action_probs / np.sum(action_probs)
            
            # Get top 5 actions
            top_5_indices = np.argsort(action_probs)[-5:][::-1]
            top_5_values = action_probs[top_5_indices]
            
            # Create mock torch-like objects
            class MockTensor:
                def __init__(self, values, indices):
                    self.values = values
                    self.indices = indices
                    
                def item(self):
                    return float(self.values[0]) if hasattr(self.values, '__len__') else float(self.values)
            
            top_5 = MockTensor(top_5_values, top_5_indices)
            value = np.random.uniform(0.5, 1.0)  # Mock value
            
            return action_probs, value, top_5

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = PPOAgent(12, 60)
if 'env' not in st.session_state:
    st.session_state.env = WorkoutSchedulerEnvV2()
if 'history' not in st.session_state:
    st.session_state.history = []

# Main App
st.markdown('<div class="main-header">🏃‍♂️ AI Workout Scheduler</div>', unsafe_allow_html=True)
st.markdown("### Dapatkan rekomendasi olahraga yang dipersonalisasi menggunakan AI")

# Show status
if not TORCH_AVAILABLE:
    st.warning("⚠️ PyTorch tidak tersedia. Menggunakan algoritma fallback untuk rekomendasi.")

# Sidebar for inputs
with st.sidebar:
    st.header("📊 Input Data Anda")
    
    # Personal Information
    st.subheader("Informasi Pribadi")
    age = st.slider("Usia", 18, 80, 30)
    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    height = st.slider("Tinggi (cm)", 140, 200, 170)
    weight = st.slider("Berat (kg)", 40, 120, 65)
    
    # Activity Information
    st.subheader("Informasi Aktivitas")
    activity_types = ["Santai", "Ringan", "Sedang", "Intensitas Tinggi", "Sangat Berat"]
    activity_type = st.selectbox("Jenis Aktivitas Sehari-hari", activity_types)
    
    duration = st.slider("Durasi Olahraga Terakhir (menit)", 0, 180, 30)
    intensity = st.slider("Intensitas Olahraga (0-1)", 0.0, 1.0, 0.5, 0.1)
    
    # Health Information
    st.subheader("Informasi Kesehatan")
    hours_sleep = st.slider("Jam Tidur Semalam", 4.0, 12.0, 7.0, 0.5)
    stress = st.slider("Tingkat Stres (0-1)", 0.0, 1.0, 0.3, 0.1)
    
    health_conditions = ["Baik", "Sedang", "Lemah"]
    health = st.selectbox("Kondisi Kesehatan", health_conditions)
    
    mode = st.selectbox("Mode", ["Normal", "Puasa"])
    
    # Calculate BMI
    bmi = weight / ((height/100) ** 2)
    
    # Convert inputs to model format
    gender_num = 1 if gender == "Laki-laki" else 0
    activity_num = activity_types.index(activity_type)
    health_num = health_conditions.index(health)
    mode_num = 1 if mode == "Puasa" else 0

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Informasi Kesehatan Anda")
    
    # Health metrics
    col_bmi, col_category = st.columns(2)
    
    with col_bmi:
        if bmi < 18.5:
            bmi_category = "Underweight"
            bmi_color = "blue"
        elif bmi < 25:
            bmi_category = "Normal"
            bmi_color = "green"
        elif bmi < 30:
            bmi_category = "Overweight"
            bmi_color = "orange"
        else:
            bmi_category = "Obese"
            bmi_color = "red"
            
        st.metric("BMI", f"{bmi:.1f}", f"{bmi_category}")
    
    with col_category:
        st.metric("Kategori Kesehatan", health_conditions[health_num])
    
    # Create state for prediction
    state = np.array([
        age, gender_num, height, weight, activity_num, duration,
        intensity, hours_sleep, stress, bmi, health_num, mode_num
    ], dtype=np.float32)
    
    # Get recommendation
    if st.button("🎯 Dapatkan Rekomendasi", type="primary"):
        with st.spinner("Menganalisis data Anda..."):
            try:
                action_probs, value, top_5 = st.session_state.agent.get_recommendation(state)
                
                # Decode recommendations
                recommendations = []
                for i in range(5):
                    if TORCH_AVAILABLE:
                        prob = top_5.values[i].item()
                        action_idx = top_5.indices[i].item()
                    else:
                        prob = top_5.values[i]
                        action_idx = top_5.indices[i]
                    
                    waktu, jenis, durasi_kode = st.session_state.env.decode_action(action_idx)
                    
                    waktu_map = {0: "Pagi", 1: "Sore", 2: "Tidak olahraga"}
                    jenis_map = {0: "Jalan kaki", 1: "Jogging", 2: "Yoga", 3: "Senam ringan", 4: "Sepeda statis"}
                    durasi_map = {0: "15 menit", 1: "30 menit", 2: "45 menit", 3: "60 menit"}
                    
                    recommendations.append({
                        'rank': i + 1,
                        'waktu': waktu_map[waktu],
                        'jenis': jenis_map[jenis],
                        'durasi': durasi_map[durasi_kode],
                        'confidence': prob
                    })
                
                # Display main recommendation
                main_rec = recommendations[0]
                confidence_class = "confidence-high" if main_rec['confidence'] > 0.7 else "confidence-medium" if main_rec['confidence'] > 0.4 else "confidence-low"
                
                st.markdown(f"""
                <div class="recommendation-box">
                    <h3>🏆 Rekomendasi Utama</h3>
                    <p><strong>🕒 Waktu:</strong> {main_rec['waktu']}</p>
                    <p><strong>🏃 Jenis Olahraga:</strong> {main_rec['jenis']}</p>
                    <p><strong>⏱️ Durasi:</strong> {main_rec['durasi']}</p>
                    <p><strong>📊 Confidence:</strong> <span class="{confidence_class}">{main_rec['confidence']:.1%}</span></p>
                    <p><strong>💰 Estimated Value:</strong> {value:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display alternative recommendations
                st.subheader("🔄 Rekomendasi Alternatif")
                
                rec_df = pd.DataFrame(recommendations[1:])
                rec_df['confidence'] = rec_df['confidence'].apply(lambda x: f"{x:.1%}")
                rec_df.columns = ['Rank', 'Waktu', 'Jenis Olahraga', 'Durasi', 'Confidence']
                
                st.dataframe(rec_df, use_container_width=True, hide_index=True)
                
                # Save to history
                st.session_state.history.append({
                    'timestamp': pd.Timestamp.now(),
                    'age': age,
                    'bmi': bmi,
                    'recommendation': f"{main_rec['waktu']} - {main_rec['jenis']} - {main_rec['durasi']}",
                    'confidence': main_rec['confidence'],
                    'value': value
                })
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
                st.info("Silakan coba lagi atau hubungi administrator.")

with col2:
    st.subheader("📊 Visualisasi")
    
    # BMI gauge chart
    try:
        fig_bmi = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = bmi,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "BMI"},
            gauge = {
                'axis': {'range': [None, 40]},
                'bar': {'color': bmi_color},
                'steps': [
                    {'range': [0, 18.5], 'color': "lightblue"},
                    {'range': [18.5, 25], 'color': "lightgreen"},
                    {'range': [25, 30], 'color': "yellow"},
                    {'range': [30, 40], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        fig_bmi.update_layout(height=300)
        st.plotly_chart(fig_bmi, use_container_width=True)
    except Exception as e:
        st.error(f"Error menampilkan grafik BMI: {str(e)}")
    
    # Health metrics radar chart
    try:
        categories = ['Tidur', 'Aktivitas', 'Intensitas', 'Kesehatan']
        values = [
            hours_sleep/12*100,
            (activity_num+1)/5*100,
            intensity*100,
            (3-health_num)/3*100  # Invert so higher is better
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Health Profile'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    except Exception as e:
        st.error(f"Error menampilkan radar chart: {str(e)}")

# History section
if st.session_state.history:
    st.subheader("📈 Riwayat Rekomendasi")
    
    try:
        history_df = pd.DataFrame(st.session_state.history)
        history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Display recent history
        st.dataframe(history_df.tail(10), use_container_width=True, hide_index=True)
        
        # Plot confidence trend
        if len(history_df) > 1:
            fig_trend = px.line(
                history_df.tail(20), 
                x='timestamp', 
                y='confidence',
                title='Confidence Trend',
                labels={'confidence': 'Confidence (%)', 'timestamp': 'Time'}
            )
            fig_trend.update_layout(height=300)
            st.plotly_chart(fig_trend, use_container_width=True)
    except Exception as e:
        st.error(f"Error menampilkan riwayat: {str(e)}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666;">
    <p>🤖 Powered by {'PPO (Proximal Policy Optimization)' if TORCH_AVAILABLE else 'Rule-based'} AI</p>
    <p>💡 Rekomendasi berdasarkan kondisi kesehatan dan preferensi personal Anda</p>
    <p>📊 Status: {'PyTorch Active' if TORCH_AVAILABLE else 'Fallback Mode'}</p>
</div>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ Informasi")
    st.info("""
    **Cara menggunakan:**
    1. Isi semua data di sidebar
    2. Klik tombol 'Dapatkan Rekomendasi'
    3. Lihat hasil rekomendasi dan alternatif
    4. Pantau riwayat rekomendasi Anda
    """)
    
    st.subheader("🎯 Tentang Confidence")
    st.info("""
    - **Tinggi (>70%)**: Rekomendasi sangat cocok
    - **Sedang (40-70%)**: Rekomendasi cukup baik
    - **Rendah (<40%)**: Pertimbangkan alternatif
    """)
    
    if st.button("🗑️ Hapus Riwayat"):
        st.session_state.history = []
        st.success("Riwayat berhasil dihapus!")

    # Show system info
    st.markdown("---")
    st.subheader("🔧 System Info")
    st.info(f"""
    **PyTorch**: {'✅ Available' if TORCH_AVAILABLE else '❌ Not Available'}
    **Matplotlib**: {'✅ Available' if MATPLOTLIB_AVAILABLE else '❌ Not Available'}
    **Gym**: {'✅ Available' if GYM_AVAILABLE else '❌ Not Available'}
    """)
