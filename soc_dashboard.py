import streamlit as st
import pandas as pd
import requests
import time
import random
import plotly.graph_objects as go
import io
import time
import threading
import tempfile
import os
from live_capture import LiveCapture, get_interfaces
from monitor import MonitorWorker
from streamlit_autorefresh import st_autorefresh
# --- AYARLAR ---
API_URL = "http://localhost:8000/predict_pcap"
st.set_page_config(page_title="SOC | Ağ Güvenliği Paneli", layout="wide", page_icon="🛡️")

# --- CUSTOM CSS ---
st.markdown("""
<style>
.threat-tag { color: #ff4b4b; font-weight: bold; }
.safe-tag { color: #00c853; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- STATE YÖNETİMİ ---
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0
if 'total_safe' not in st.session_state:
    st.session_state.total_safe = 0
if 'total_threats' not in st.session_state:
    st.session_state.total_threats = 0
@st.cache_resource
def get_monitor_holder():
    return {
        "worker": None
    }

monitor_holder = get_monitor_holder()
def create_gauge(probability, is_attack):
    color = "#ff4b4b" if is_attack else "#00c853"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Tehdit Skoru (%)", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "rgba(0, 200, 83, 0.1)"},
                {'range': [50, 100], 'color': "rgba(255, 75, 75, 0.1)"}],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def predict_pcap(uploaded_file):
    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/octet-stream"
            )
        }

        response = requests.post(
            API_URL,
            files=files,
            timeout=300
        )

        if response.status_code != 200:
            st.error(response.text)
            return None

        return pd.read_csv(io.StringIO(response.text))

    except Exception as e:
        st.error(f"Connection error: {e}")
        return None
# --- ANA PANEL ---
st.title("🛡️ Güvenlik Operasyon Merkezi (SOC)")
st.markdown("Ağ akışlarını (Network Flows) gerçek zamanlı ve geriye dönük analiz edin.")

m1, m2, m3 = st.columns(3)
m1.metric("Toplam Analiz Edilen Akış", st.session_state.total_analyzed)
m2.metric("Güvenli Trafik (Normal)", st.session_state.total_safe)
m3.metric("Tespit Edilen Tehdit (Attack)", st.session_state.total_threats)

st.divider()

tab1, tab2 = st.tabs(["📁 PCAP/CSV Derin Analiz", "📡 Canlı Ağ Akışı (Live)"])

def display_flow_card(flow_data, result, index):
    is_success = result.get("status") == "success"
    if not is_success:
        st.error(f"Akış #{index} analiz edilemedi.")
        return

    pred = result["prediction"]
    prob = result["attack_probability"]
    is_attack = pred == "Attack"
    
    icon = "🔴" if is_attack else "🟢"
    status_text = "TEHDİT (ATTACK)" if is_attack else "GÜVENLİ (NORMAL)"
    
    with st.expander(f"{icon} Akış #{index} | Port: {flow_data.get('Destination Port', 'N/A')} | Durum: {status_text} | Olasılık: %{prob*100:.1f}", expanded=is_attack):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown("**Akış Özeti**")
            st.write(f"**Hedef Port:** {flow_data.get('Destination Port', '-')}")
            st.write(f"**Süre:** {flow_data.get('Flow Duration', '-')} ms")
            st.write(f"**Byte/s:** {flow_data.get('Flow Bytes/s', '-'):.2f}")
        with c2:
            st.markdown("**Paket İstatistikleri**")
            st.write(f"**İleri (Fwd):** {flow_data.get('Total Fwd Packets', '-')}")
            st.write(f"**Geri (Bwd):** {flow_data.get('Total Backward Packets', '-')}")
            st.write(f"**Max Uzunluk:** {flow_data.get('Max Packet Length', '-')}")
        with c3:
            st.plotly_chart(create_gauge(prob, is_attack), use_container_width=True)

with tab1:

    st.subheader("📁 Offline PCAP Analysis")

    uploaded_file = st.file_uploader(
        "Upload a PCAP/PCAPNG file",
        type=["pcap", "pcapng"]
    )

    if uploaded_file:

        st.success(f"Selected file: {uploaded_file.name}")

        if st.button("🚀 Analyze", type="primary"):

            with st.spinner("Analyzing network traffic..."):

                result_df = predict_pcap(uploaded_file)

            if result_df is not None:

                attacks = (result_df["Prediction"] == "Attack").sum()
                normal = (result_df["Prediction"] == "Normal").sum()
                total = len(result_df)

                st.session_state.total_analyzed += total
                st.session_state.total_safe += normal
                st.session_state.total_threats += attacks

                c1, c2, c3 = st.columns(3)

                c1.metric("Total Flows", total)
                c2.metric("Normal", normal)
                c3.metric("Attacks", attacks)

                st.divider()

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    height=500
                )

                csv = result_df.to_csv(index=False)

                st.download_button(
                    "⬇ Download Predictions",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )
