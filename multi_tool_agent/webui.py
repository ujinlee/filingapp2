import streamlit as st
st.set_page_config(
    page_title="Filing Cast : Financial Filing Podcast Generator",
    page_icon="🎙️",
    layout="wide"
)
import requests
import json
from datetime import datetime
import re

def escape_markdown(text):
    # Escape Markdown special characters
    escape_chars = r'\\`*_{}[]()#+-.!'
    return re.sub(r'([%s])' % re.escape(escape_chars), r'\\\\1', text)

# Inject custom CSS for color scheme
st.markdown(
    """
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #232946 !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background-color: #232946 !important;
        color: #fff !important;
        border: 1px solid #b8c1ec !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background-color: #6246ea !important;
        color: #fff !important;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    /* Main area */
    .main {
        background-color: #f4f6fc !important;
    }
    .stApp {
        background-color: #f4f6fc !important;
    }
    /* Card/expander */
    .stExpander {
        background-color: #fff !important;
        border: 1px solid #b8c1ec !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(98,70,234,0.05);
    }
    .stExpanderHeader {
        color: #232946 !important;
        font-weight: bold;
    }
    /* Buttons */
    .stButton > button {
        background-color: #3e54a3 !important;
        color: #fff !important;
        border-radius: 8px;
        font-weight: bold;
    }
    /* Headings */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #232946 !important;
    }
    /* Links */
    a {
        color: #6246ea !important;
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Backend API URL
API_URL = "http://127.0.0.1:8000"

st.title("🎙️ Financial Filing Podcast Generator")

# Sidebar for ticker input
with st.sidebar:
    st.header("Search Filings")
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    
    if st.button("Search Filings"):
        with st.spinner("Fetching filings..."):
            try:
                response = requests.get(f"{API_URL}/api/filings", params={"ticker": ticker})
                if response.status_code == 200:
                    st.session_state.filings = response.json()
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Main content area
if 'filings' in st.session_state and st.session_state.filings:
    st.subheader(f"Latest Filings for {ticker}")
    
    # Display filings in a more organized way
    for filing in st.session_state.filings:
        with st.expander(f"{filing['form']} - {filing['date']}"):
            st.write(f"**Document URL:** {filing['url']}")
            
            # Language selection
            language = st.selectbox(
                "Select Language",
                ["en-US", "ko-KR", "ja-JP", "es-ES", "zh-CN", "fr-FR", "de-DE"],
                key=f"lang_{filing['date']}"
            )
            
            if st.button("Generate Podcast", key=f"gen_{filing['date']}"):
                with st.spinner("Generating podcast..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/api/summarize",
                            json={
                                "documentUrl": filing['url'],
                                "language": language
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display the audio player with the correct URL
                            st.audio(f"{API_URL}{result['audio_url']}", format="audio/mp3")
                            
                            # Display transcript
                            st.markdown("**Transcript:**")
                            st.text_area("Transcript", result['transcript'], height=300)
                            
                            # Display summary
                            st.markdown("**Summary:**")
                            st.write(result['summary'])
                        else:
                            st.error(f"Error generating podcast: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
else:
    st.info("Enter a ticker symbol and click 'Search Filings' to get started.") 