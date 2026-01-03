import streamlit as st
import requests
import time

# --- SETTINGS ---
REFRESH_EVERY_SECONDS = 30
st.set_page_config(page_title="OpenAI Rate Limits Dashboard", page_icon="🤖")

# --- SIDEBAR ---
st.sidebar.title("🔑 OpenAI Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# --- MAIN PAGE ---
st.title("📊 OpenAI Rate Limits Dashboard")

if api_key:
    headers = {"Authorization": f"Bearer {api_key}"}

    # --- Usage Info ---
    try:
        usage_response = requests.get(
            "https://api.openai.com/v1/dashboard/billing/usage?start_date=2025-04-01&end_date=2025-04-30",
            headers=headers
        )
        if usage_response.status_code == 200:
            usage = usage_response.json()
            total_usage = float(usage.get('total_usage', 0)) / 100  # in dollars
            st.metric(label="💵 Used So Far", value=f"${total_usage:.2f}")
        else:
            st.warning("⚠️ Usage info not available.")
    except Exception as e:
        st.error(f"Failed to fetch usage info: {e}")

    # --- Rate Limits Info (TPM/RPM) ---
    try:
        rate_response = requests.get(
            "https://api.openai.com/v1/rate_limits",
            headers=headers
        )
        if rate_response.status_code == 200:
            rate_limits = rate_response.json()

            st.subheader("🚀 Model Rate Limits (TPM and RPM)")
            model_data = {}

            for limit in rate_limits.get('limits', []):
                model = limit.get('model')
                limit_type = limit.get('limit_type')
                value = limit.get('limit_value')

                if model not in model_data:
                    model_data[model] = {}

                model_data[model][limit_type] = value

            for model, limits in model_data.items():
                tpm = limits.get('tokens', 'N/A')
                rpm = limits.get('requests', 'N/A')
                st.write(f"**{model}** ➔ 🧠 {tpm:,} TPM | 🔁 {rpm:,} RPM")
        else:
            st.warning("⚠️ Rate limits info not available.")
    except Exception as e:
        st.error(f"Failed to fetch rate limits info: {e}")

    # --- Footer Info ---
    st.caption(f"🔄 Auto-refreshing every {REFRESH_EVERY_SECONDS} seconds...")
    
    # --- Auto-refresh ---
    time.sleep(REFRESH_EVERY_SECONDS)
    st.rerun()

else:
    st.info("⚡ Please enter your OpenAI API key in the sidebar to begin.")
