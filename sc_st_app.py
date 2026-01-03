import streamlit as st
import re
from ncs540_serial import NCS540Serial

st.title("Cisco NCS540 Serial Console")

port = st.text_input("Serial Port", "COM8")
username = st.text_input("Username", "cisco")
password = st.text_input("Password", type="password")

if "conn" not in st.session_state:
    st.session_state.conn = None

# Connect
if st.button("Connect"):
    try:
        st.session_state.conn = NCS540Serial(port=port, username=username, password=password)
        st.success(f"✅ Connected and logged in via {port} as {username}")
    except Exception as e:
        st.error(f"❌ Failed to connect/login: {e}")

# Disconnect
if st.button("Disconnect"):
    if st.session_state.conn:
        st.session_state.conn.close()
        st.session_state.conn = None
        st.info("🔌 Disconnected")

# If connected, show features
if st.session_state.conn:
    st.subheader("Quick Device Info")
    if st.button("Fetch Device Info"):
        inv = st.session_state.conn.send_cmd("show inventory", wait=2)
        ver = st.session_state.conn.send_cmd("show version", wait=2)   # fixed
        plat = st.session_state.conn.send_cmd("show platform", wait=2)

        pid_match = re.search(r"PID:\s+(\S+)", inv)
        sn_match = re.search(r"SN:\s+(\S+)", inv)
        ver_match = re.search(r"Version\s+([\w\.\(\)]+)", ver)

        st.write(f"**Model:** {pid_match.group(1) if pid_match else 'Unknown'}")
        st.write(f"**Serial Number:** {sn_match.group(1) if sn_match else 'Unknown'}")
        st.write(f"**Software Version:** {ver_match.group(1) if ver_match else 'Unknown'}")

        with st.expander("show inventory"):
            st.text(inv)
        with st.expander("show version"):
            st.text(ver)
        with st.expander("show platform"):
            st.text(plat)

    st.subheader("Run Custom CLI Command")
    user_cmd = st.text_input("Enter CLI command", "show l2vpn bridge-domain")
    if st.button("Run Command") and user_cmd.strip():
        try:
            output = st.session_state.conn.send_cmd(user_cmd, wait=2)
            st.text_area("Command Output", output, height=300)
        except Exception as e:
            st.error(f"Error: {e}")
