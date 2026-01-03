import streamlit as st
import pandas as pd
import requests
import os

st.title("📡 O-RAN U-Plane PCAP to CSV Converter")

uploaded_file = st.file_uploader("Upload O-RAN U-Plane PCAP", type=["pcap", "pcapng"])
if uploaded_file:
    with open("temp.pcap", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Uploading and processing..."):
        res = requests.post("http://localhost:8010/upload", files={"pcap": open("temp.pcap", "rb")})
        if res.ok:
            data = res.json()
            csv_path = data["output_file"]
            st.success("✅ Converted successfully!")
            df = pd.read_csv(csv_path)

            st.dataframe(df.head())

            port_id = st.selectbox("Select Port", sorted(df['Port'].unique()))
            subframe = st.selectbox("Select Subframe", sorted(df['Subframe'].unique()))
            slot = st.selectbox("Select Slot", sorted(df['Slot'].unique()))
            symbol = st.selectbox("Select Symbol", sorted(df['Symbol'].unique()))

            filtered = df[
                (df["Port"] == port_id) &
                (df["Subframe"] == subframe) &
                (df["Slot"] == slot) &
                (df["Symbol"] == symbol)
            ]

            st.line_chart({"I": filtered["Real"], "Q": filtered["Imag"]})
        else:
            st.error("❌ Failed to process PCAP")
            
        # # Load the CSV file
        # df = pd.read_csv(csv_path)

        # # Drop the first row
        # df = df.iloc[1:, :]

        # # Drop the first 5 columns
        # df = df.iloc[:, 5:]

        # # Save to a new CSV
        # df.to_csv(csv_path)
