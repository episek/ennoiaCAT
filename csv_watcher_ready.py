#!/usr/bin/env python3
import os
import time
import logging
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pyshark
import re
import math
import csv
import matplotlib
matplotlib.use('Agg')  # No GUI (good for saving plots)
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from flask import Flask, render_template_string, render_template, jsonify, request
import openai
from openai import OpenAI
import os
import datetime
import time
from threading import Thread, Event
import asyncio
import markdown2
import pandas as pd
from io import StringIO
from scipy.linalg import toeplitz
from itertools import islice
from PIL import Image
from flask import session
import bitstring
from bitstring import BitStream
from tqdm import tqdm


# define these once at module scope
layer_interf_start = [0, 0, 0, 0]
layer_interf_end   = [0, 0, 0, 0]



def save_report_for_csv(csv_path: str, genreport: str) -> str:
    """
    Writes `genreport` to OUT_DIR/<csv_basename>.report.md atomically
    and returns the final path.
    """
    base = os.path.basename(csv_path)                 # e.g., "data.csv"
    report_final = os.path.join(OUT_DIR, base + ".report.md")
    report_tmp   = report_final + ".part"

    # Write to a temp file first (UTF-8, LF newlines), then atomic rename
    with open(report_tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(genreport)

    os.replace(report_tmp, report_final)  # atomic on POSIX
    logging.info(f"Report written: {report_final}")
    return report_final

def write_done_flag(report_path: str):
    done_final = report_path + ".done"
    done_tmp   = done_final + ".part"
    with open(done_tmp, "w", encoding="utf-8") as f:
        f.write("ok\n")
    os.replace(done_tmp, done_final)

def df_to_markdown_table(df: pd.DataFrame) -> str:
    # simple Markdown table; replace with your existing function if you have it
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)

def align_markdown_table(md: str) -> str:
    # placeholder passthrough; replace with your real aligner
    return md


INBOX = "/home/ec2-user/ennoiaCAT/csv_files"
PROCESSED = "/home/ec2-user/ennoiaCAT/csv_files/processed"
FAILED = "/home/ec2-user/ennoiaCAT/csv_files/failed"
LOGFILE = "/home/ec2-user/ennoiaCAT/csv_files/csv_watcher.log"
OUT_DIR = "/home/ec2-user/ennoiaCAT/csv_files/out"

STABLE_SECONDS = 2.0
SCAN_INTERVAL = 1.0
MAX_WORKERS = 2


os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(INBOX, exist_ok=True)
os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(FAILED, exist_ok=True)

logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

def is_stable(path: str, secs: float) -> bool:
    """Wait until file exists and size is unchanged for secs."""
    if not os.path.exists(path):
        return False
    last_size = -1
    last_change = time.time()
    while True:
        if not os.path.exists(path):
            return False
        size = os.path.getsize(path)
        if size != last_size:
            last_size = size
            last_change = time.time()
        elif time.time() - last_change >= secs:
            return True
        time.sleep(0.3)

def timestamped(root_dir: str, fname: str) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return os.path.join(root_dir, f"{stamp}__{os.path.basename(fname)}")

def process_csv(path: str):
    """
    ⚙️ Your real processing logic goes here.
    Replace this stub with whatever computation you need.
    """
    import csv
    with open(path, newline="") as f:
        rows = sum(1 for _ in csv.reader(f))
    logging.info(f"Processed {path} | rows={rows}")

def handle_ready(csv_path: str, ready_path: str):
    """Called once for each new .ready flag."""
    try:
        logging.info(f"Triggered by {ready_path}")
        # Wait until CSV is stable before reading it
        if not is_stable(csv_path, STABLE_SECONDS):
            raise RuntimeError("File not stable yet")

        process_csv(csv_path)

        # Archive CSV and remove .ready flag
        dst = timestamped(PROCESSED, csv_path)
        shutil.move(csv_path, dst)
        os.remove(ready_path)
        logging.info(f"Moved {csv_path} -> {dst}")
    except Exception as e:
        logging.exception(f"Processing failed for {csv_path}: {e}")
        dst = timestamped(FAILED, csv_path)
        shutil.move(csv_path, dst)
        shutil.move(ready_path, dst + ".ready")
        logging.info(f"Moved failed files to {dst}")

def generate_report(fname):
    global prompt
    interf = 1
    layer_interf_start[0] = 0
    layer_interf_start[1] = 0
    layer_interf_start[2] = 249
    layer_interf_start[3] = 249
    layer_interf_end[0] = 0
    layer_interf_end[1] = 0
    layer_interf_end[2] = 272
    layer_interf_end[3] = 272
    """Generates a report using OpenAI based on the provided structured data."""
    #formatted_data = data
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    #safe_fname = fname.replace("_", "\\_")
    safe_fname = f"`{fname}`"
    safe_now = f"`{now}`"
    safe_interf = f"`{interf}`"
    safe_interf_start_L0 = f"`{layer_interf_start[0]}`"
    safe_interf_end_L0 = f"`{layer_interf_end[0]}`"
    safe_interf_start_L1 = f"`{layer_interf_start[1]}`"
    safe_interf_end_L1 = f"`{layer_interf_end[1]}`"
    safe_interf_start_L2 = f"`{layer_interf_start[2]}`"
    safe_interf_end_L2 = f"`{layer_interf_end[2]}`"
    safe_interf_start_L3 = f"`{layer_interf_start[3]}`"
    safe_interf_end_L3 = f"`{layer_interf_end[3]}`"
    
    if (interf == 0):
        
        report_header_template = pd.DataFrame({
        
        "Item"  : ["Filename", "Date", "Status", "Issues"],
        "Value" : [{safe_fname}, {safe_now},  "✅", "No Issues"]
        })
        progress_status = {"status": "Analyzer Successfully Finished the Analysis - No Issues Found"}

    else:
        report_header_template = pd.DataFrame({
        
        "Item"  : ["Filename", "Date", "Status", "Issues"],
        "Value" : [{safe_fname}, {safe_now},  "❌", "Found Interference"]
        })
        progress_status = {"status": "Analyzer Successfully Finished the Analysis - Interference Found"}
        
    
    #print(report_header_template.to_markdown(index=False))
    
    markdown_header = df_to_markdown_table(report_header_template)
    # report_header_filled = report_header_template.format(safe_fname=safe_fname, safe_now=safe_now)
    
    # aligned_header = align_markdown_table(report_header_filled)
    
    if (interf == 0):
        link_dir = "Downlink"
        interf_l0 = "None"
        interf_l1 = "None"
        interf_l2 = "None"
        interf_l3 = "None"        
    else:
        link_dir = "Uplink"
        if (layer_interf_end[0] - layer_interf_start[0] == 0):
            interf_l0 = "None"
        else:
            interf_l0 = f"Detected in PRBs {layer_interf_start[0]+1}-{layer_interf_end[0]}"
        if (layer_interf_end[1] - layer_interf_start[1] == 0):
            interf_l1 = "None"
        else:
            interf_l1 = f"Detected in PRBs {layer_interf_start[1]+1}-{layer_interf_end[1]}"
        if (layer_interf_end[2] - layer_interf_start[2] == 0):
            interf_l2 = "None"
        else:
            interf_l2 = f"Detected in PRBs {layer_interf_start[2]+1}-{layer_interf_end[2]}"
        if (layer_interf_end[3] - layer_interf_start[3] == 0):
            interf_l3 = "None"
        else:
            interf_l3 = f"Detected in PRBs {layer_interf_start[3]+1}-{layer_interf_end[3]}"
        # interf_l1 = "None"
        # interf_l2 = "Detected in PRBs 250-272"
        # interf_l3 = "Detected in PRBs 250-272"

        
    scs_str = "30 KHz"
    
    data_summary_template = """
    | **Variable**                       | **Value**     | **Description**                                             |
    |                                    |               |                                                             |
    | **Sub-carrier spacing (KHz)**      | {scs_str}     | Defines spacing between sub-carriers.                       | 
    | **Number Of Antennas**             | 4             | Represents antennas configured in this setup.               | 
    | **Max Frames**                     | 1             | Indicates the max number of frames transmitted.             | 
    | **DL Direction**                   | {link_dir}    | Specifies the traffic direction.                            | 
    | **U-Plane Packet Type**            | U-plane       | Denotes user plane packets carrying user data.              | 
    | **Number Of PRBs**                 | 273           | Count of Physical Resource Blocks for transmission.         | 
    | **Bandwidth Frequency (MHz)**      | 98.28 MHz     | Total bandwidth allocated for transmission.                 | 
    | **Interference**                   | {safe_interf} | Indicates if interference was found in the packet.          |
    | **Interference - L0**              | {interf_l0}   | Indicates if interference was found in layer 0 the packet.  |
    | **Interference - L1**              | {interf_l1}   | Indicates if interference was found in layer 1 the packet.  |
    | **Interference - L2**              | {interf_l2}   | Indicates if interference was found in layer 2 the packet.  |
    | **Interference - L3**              | {interf_l3}   | Indicates if interference was found in layer 3 the packet.  |
    """
    #print(data_summary_template)    
    data_summary_filled = data_summary_template.format(safe_interf=safe_interf, link_dir=link_dir, scs_str=scs_str, interf_l0=interf_l0, interf_l1=interf_l1, interf_l2=interf_l2, interf_l3=interf_l3)
    #print(data_summary_filled)
    aligned_summary = align_markdown_table(data_summary_filled)
    #print(aligned_summary)
    
    prompt = f"""
    You are an O-RAN fronthaul packet analyzer following the O-RAN fronthaul specification from www.o-ran.org. 
    Generate a professional report summarizing the following structured O-RAN fronthaul data:


    The report provides a high level overview of the contents of the fronthaul data.

    The report will contain a header mentioning the following:
    - The filename based on {fname}
    - Date and time of the report generation using the format {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    - A green checkmark emoji if the fronthaul data has no issues, or a red cross emoji if there are issues.
    - A list of issues if any are found in the fronthaul data or say "None" if no issues are found.
    - Elaborate on each line of the fronthaul data summary table in the detailed analysis 
    - Make sure all the lines of the data_summary_template exist starting from the Sub-carrier Spacing {scs_str}
    - Add a conclusion sentence in the end-user
    
    You have to use the following specific format for the report:
    
    ## Report Header

    {markdown_header}


    ## Fronthaul Data Summary

    {aligned_summary}

    ## Detailed Analysis
    **1. Sub-carrier Spacing** - 30 KHz reflects granularity, influencing latency and spectral efficiency.

    **2. Number of Antennas** - A value of 2 indicates a dual-antenna configuration, which may enhance signal reliability and increase throughput capabilities compared to single-antenna setups.

    """
    #print(prompt)
    sections = prompt.split("## ")
    #print(sections)
    header_section = sections[1] if len(sections) >= 1 else ""
    #print(header_section)
    summary_section = sections[2] if len(sections) >= 2 else ""
    #print(summary_section)
    analysis_section = sections[3] if len(sections) >= 3 else ""
    #print(analysis_section)
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return (response.choices[0].message.content, sections, header_section, summary_section, analysis_section)
    #return (sections, header_section, summary_section, analysis_section)


def main():
    logging.info("Watcher started. Waiting for .ready files…")
    last_seen = {}  # map ready_path -> last mtime
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        while True:
            try:
                for fname in os.listdir(INBOX):
                    if not fname.endswith(".ready"):
                        continue

                    ready_path = os.path.join(INBOX, fname)
                    csv_path = ready_path[:-6]  # remove ".ready"

                    if not os.path.exists(csv_path):
                        continue  # ignore orphan flags

                    # detect new or re-created .ready file
                    mtime = os.path.getmtime(ready_path)
                    if ready_path not in last_seen or mtime > last_seen[ready_path]:
                        last_seen[ready_path] = mtime
                        pool.submit(handle_ready, csv_path, ready_path)
                        print("Successfully Processed", flush=True)

                        (genReport, sections, header_section, summary_section, analysis_section) = generate_report("temp.pcap.csv")
                        #print(genReport, sections, header_section, summary_section, analysis_section)
                        print(genReport)
                        #print(sections, header_section, summary_section, analysis_section)
                        report_path = save_report_for_csv(csv_path, genReport)
                        write_done_flag(report_path)

            except Exception:
                logging.exception("Watcher loop error")
            time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
