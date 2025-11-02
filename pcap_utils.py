import struct
import numpy as np
from scapy.all import rdpcap

def decode_bfp9_payload(payload: bytes, start_offset: int = 0, prb_count: int = 273):
    iq_samples = []
    i = start_offset
    prb_size = 28

    for prb_idx in range(prb_count):
        # if i + prb_size > len(payload):
            # break

        # Exponent is in lower 4 bits of PRB header
        exponent = payload[i] & 0x0F
        # if (prb_idx == 0):
            # print(exponent)
        scale = 2 ** (-exponent)
        #print(exponent)
        i += 1  # Move past header

        prb_data = payload[i:i+27]  # 27 bytes of packed BFP9 data
        bits = int.from_bytes(prb_data, byteorder='big')  # Convert to bitstream
        for k in range(12):
            shift = (11 - k) * 18
            sample_bits = (bits >> shift) & 0x3FFFF  # 18 bits mask

            i_part = (sample_bits >> 9) & 0x1FF
            q_part = sample_bits & 0x1FF

            # Convert 9-bit signed to int
            if i_part >= 256:
                i_part -= 512
            if q_part >= 256:
                q_part -= 512

            iq = complex(i_part / scale, q_part / scale)
            iq_samples.append(iq)

        i += 27  # Move to next PRB

    return iq_samples
    
def extract_oran_uplane(pcap_file, start_offset=0, mav=0):
    pkts = rdpcap(pcap_file)
    results = []
    for pkt in pkts:
        try:
            raw = bytes(pkt["Raw"])
            if(mav):
                ecpri_hdr_len = 0
            else:
                ecpri_hdr_len = 4                
            header_offset = ecpri_hdr_len
            section_id = raw[header_offset + 9]
            frame_id = raw[header_offset + 5]
            subframe_id = raw[header_offset + 6] >> 4
            slot_id = raw[header_offset + 7] >> 6
            symbol_id = raw[header_offset + 7] & 0xf
            port_id = raw[header_offset + 1]
            payload1 = raw[header_offset + 12:header_offset + 12+28]
            payload = raw[header_offset + 12:]
            iq_samples = decode_bfp9_payload(payload, start_offset)  # adjust offset if needed
            if (results == []):
                print([hex(b) for b in payload1])

            results.append({
                "port_id": port_id,
                "subframe": subframe_id,
                "slot": slot_id,
                "symbol": symbol_id,
                "iq": iq_samples
            })
        except Exception as e:
            print(f"[WARN] Skipping packet due to error: {e}")
    return results
