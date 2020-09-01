from __future__ import print_function

ISCX_PKT_DATASET_HEADER = [
    "Pkt index per flow", "Dst Port",         "Protocol",          "Flow Duration",     "Tot Fwd Pkts",
    "Tot Bwd Pkts",       "TotLen Fwd Pkts",  "TotLen Bwd Pkts",   "Fwd Pkt Len Max",   "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",   "Fwd Pkt Len Std",  "Bwd Pkt Len Max",   "Bwd Pkt Len Min",   "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",    "Flow Byts/s",      "Flow Pkts/s",       "Flow IAT Mean",     "Flow IAT Std",
    "Flow IAT Max",       "Flow IAT Min",     "Fwd IAT Tot",       "Fwd IAT Mean",      "Fwd IAT Std",
    "Fwd IAT Max",        "Fwd IAT Min",      "Bwd IAT Tot",       "Bwd IAT Mean",      "Bwd IAT Std",
    "Bwd IAT Max",        "Bwd IAT Min",      "Fwd PSH Flags",     "Bwd PSH Flags",     "Fwd URG Flags",
    "Bwd URG Flags",      "Fwd Header Len",   "Bwd Header Len",    "Fwd Pkts/s",        "Bwd Pkts/s",
    "Pkt Len Min",        "Pkt Len Max",      "Pkt Len Mean",      "Pkt Len Std",       "Pkt Len Var",
    "FIN Flag Cnt",       "SYN Flag Cnt",     "RST Flag Cnt",      "PSH Flag Cnt",      "ACK Flag Cnt",
    "URG Flag Cnt",       "CWE Flag Count",   "ECE Flag Cnt",      "Down/Up Ratio",     "Pkt Size Avg",
    "Fwd Seg Size Avg",   "Bwd Seg Size Avg", "Fwd Byts/b Avg",    "Fwd Pkts/b Avg",    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",     "Bwd Pkts/b Avg",   "Bwd Blk Rate Avg",  "Subflow Fwd Pkts",  "Subflow Fwd Byts",
    "Subflow Bwd Pkts",   "Subflow Bwd Byts", "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
    "Fwd Seg Size Min",   "Active Mean",      "Active Std",        "Active Max",        "Active Min",
    "Idle Mean",          "Idle Std",         "Idle Max",          "Idle Min",          "Label"
]

# Use it only when open very original dataset
ISCX_DATASET_ENCODING = "ISO-8859-1"

# Protocols
HOPOPT = 0
TCP = 6
UDP = 17

USEC = 1e6
# Flow timeout is defined as 120 in original CICFlowMeter source code
FLOW_TIMEOUT = 60 * USEC

log_path = '/tf/md0/thkim/log/'

RAID_PATH = '/tf/md0/thkim/'

# Slot work const
SLOT_PATH = '/tf/md0/thkim/slot_work/'

