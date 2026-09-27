"""Central configuration: paths, feature schema, label taxonomy."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLE = ROOT / "data" / "sample"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

for _p in (DATA_RAW, DATA_SAMPLE, DATA_PROCESSED, MODELS, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# Feature schema.
# A subset of CICFlowMeter flow features chosen so that the SAME features can be
# computed by our own PCAP extractor (src/flow_extractor.py). This keeps the
# trained model usable on user-uploaded captures.
# Identifiers (IPs, ports, timestamps) are deliberately excluded to avoid
# shortcut learning / data leakage.
# ---------------------------------------------------------------------------
FEATURES = [
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Bwd Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
]

# Different CICFlowMeter versions spell columns differently. Map every known
# spelling (lower-cased, spaces/underscores stripped) to our canonical name.
_ALIASES = {
    "Protocol": ["protocol", "proto"],
    "Flow Duration": ["flowduration"],
    "Total Fwd Packets": ["totalfwdpackets", "totalfwdpacket", "totfwdpkts", "fwdpkts"],
    "Total Bwd Packets": ["totalbwdpackets", "totalbwdpacket", "totalbackwardpackets", "totbwdpkts", "bwdpkts"],
    "Total Length of Fwd Packets": ["totallengthoffwdpackets", "totallengthoffwdpacket", "totlenfwdpkts", "fwdpktslentotal"],
    "Total Length of Bwd Packets": ["totallengthofbwdpackets", "totallengthofbwdpacket", "totlenbwdpkts", "bwdpktslentotal"],
    "Fwd Packet Length Max": ["fwdpacketlengthmax", "fwdpktlenmax"],
    "Fwd Packet Length Min": ["fwdpacketlengthmin", "fwdpktlenmin"],
    "Fwd Packet Length Mean": ["fwdpacketlengthmean", "fwdpktlenmean"],
    "Fwd Packet Length Std": ["fwdpacketlengthstd", "fwdpktlenstd"],
    "Bwd Packet Length Max": ["bwdpacketlengthmax", "bwdpktlenmax"],
    "Bwd Packet Length Min": ["bwdpacketlengthmin", "bwdpktlenmin"],
    "Bwd Packet Length Mean": ["bwdpacketlengthmean", "bwdpktlenmean"],
    "Bwd Packet Length Std": ["bwdpacketlengthstd", "bwdpktlenstd"],
    "Flow Bytes/s": ["flowbytes/s", "flowbytss", "flowbytes/sec", "flowbytespersec"],
    "Flow Packets/s": ["flowpackets/s", "flowpktss", "flowpackets/sec"],
    "Flow IAT Mean": ["flowiatmean"],
    "Flow IAT Std": ["flowiatstd"],
    "Flow IAT Max": ["flowiatmax"],
    "Flow IAT Min": ["flowiatmin"],
    "Fwd IAT Total": ["fwdiattotal", "fwdiattot"],
    "Fwd IAT Mean": ["fwdiatmean"],
    "Fwd IAT Std": ["fwdiatstd"],
    "Fwd IAT Max": ["fwdiatmax"],
    "Fwd IAT Min": ["fwdiatmin"],
    "Bwd IAT Total": ["bwdiattotal", "bwdiattot"],
    "Bwd IAT Mean": ["bwdiatmean"],
    "Bwd IAT Std": ["bwdiatstd"],
    "Bwd IAT Max": ["bwdiatmax"],
    "Bwd IAT Min": ["bwdiatmin"],
    "Fwd Header Length": ["fwdheaderlength", "fwdheaderlen"],
    "Bwd Header Length": ["bwdheaderlength", "bwdheaderlen"],
    "Fwd Packets/s": ["fwdpackets/s", "fwdpktss"],
    "Bwd Packets/s": ["bwdpackets/s", "bwdpktss"],
    "Packet Length Min": ["packetlengthmin", "minpacketlength", "pktlenmin"],
    "Packet Length Max": ["packetlengthmax", "maxpacketlength", "pktlenmax"],
    "Packet Length Mean": ["packetlengthmean", "pktlenmean"],
    "Packet Length Std": ["packetlengthstd", "pktlenstd"],
    "Packet Length Variance": ["packetlengthvariance", "pktlenvar"],
    "FIN Flag Count": ["finflagcount", "finflagcnt"],
    "SYN Flag Count": ["synflagcount", "synflagcnt"],
    "RST Flag Count": ["rstflagcount", "rstflagcnt"],
    "PSH Flag Count": ["pshflagcount", "pshflagcnt"],
    "ACK Flag Count": ["ackflagcount", "ackflagcnt"],
    "URG Flag Count": ["urgflagcount", "urgflagcnt"],
    "Down/Up Ratio": ["down/upratio", "downupratio"],
    "Average Packet Size": ["averagepacketsize", "pktsizeavg"],
    "Init Fwd Win Bytes": ["initfwdwinbytes", "initwinbytesforward", "initfwdwinbyts", "fwdinitwinbytes"],
    "Init Bwd Win Bytes": ["initbwdwinbytes", "initwinbytesbackward", "initbwdwinbyts", "bwdinitwinbytes"],
}


def _key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "")


ALIAS_LOOKUP = {}
for canonical, aliases in _ALIASES.items():
    ALIAS_LOOKUP[_key(canonical)] = canonical
    for a in aliases:
        ALIAS_LOOKUP[_key(a)] = canonical

# ---------------------------------------------------------------------------
# Label taxonomy (CIC IoT-DIAD 2024: benign + 7 attack categories)
# ---------------------------------------------------------------------------
CATEGORIES = ["Benign", "DDoS", "DoS", "Recon", "Web-based", "BruteForce", "Spoofing", "Mirai"]

# Ordered keyword rules; first match wins (so "mirai-udpplain" -> Mirai, "ddos" before "dos").
_CATEGORY_RULES = [
    ("Mirai", ["mirai"]),
    ("Benign", ["benign", "normal"]),
    ("DDoS", ["ddos"]),
    ("DoS", ["dos"]),
    ("BruteForce", ["brute", "dictionary", "password"]),
    ("Spoofing", ["spoof", "mitm", "arp"]),
    ("Web-based", ["sql", "xss", "commandinjection", "command_injection", "backdoor",
                   "upload", "browserhijack", "browser_hijack", "web"]),
    ("Recon", ["recon", "scan", "sweep", "osscan", "vulnerab", "hostdiscovery", "portscan"]),
]


def to_category(text: str):
    """Map a raw label or file path to one of CATEGORIES (None if unknown)."""
    t = str(text).lower().replace(" ", "").replace("-", "")
    for cat, kws in _CATEGORY_RULES:
        if any(k.replace("-", "") in t for k in kws):
            return cat
    return None


SEVERITY = {
    "Benign": "None", "Recon": "Low", "Spoofing": "Medium", "BruteForce": "Medium",
    "Web-based": "High", "DoS": "High", "DDoS": "Critical", "Mirai": "Critical",
    "Unknown attack": "High",
}

MITIGATION = {
    "Benign": "No action required.",
    "DDoS": "Enable upstream rate-limiting / scrubbing, apply SYN cookies, block top offending source ranges.",
    "DoS": "Rate-limit the source IP, tune connection timeouts, and enable SYN cookies on the target.",
    "Recon": "Block the scanning host at the firewall, review exposed services, and enable port-scan detection.",
    "Web-based": "Enable WAF rules (SQLi/XSS), sanitise inputs, patch the web application, and review logs for compromise.",
    "BruteForce": "Lock out the account after failed attempts, enforce MFA, and block the source IP.",
    "Spoofing": "Enable Dynamic ARP Inspection / DNSSEC, use static ARP entries for critical hosts, and segment the network.",
    "Mirai": "Isolate the infected device, reset its default credentials, close telnet (23/2323), and re-flash the firmware.",
    "Unknown attack": "Novel behaviour detected: quarantine the flow's source, capture full PCAP for forensic analysis.",
}
