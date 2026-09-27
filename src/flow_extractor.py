"""PCAP -> CICFlowMeter-style bidirectional flow features.

A lightweight re-implementation of the CICFlowMeter feature subset listed in
config.FEATURES, so the trained model can analyse any capture a user uploads
(Wireshark/tcpdump .pcap/.pcapng). Time-based features are in microseconds,
matching CICFlowMeter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURES

FLOW_TIMEOUT_S = 120.0   # CICFlowMeter default activity timeout
_US = 1e6


class _Flow:
    __slots__ = ("key", "src", "dst", "sport", "dport", "proto", "t0", "tl",
                 "fwd_len", "bwd_len", "fwd_ts", "bwd_ts", "all_ts", "all_len",
                 "fwd_hdr", "bwd_hdr", "flags", "init_fwd_win", "init_bwd_win", "closed")

    def __init__(self, key, src, dst, sport, dport, proto, ts):
        self.key, self.src, self.dst = key, src, dst
        self.sport, self.dport, self.proto = sport, dport, proto
        self.t0 = self.tl = ts
        self.fwd_len, self.bwd_len, self.fwd_ts, self.bwd_ts = [], [], [], []
        self.all_ts, self.all_len = [], []
        self.fwd_hdr = self.bwd_hdr = 0
        self.flags = {"F": 0, "S": 0, "R": 0, "P": 0, "A": 0, "U": 0}
        self.init_fwd_win = self.init_bwd_win = -1
        self.closed = False

    def add(self, ts, fwd, length, hdr, tcp_flags, win):
        self.tl = ts
        self.all_ts.append(ts)
        self.all_len.append(length)
        if fwd:
            self.fwd_len.append(length); self.fwd_ts.append(ts); self.fwd_hdr += hdr
            if win is not None and self.init_fwd_win < 0:
                self.init_fwd_win = win
        else:
            self.bwd_len.append(length); self.bwd_ts.append(ts); self.bwd_hdr += hdr
            if win is not None and self.init_bwd_win < 0:
                self.init_bwd_win = win
        if tcp_flags:
            for f in self.flags:
                if f in tcp_flags:
                    self.flags[f] += 1

    # -- feature computation ------------------------------------------------
    @staticmethod
    def _stats(a):
        if len(a) == 0:
            return 0.0, 0.0, 0.0, 0.0
        a = np.asarray(a, dtype=float)
        return float(a.max()), float(a.min()), float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0

    @staticmethod
    def _iat(ts):
        if len(ts) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        d = np.diff(np.asarray(ts, dtype=float)) * _US
        return float(d.sum()), float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0, float(d.max()), float(d.min())

    def features(self) -> dict:
        dur = (self.tl - self.t0) * _US
        dur_s = max(dur / _US, 1e-6)
        nf, nb = len(self.fwd_len), len(self.bwd_len)
        tf, tb = float(sum(self.fwd_len)), float(sum(self.bwd_len))
        fmax, fmin, fmean, fstd = self._stats(self.fwd_len)
        bmax, bmin, bmean, bstd = self._stats(self.bwd_len)
        pmax, pmin, pmean, pstd = self._stats(self.all_len)
        _, iat_mean, iat_std, iat_max, iat_min = self._iat(self.all_ts)
        ftot, fiat_mean, fiat_std, fiat_max, fiat_min = self._iat(self.fwd_ts)
        btot, biat_mean, biat_std, biat_max, biat_min = self._iat(self.bwd_ts)
        n = nf + nb
        f = {
            "Protocol": self.proto,
            "Flow Duration": dur,
            "Total Fwd Packets": nf,
            "Total Bwd Packets": nb,
            "Total Length of Fwd Packets": tf,
            "Total Length of Bwd Packets": tb,
            "Fwd Packet Length Max": fmax, "Fwd Packet Length Min": fmin,
            "Fwd Packet Length Mean": fmean, "Fwd Packet Length Std": fstd,
            "Bwd Packet Length Max": bmax, "Bwd Packet Length Min": bmin,
            "Bwd Packet Length Mean": bmean, "Bwd Packet Length Std": bstd,
            "Flow Bytes/s": (tf + tb) / dur_s,
            "Flow Packets/s": n / dur_s,
            "Flow IAT Mean": iat_mean, "Flow IAT Std": iat_std,
            "Flow IAT Max": iat_max, "Flow IAT Min": iat_min,
            "Fwd IAT Total": ftot, "Fwd IAT Mean": fiat_mean, "Fwd IAT Std": fiat_std,
            "Fwd IAT Max": fiat_max, "Fwd IAT Min": fiat_min,
            "Bwd IAT Total": btot, "Bwd IAT Mean": biat_mean, "Bwd IAT Std": biat_std,
            "Bwd IAT Max": biat_max, "Bwd IAT Min": biat_min,
            "Fwd Header Length": self.fwd_hdr, "Bwd Header Length": self.bwd_hdr,
            "Fwd Packets/s": nf / dur_s, "Bwd Packets/s": nb / dur_s,
            "Packet Length Min": pmin, "Packet Length Max": pmax,
            "Packet Length Mean": pmean, "Packet Length Std": pstd,
            "Packet Length Variance": pstd ** 2,
            "FIN Flag Count": self.flags["F"], "SYN Flag Count": self.flags["S"],
            "RST Flag Count": self.flags["R"], "PSH Flag Count": self.flags["P"],
            "ACK Flag Count": self.flags["A"], "URG Flag Count": self.flags["U"],
            "Down/Up Ratio": (nb / nf) if nf else 0.0,
            "Average Packet Size": (tf + tb) / n if n else 0.0,
            "Init Fwd Win Bytes": self.init_fwd_win,
            "Init Bwd Win Bytes": self.init_bwd_win,
        }
        meta = {"Src IP": self.src, "Dst IP": self.dst, "Src Port": self.sport,
                "Dst Port": self.dport, "Start Time": self.t0}
        return {**meta, **f}


def _parse(pkt):
    """Return (src, dst, sport, dport, proto, length, hdr_len, flags, win) or None."""
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.inet6 import IPv6
    if IP in pkt:
        ip = pkt[IP]; src, dst, proto = ip.src, ip.dst, int(ip.proto)
    elif IPv6 in pkt:
        ip = pkt[IPv6]; src, dst, proto = ip.src, ip.dst, int(ip.nh)
    else:
        return None
    sport = dport = 0
    flags = ""
    win = None
    hdr = 0
    if TCP in pkt:
        t = pkt[TCP]; sport, dport = int(t.sport), int(t.dport)
        flags = str(t.flags); win = int(t.window); hdr = int(t.dataofs or 5) * 4
        payload = len(t.payload)
    elif UDP in pkt:
        u = pkt[UDP]; sport, dport = int(u.sport), int(u.dport); hdr = 8
        payload = len(u.payload)
    elif ICMP in pkt:
        hdr = 8; payload = len(pkt[ICMP].payload)
    else:
        payload = len(ip.payload)
    return src, dst, sport, dport, proto, payload, hdr, flags, win


def extract_flows(pcap_path, max_packets: int | None = 2_000_000) -> pd.DataFrame:
    """Read a pcap/pcapng file and return one row per bidirectional flow."""
    import scapy.all  # noqa: F401  (registers Ethernet/IP dissectors before reading)
    from scapy.utils import PcapReader

    active: dict = {}
    done: list = []
    with PcapReader(str(pcap_path)) as reader:
        for i, pkt in enumerate(reader):
            if max_packets and i >= max_packets:
                break
            p = _parse(pkt)
            if p is None:
                continue
            src, dst, sport, dport, proto, length, hdr, flags, win = p
            ts = float(pkt.time)
            a, b = (src, sport), (dst, dport)
            key = (min(a, b), max(a, b), proto)
            fl = active.get(key)
            if fl is not None and (ts - fl.tl > FLOW_TIMEOUT_S or fl.closed):
                done.append(fl); fl = None
            if fl is None:
                fl = _Flow(key, src, dst, sport, dport, proto, ts)
                active[key] = fl
            fwd = (src == fl.src and sport == fl.sport)
            fl.add(ts, fwd, length, hdr, flags, win)
            if "F" in flags or "R" in flags:
                fl.closed = "R" in flags or fl.flags["F"] >= 2
    done.extend(active.values())
    rows = [f.features() for f in done]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Src IP", "Dst IP", "Src Port", "Dst Port", "Start Time"] + FEATURES)
    return df.sort_values("Start Time").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    out = extract_flows(sys.argv[1])
    print(out.head())
    print(f"{len(out)} flows")
