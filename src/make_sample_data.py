"""Generate a SYNTHETIC sample dataset for smoke-testing the pipeline and UI.

It crafts packet captures for benign traffic and the 7 attack categories of
CIC IoT-DIAD 2024 with scapy, writes them as .pcap files, then runs our own
flow extractor to build a CICFlowMeter-format CSV (data/sample/sample_flows.csv).

NOTE: This data is only for demonstrating that everything runs end-to-end.
Report results must come from the real CIC IoT-DIAD 2024 dataset in data/raw/.
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
from scapy.all import GRE, ICMP, IP, TCP, UDP, DNS, DNSQR, DNSRR, Raw, wrpcap

from config import DATA_SAMPLE, FEATURES, SEED
from flow_extractor import extract_flows

R = random.Random(SEED)
NP = np.random.default_rng(SEED)
SUBNET = "192.168.1."
SERVERS = [f"192.168.1.{i}" for i in range(2, 20)]          # IoT devices / servers
CLIENTS = [f"192.168.1.{i}" for i in range(100, 160)]
EXTERNAL = [f"{R.randint(11, 220)}.{R.randint(0, 255)}.{R.randint(0, 255)}.{R.randint(1, 254)}" for _ in range(400)]


def _p(pkt, t):
    pkt.time = t
    return pkt


def _payload(n):
    return Raw(load=os.urandom(max(0, int(n))))


def tcp_session(cli, srv, dport, t, exchanges, gap=(0.005, 0.2), rtt=(0.001, 0.03),
                win=(8192, 65535), close=True, sport=None):
    """Full TCP session: 3-way handshake, request/response exchanges, FIN close."""
    sport = sport or R.randint(1024, 65000)
    pk = []
    r = lambda: R.uniform(*rtt)
    w = R.randint(*win)
    pk.append(_p(IP(src=cli, dst=srv) / TCP(sport=sport, dport=dport, flags="S", window=w), t)); t += r()
    pk.append(_p(IP(src=srv, dst=cli) / TCP(sport=dport, dport=sport, flags="SA", window=R.randint(*win)), t)); t += r()
    pk.append(_p(IP(src=cli, dst=srv) / TCP(sport=sport, dport=dport, flags="A", window=w), t)); t += r()
    for req, resp in exchanges:
        pk.append(_p(IP(src=cli, dst=srv) / TCP(sport=sport, dport=dport, flags="PA", window=w) / _payload(req), t)); t += r()
        remaining = resp
        while remaining > 0:
            seg = min(1448, remaining)
            pk.append(_p(IP(src=srv, dst=cli) / TCP(sport=dport, dport=sport, flags="PA" if seg == remaining else "A") / _payload(seg), t))
            t += R.uniform(0.0002, 0.003)
            remaining -= seg
        pk.append(_p(IP(src=cli, dst=srv) / TCP(sport=sport, dport=dport, flags="A", window=w), t))
        t += R.uniform(*gap)
    if close:
        pk.append(_p(IP(src=cli, dst=srv) / TCP(sport=sport, dport=dport, flags="FA", window=w), t)); t += r()
        pk.append(_p(IP(src=srv, dst=cli) / TCP(sport=dport, dport=sport, flags="FA"), t)); t += r()
        pk.append(_p(IP(src=cli, dst=srv) / TCP(sport=sport, dport=dport, flags="A", window=w), t))
    return pk


# ---------------------------------------------------------------------------
# Per-category flow generators. Each returns a list of packets for ONE flow.
# ---------------------------------------------------------------------------
def gen_benign(t):
    kind = R.choices(["https", "http", "dns", "mqtt", "ntp", "stream"], [30, 15, 20, 15, 5, 15])[0]
    cli, srv = R.choice(CLIENTS), R.choice(SERVERS + EXTERNAL[:50])
    if kind in ("https", "http"):
        ex = [(R.randint(200, 700), int(NP.lognormal(8, 1.2))) for _ in range(R.randint(1, 6))]
        return tcp_session(cli, srv, 443 if kind == "https" else 80, t, ex, gap=(0.05, 2.0))
    if kind == "mqtt":
        ex = [(R.randint(2, 120), R.randint(2, 60)) for _ in range(R.randint(2, 12))]
        return tcp_session(cli, srv, 1883, t, ex, gap=(1.0, 15.0), win=(1024, 8192))
    if kind == "stream":
        ex = [(R.randint(300, 500), R.randint(20000, 200000)) for _ in range(R.randint(2, 5))]
        return tcp_session(cli, srv, 443, t, ex, gap=(0.5, 4.0))
    if kind == "dns":
        sp = R.randint(1024, 65000); name = f"host{R.randint(1, 999)}.example.com"
        q = _p(IP(src=cli, dst="192.168.1.1") / UDP(sport=sp, dport=53) / DNS(rd=1, qd=DNSQR(qname=name)), t)
        a = _p(IP(src="192.168.1.1", dst=cli) / UDP(sport=53, dport=sp) /
               DNS(qr=1, qd=DNSQR(qname=name), an=DNSRR(rrname=name, rdata=R.choice(EXTERNAL))), t + R.uniform(0.002, 0.05))
        return [q, a]
    sp = R.randint(1024, 65000)                                            # ntp
    return [_p(IP(src=cli, dst=srv) / UDP(sport=sp, dport=123) / _payload(48), t),
            _p(IP(src=srv, dst=cli) / UDP(sport=123, dport=sp) / _payload(48), t + R.uniform(0.01, 0.1))]


def gen_ddos(t):
    kind = R.choice(["syn", "udp", "icmp", "ack", "http"])
    src, dst = R.choice(EXTERNAL), R.choice(SERVERS)
    sp, dp = R.randint(1024, 65000), R.choice([80, 443, 8080, 53])
    if kind == "syn":
        n = R.randint(1, 3)
        return [_p(IP(src=src, dst=dst) / TCP(sport=sp, dport=dp, flags="S", window=R.choice([512, 1024, 64240])), t + i * R.uniform(1e-5, 1e-3)) for i in range(n)]
    if kind == "ack":
        n = R.randint(5, 60)
        return [_p(IP(src=src, dst=dst) / TCP(sport=sp, dport=dp, flags="A", window=512) / _payload(R.randint(0, 20)), t + i * R.uniform(1e-5, 5e-4)) for i in range(n)]
    if kind == "http":
        ex = [(R.randint(60, 120), R.randint(100, 400)) for _ in range(R.randint(10, 40))]
        return tcp_session(src, dst, 80, t, ex, gap=(0.0001, 0.002), rtt=(0.0001, 0.001))
    n, size = R.randint(20, 300), R.randint(0, 1400)
    if kind == "udp":
        return [_p(IP(src=src, dst=dst) / UDP(sport=sp, dport=dp) / _payload(size), t + i * R.uniform(1e-5, 5e-4)) for i in range(n)]
    return [_p(IP(src=src, dst=dst) / ICMP() / _payload(size), t + i * R.uniform(1e-5, 5e-4)) for i in range(n)]


def gen_dos(t):
    kind = R.choice(["syn", "udp", "slowloris", "http"])
    src, dst = R.choice(CLIENTS[:5] + EXTERNAL[:10]), R.choice(SERVERS)
    sp = R.randint(1024, 65000)
    if kind == "syn":
        n = R.randint(30, 400)
        return [_p(IP(src=src, dst=dst) / TCP(sport=sp, dport=80, flags="S", window=1024), t + i * R.uniform(1e-4, 2e-3)) for i in range(n)]
    if kind == "udp":
        n = R.randint(100, 600)
        return [_p(IP(src=src, dst=dst) / UDP(sport=sp, dport=R.randint(1, 65535)) / _payload(R.randint(500, 1400)), t + i * R.uniform(1e-4, 1e-3)) for i in range(n)]
    if kind == "slowloris":
        ex = [(R.randint(5, 30), 0) for _ in range(R.randint(5, 20))]
        return tcp_session(src, dst, 80, t, ex, gap=(5.0, 15.0), close=False)
    ex = [(R.randint(80, 200), R.randint(200, 800)) for _ in range(R.randint(40, 120))]
    return tcp_session(src, dst, 80, t, ex, gap=(0.0005, 0.005), rtt=(0.0002, 0.002))


def gen_recon(t):
    kind = R.choice(["syn_scan", "ping", "xmas", "fin", "connect"])
    src, dst = R.choice(CLIENTS[:3] + EXTERNAL[:5]), R.choice(SERVERS)
    sp, dp = R.randint(30000, 65000), R.randint(1, 10000)
    if kind == "ping":
        return [_p(IP(src=src, dst=dst) / ICMP(), t), _p(IP(src=dst, dst=src) / ICMP(type=0), t + R.uniform(1e-4, 5e-3))]
    if kind in ("xmas", "fin"):
        fl = "FPU" if kind == "xmas" else "F"
        return [_p(IP(src=src, dst=dst) / TCP(sport=sp, dport=dp, flags=fl, window=1024), t),
                _p(IP(src=dst, dst=src) / TCP(sport=dp, dport=sp, flags="RA", window=0), t + R.uniform(1e-4, 3e-3))]
    pk = [_p(IP(src=src, dst=dst) / TCP(sport=sp, dport=dp, flags="S", window=1024), t)]
    if R.random() < 0.3:
        pk.append(_p(IP(src=dst, dst=src) / TCP(sport=dp, dport=sp, flags="SA", window=29200), t + R.uniform(1e-4, 3e-3)))
        pk.append(_p(IP(src=src, dst=dst) / TCP(sport=sp, dport=dp, flags="R", window=0), t + R.uniform(3e-3, 6e-3)))
    else:
        pk.append(_p(IP(src=dst, dst=src) / TCP(sport=dp, dport=sp, flags="RA", window=0), t + R.uniform(1e-4, 3e-3)))
    return pk


def gen_web(t):
    cli, srv = R.choice(CLIENTS[:8] + EXTERNAL[:20]), R.choice(SERVERS)
    ex = [(R.randint(400, 1400), R.choice([R.randint(300, 600), R.randint(1500, 9000)])) for _ in range(R.randint(2, 15))]
    return tcp_session(cli, srv, R.choice([80, 8080, 443]), t, ex, gap=(0.02, 0.6))


def gen_brute(t):
    cli, srv = R.choice(CLIENTS[:4] + EXTERNAL[:10]), R.choice(SERVERS)
    dp = R.choice([22, 23, 21, 80])
    ex = [(R.randint(20, 90), R.randint(20, 90)) for _ in range(R.randint(3, 8))]
    return tcp_session(cli, srv, dp, t, ex, gap=(0.05, 0.4), win=(29200, 64240))


def gen_spoof(t):
    kind = R.choice(["dns", "ipspoof"])
    victim = R.choice(CLIENTS)
    if kind == "dns":
        sp, name = R.randint(1024, 65000), f"bank{R.randint(1, 9)}.example.com"
        n = R.randint(3, 40)
        return [_p(IP(src="192.168.1.1", dst=victim) / UDP(sport=53, dport=sp) /
                   DNS(id=R.randint(0, 65535), qr=1, qd=DNSQR(qname=name), an=DNSRR(rrname=name, rdata="10.66.66.66")),
                   t + i * R.uniform(1e-4, 2e-3)) for i in range(n)]
    srv = R.choice(SERVERS)
    ex = [(R.randint(40, 300), R.randint(40, 1200)) for _ in range(R.randint(2, 10))]
    return tcp_session(victim, srv, R.choice([80, 1883, 8883, 5683]), t, ex, gap=(0.001, 0.05), rtt=(0.02, 0.12))


def gen_mirai(t):
    kind = R.choice(["telnet_scan", "greeth", "udpplain", "greip"])
    bot, target = R.choice(CLIENTS[40:]), R.choice(EXTERNAL)
    if kind == "telnet_scan":
        sp = R.randint(1024, 65000)
        return [_p(IP(src=bot, dst=target) / TCP(sport=sp, dport=R.choice([23, 2323]), flags="S", window=R.randint(1000, 60000)), t)]
    n = R.randint(30, 400)
    if kind == "udpplain":
        return [_p(IP(src=bot, dst=target) / UDP(sport=R.randint(1024, 65000), dport=R.randint(1, 65535)) / _payload(512), t + i * R.uniform(1e-5, 3e-4)) for i in range(n)]
    return [_p(IP(src=bot, dst=target, proto=47) / GRE() / _payload(512 if kind == "greeth" else 530), t + i * R.uniform(1e-5, 3e-4)) for i in range(n)]


GENERATORS = {
    "Benign": gen_benign, "DDoS": gen_ddos, "DoS": gen_dos, "Recon": gen_recon,
    "Web-based": gen_web, "BruteForce": gen_brute, "Spoofing": gen_spoof, "Mirai": gen_mirai,
}


def build_category(cat, n_flows, t0=1.7e9):
    pkts, t = [], t0
    for _ in range(n_flows):
        pkts.extend(GENERATORS[cat](t))
        t += R.uniform(0.5, 3.0)
    pkts.sort(key=lambda p: p.time)
    return pkts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", type=int, default=400, help="flows per category")
    ap.add_argument("--demo-only", action="store_true", help="only rebuild demo files from existing sample_flows.csv")
    args = ap.parse_args()
    pcap_dir = DATA_SAMPLE / "pcaps"; pcap_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for cat in ([] if args.demo_only else GENERATORS):
        pk = build_category(cat, args.flows)
        path = pcap_dir / f"{cat}.pcap"
        wrpcap(str(path), pk)
        df = extract_flows(path)
        df["Label"] = cat
        frames.append(df)
        print(f"{cat:11s} {len(pk):7d} packets -> {len(df):5d} flows")
    if args.demo_only:
        full = pd.read_csv(DATA_SAMPLE / "sample_flows.csv")
    else:
        full = pd.concat(frames, ignore_index=True)
        full[FEATURES + ["Label"]].to_csv(DATA_SAMPLE / "sample_flows.csv", index=False)

    # A small mixed capture + CSV for demoing the Upload page.
    mixed = []
    for cat in GENERATORS:
        for _ in range(R.randint(3, 6) if cat != "Benign" else 400):
            pk = GENERATORS[cat](1.8e9 + R.uniform(0, 120))
            if cat == "DoS" and len({(p[IP].dst, p.dport) for p in pk if UDP in p}) > 20:
                continue            # skip random-port UDP floods: they explode into hundreds of 1-packet flows
            mixed.extend(pk)
    mixed.sort(key=lambda p: p.time)
    wrpcap(str(DATA_SAMPLE / "demo_mixed_traffic.pcap"), mixed)
    demo = pd.concat([g.sample(min(len(g), 40), random_state=SEED) for _, g in full.groupby("Label")])
    demo.drop(columns=["Label"]).sample(frac=1, random_state=SEED).to_csv(DATA_SAMPLE / "demo_flows_unlabelled.csv", index=False)
    print(f"sample dataset: {len(full)} flows -> {DATA_SAMPLE / 'sample_flows.csv'}")


if __name__ == "__main__":
    main()
