# Simulation of MI-MAC protocol (no hardware) - Python implementation
# --------------------------------------------------------------
# This code simulates the Medium Access Control (MAC) protocol
# for Magneto-Inductive Wireless Sensor Networks (MIWSN) using
# 3 coil configurations: Sequential, Simultaneous, and Hybrid.
# It models node behavior (Idle → Sense → Transmit → Receive)
# and computes energy + throughput results — no hardware needed.
# --------------------------------------------------------------

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Simulation parameters ----------
N = 10                   # number of nodes
sim_time = 200.0         # total simulation time (ms)
lambda_rate = 0.02       # message generation rate per ms per node
Vcc = 3.3                # supply voltage (V)

# Durations (ms)
tau_txW = 1.0
tau_txAck = 0.7
tau_txData = 3.0
tau_sense = 0.5
tau_safe = 0.3
tau_noAck = tau_txW + tau_txAck + tau_safe

# Currents (Amperes)
I_idle = 60e-6
I_receive = 0.49e-3
I_sense = 0.74e-3
I_tx_low = 220e-3
I_tx_high = 528e-3

# Packet sizes (bytes)
size_W = 13
size_ACK = 9
size_DATA = 24

configs = ["config1", "config2", "config3"]  # sequential, simultaneous, hybrid

# ---------- Helper functions ----------
def schedule_attempts(N, sim_time, lam):
    """Generate random communication attempts (Poisson arrivals)."""
    attempts = []
    for node in range(N):
        t = 0.0
        while t < sim_time:
            isi = np.random.exponential(1.0 / lam)
            t += isi
            if t >= sim_time:
                break
            target = random.choice([x for x in range(N) if x != node])
            attempts.append({'node': node, 'time': t, 'target': target})
    attempts.sort(key=lambda x: x['time'])
    return attempts

# ---------- Simulation core ----------
def run_simulation(config_name):
    attempts = schedule_attempts(N, sim_time, lambda_rate)
    transmissions = []
    energy = np.zeros(N)
    bytes_sent = np.zeros(N)
    bytes_success = np.zeros(N)
    attempts_done = 0

    for a in attempts:
        sender = a['node']
        target = a['target']
        start_time = a['time']
        time = start_time

        # Sense channel
        time += tau_sense
        energy[sender] += (I_sense * Vcc) * tau_sense / 1000.0
        energy[target] += (I_receive * Vcc) * (tau_sense / 2.0) / 1000.0

        # Transmit WakeUp
        if config_name == "config1":
            W_dur, I_tx = tau_txW * 3.0, I_tx_low
        elif config_name == "config2":
            W_dur, I_tx = tau_txW, I_tx_high
        else:  # hybrid
            W_dur, I_tx = tau_txW * 3.0, I_tx_high

        W_start, W_end = time, time + W_dur
        transmissions.append({'start': W_start, 'end': W_end, 'sender': sender, 'target': target, 'type': 'W'})
        energy[sender] += (I_tx * Vcc) * (W_dur / 1000.0)
        time = W_end

        # Check collision
        collided = False
        for tr in transmissions[:-1]:
            if tr['target'] == target and tr['type'] == 'W':
                if not (tr['end'] <= W_start or tr['start'] >= W_end):
                    collided = True
                    break

        if collided:
            energy[sender] += (I_idle * Vcc) * (tau_noAck / 1000.0)
            bytes_sent[sender] += size_W
            attempts_done += 1
            continue

        energy[target] += (I_receive * Vcc) * (W_dur / 1000.0)

        # ACK transmission
        if config_name == "config2":
            ack_I, ack_dur = I_tx_high, tau_txAck
        else:
            ack_I, ack_dur = I_tx_low, tau_txAck

        ACK_start = time + 0.1
        ACK_end = ACK_start + ack_dur
        transmissions.append({'start': ACK_start, 'end': ACK_end, 'sender': target, 'target': sender, 'type': 'ACK'})
        energy[target] += (ack_I * Vcc) * (ack_dur / 1000.0)
        energy[sender] += (I_receive * Vcc) * (ack_dur / 1000.0)
        time = ACK_end

        # Data transmission
        if config_name == "config1":
            data_dur, data_I = tau_txData * 3.0, I_tx_low
        elif config_name == "config2":
            data_dur, data_I = tau_txData, I_tx_high
        else:
            data_dur, data_I = tau_txData, I_tx_low

        DATA_start = time + 0.05
        DATA_end = DATA_start + data_dur
        transmissions.append({'start': DATA_start, 'end': DATA_end, 'sender': sender, 'target': target, 'type': 'DATA'})
        energy[sender] += (data_I * Vcc) * (data_dur / 1000.0)
        energy[target] += (I_receive * Vcc) * (data_dur / 1000.0)
        bytes_sent[sender] += (size_W + size_ACK + size_DATA)
        bytes_success[sender] += size_DATA
        attempts_done += 1

    # Add baseline idle energy
    for n in range(N):
        energy[n] += (I_idle * Vcc) * (sim_time / 1000.0)

    df = pd.DataFrame({
        'node': np.arange(N),
        'energy_J': energy,
        'bytes_sent': bytes_sent,
        'bytes_success': bytes_success
    })
    total_throughput_bytes = df['bytes_success'].sum()
    return df, total_throughput_bytes, attempts_done

# ---------- Run simulations ----------
results = {}
for cfg in configs:
    df, total_bytes, attempts_done = run_simulation(cfg)
    results[cfg] = {'df': df, 'total_bytes': total_bytes, 'attempts_done': attempts_done}

# ---------- Show results ----------
for cfg in configs:
    print(f"\nConfiguration: {cfg}")
    print(f" Total successful data bytes delivered: {results[cfg]['total_bytes']:.1f}")
    print(f" Total attempts processed: {results[cfg]['attempts_done']}")
    print(results[cfg]['df'][['energy_J', 'bytes_success']].describe())

# ---------- Plot graphs ----------
for cfg in configs:
    df = results[cfg]['df']
    plt.figure(figsize=(6, 3.5))
    plt.bar(df['node'], df['energy_J'])
    plt.title(f'Per-node Energy (J) - {cfg}')
    plt.xlabel('Node')
    plt.ylabel('Energy (J)')
    plt.tight_layout()
    plt.show()

throughputs = [results[c]['total_bytes'] for c in configs]
plt.figure(figsize=(6, 3.5))
plt.bar(configs, throughputs)
plt.title('Total Delivered Data (bytes) by Configuration')
plt.ylabel('Bytes delivered')
plt.tight_layout()
plt.show()

print("\nExample per-node data for config3:\n")
print(results['config3']['df'])
