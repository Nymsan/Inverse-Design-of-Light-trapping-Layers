import json
import os

with open("Results/timing_results.json", "r") as f:
    data = json.load(f)

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"    \centering")
lines.append(r"    \resizebox{\textwidth}{!}{%")
lines.append(r"    \begin{tabular}{lcccc}")
lines.append(r"        \toprule")
lines.append(r"        \textbf{Model} & \textbf{Sequential Total (s)} & \textbf{Batched Total (s)} & \textbf{Speedup (Sequential)} & \textbf{Speedup (Batched)} \\")
lines.append(r"        \midrule")

torcwa_seq_total = data["torcwa_sequential_total_s"]
num_samples = data["num_samples"]

# Add TORCWA baseline row
lines.append(f"        TORCWA & {torcwa_seq_total:.1f} & -- & 1$\\times$ & -- \\\\")

models = data["models"]
for m, stats in models.items():
    m_name = m.replace(".pt", "").replace("_", r"\_").upper() if "siren" in m else m.replace(".pt", "").replace("_", r"\_").title()
    if m_name == "Forward\\_Mlp": m_name = "Forward MLP"
    if m_name == "Skip\\_Cnn": m_name = "Skip CNN"
    
    seq_tot = stats["surrogate_sequential_total_s"]
    bat_tot = stats["surrogate_batched_total_s"]
    speed_seq = stats["speedup_sequential"]
    speed_bat = stats["speedup_batched_surrogate"]
    
    lines.append(f"        {m_name} & {seq_tot:.4f} & {bat_tot:.4f} & {speed_seq:,.0f}$\\times$ & {speed_bat:,.0f}$\\times$ \\\\")

lines.append(r"        \bottomrule")
lines.append(r"    \end{tabular}%")
lines.append(r"    }")
lines.append(r"    \caption{Computational timing comparison between TORCWA and surrogate models for evaluating 10 sample structures. Surrogate models show massive acceleration, especially when batched.}")
lines.append(r"    \label{tab:timing_comparison}")
lines.append(r"\end{table}")

os.makedirs("Report/tables", exist_ok=True)
with open("Report/tables/timing_stats.tex", "w") as f:
    f.write("\n".join(lines) + "\n")

print("Generated Report/tables/timing_stats.tex")

