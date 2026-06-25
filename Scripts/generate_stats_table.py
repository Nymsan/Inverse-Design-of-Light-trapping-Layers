import json, os

dirs = [
    ('Si_TiO2_Si3N4_frac_0.1', '10\\%'),
    ('Si_TiO2_Si3N4_frac_0.25', '25\\%'),
    ('Si_TiO2_Si3N4_frac_0.5', '50\\%'),
    ('Si_TiO2_Si3N4', '100\\%')
]

models = ['forward_mlp', 'skip_cnn', 'siren']

lines = []
for d, name in dirs:
    metrics_file = f"Checkpoints/{d}/evaluation/forward/all_data/forward_metrics.json"
    hist_file = f"Checkpoints/{d}/forward_history.json"
    
    if not os.path.exists(metrics_file):
        metrics_file = f"../{metrics_file}"
        hist_file = f"../{hist_file}"
        if not os.path.exists(metrics_file):
            continue
            
    with open(metrics_file, 'r') as f:
        eval_metrics = json.load(f)
        
    with open(hist_file, 'r') as f:
        hist_data = json.load(f)
        
    stats = {}
    for m in models:
        if m in eval_metrics and m in hist_data:
            t_loss = min(hist_data[m].get('train_loss', [float('inf')]))
            v_loss = min(hist_data[m].get('val_loss', [float('inf')]))
            stats[m] = {
                't_loss': t_loss,
                'v_loss': v_loss,
                'vmae': eval_metrics[m].get('mae'),
                'vmax': eval_metrics[m].get('max_abs_error'),
                'vr2': eval_metrics[m].get('r2')
            }
            
    if not stats: continue
    
    best_tloss = min([s['t_loss'] for s in stats.values() if s['t_loss'] != float('inf')])
    best_vloss = min([s['v_loss'] for s in stats.values() if s['v_loss'] != float('inf')])
    best_vmae = min([s['vmae'] for s in stats.values() if s['vmae'] is not None])
    best_vmax = min([s['vmax'] for s in stats.values() if s['vmax'] is not None])
    best_vr2 = max([s['vr2'] for s in stats.values() if s['vr2'] is not None])
    
    label_printed = False
    for m in models:
        if m not in stats: continue
        s = stats[m]
        c_name = f"Trained on {name} train set" if not label_printed else ""
        label_printed = True
        
        svmae = f"\\textbf{{{s['vmae']:.4f}}}" if s['vmae'] == best_vmae else f"{s['vmae']:.4f}"
        svmax = f"\\textbf{{{s['vmax']:.4f}}}" if s['vmax'] == best_vmax else f"{s['vmax']:.4f}"
        svr2 = f"\\textbf{{{s['vr2']:.4f}}}" if s['vr2'] == best_vr2 else f"{s['vr2']:.4f}"
        
        m_name = m.replace('_', '\\_')
        lines.append(f"        {c_name} & \\texttt{{{m_name}}} & {svmae} & {svmax} & {svr2} \\\\")
    lines.append(r"        \midrule")

if lines and lines[-1].strip() == r"\midrule":
    lines.pop()

tabular_lines = [
    r"\begin{tabular}{llccc}",
    r"    \toprule",
    r"    \textbf{Dataset Setup} & \textbf{Model Type} & \textbf{Test MAE} & \textbf{Test Max Err} & \textbf{Test R$^2$} \\",
    r"    \midrule"
]
tabular_lines.extend(lines)
tabular_lines.append(r"    \bottomrule")
tabular_lines.append(r"\end{tabular}")

os.makedirs('Report/tables', exist_ok=True)
if not os.path.exists('Report/tables'):
    os.makedirs('../Report/tables', exist_ok=True)
    out_path = '../Report/tables/frac_models_stats.tex'
else:
    out_path = 'Report/tables/frac_models_stats.tex'
    
with open(out_path, 'w') as f:
    f.write("\n".join(tabular_lines) + "\n")
print(f"Wrote table rows to {out_path}")

