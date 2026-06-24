import re

with open('train_active_learning.py', 'r') as f:
    code = f.read()

# 1. Update argparse
code = code.replace(
    'parser.add_argument(\'--expand_amps\', type=float, default=None, help="Temporarily expand the maximum amplitude bounds (e.g., to 25.0 nm) for Active Learning")',
    'parser.add_argument(\'--expand_amps\', type=float, default=None, help="Temporarily expand the maximum amplitude bounds (e.g., to 25.0 nm) for Active Learning")\n    parser.add_argument(\'--order_N\', type=int, default=None, help="RCWA Order N override for Torcwa evaluation")\n    parser.add_argument(\'--height_per_layer\', type=float, default=None, help="RCWA height per layer override for Torcwa evaluation")'
)

# 2. Update evaluate_oracle definition
code = code.replace(
    'def evaluate_oracle(geometries: torch.Tensor, mat_name: str, stats: dict, device: torch.device):',
    'def evaluate_oracle(geometries: torch.Tensor, mat_name: str, stats: dict, device: torch.device, order_N: int = None, height_per_layer: float = None):'
)

# 3. Apply RCWA Config overrides in evaluate_oracle
code = code.replace(
    "        base_config.reflector_type = 'pec'\n        \n    wavelengths = torch.linspace(300, 1100, stats[\"n_wavelengths\"] // 2, dtype=torch.float64, device=device) + 1e-3",
    "        base_config.reflector_type = 'pec'\n        \n    if order_N is not None:\n        base_config.order_N = order_N\n    if height_per_layer is not None:\n        base_config.height_per_layer = height_per_layer\n        \n    wavelengths = torch.linspace(300, 1100, stats[\"n_wavelengths\"] // 2, dtype=torch.float64, device=device) + 1e-3"
)

# 4. Pass args in main loop
code = code.replace(
    'true_curves = evaluate_oracle(proposals, mat_name, stats, device)',
    'true_curves = evaluate_oracle(proposals, mat_name, stats, device, order_N=args.order_N, height_per_layer=args.height_per_layer)'
)

with open('train_active_learning.py', 'w') as f:
    f.write(code)
