import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Utils.models import (
    ForwardMLP, SpatialCNN, SkipCNN, SIREN, TransformerForward,
    InverseDecoder, TandemNetwork, GenerativeTandemNetwork,
    GeometryEncoder, GeometryDecoder, SpectrumEncoder, ContrastiveVAE
)

import argparse

def get_args():
    p = argparse.ArgumentParser()
    # MLP
    p.add_argument("--mlp_hidden_dims", type=int, nargs="+", default=[512, 768, 512])
    # SpatialCNN
    p.add_argument("--cnn_conv_channels", type=int, nargs="+", default=[64, 128, 128, 64])
    p.add_argument("--cnn_kernel_size", type=int, default=7)
    p.add_argument("--cnn_fc_dims", type=int, nargs="+", default=[512, 128])
    # SkipCNN
    p.add_argument("--skipcnn_conv_channels", type=int, nargs="+", default=[32, 64, 128, 64])
    p.add_argument("--skipcnn_kernel_size", type=int, default=7)
    p.add_argument("--skipcnn_fc_dims", type=int, nargs="+", default=[256, 256])
    # SIREN
    p.add_argument("--siren_conv_channels", type=int, nargs="+", default=[32, 64, 64])
    p.add_argument("--siren_kernel_size", type=int, default=7)
    p.add_argument("--siren_fc_dims", type=int, nargs="+", default=[256, 128])
    # Transformer
    p.add_argument("--tf_d_model", type=int, default=128)
    p.add_argument("--tf_nhead", type=int, default=4)
    p.add_argument("--tf_dim_feedforward", type=int, default=512)
    p.add_argument("--tf_num_layers", type=int, default=3)
    # InverseDecoder
    p.add_argument("--inv_conv_channels", type=int, nargs="+", default=[32, 64, 128, 64])
    p.add_argument("--inv_kernel_size", type=int, default=7)
    p.add_argument("--inv_fc_dims", type=int, nargs="+", default=[256, 256])
    # CVAE
    p.add_argument("--cvae_geo_enc_conv", type=int, nargs="+", default=[32, 64, 64])
    p.add_argument("--cvae_geo_enc_kernel", type=int, default=7)
    p.add_argument("--cvae_geo_enc_fc", type=int, nargs="+", default=[256, 256])
    p.add_argument("--cvae_geo_dec_fc", type=int, nargs="+", default=[256, 256])
    p.add_argument("--cvae_spec_enc_conv", type=int, nargs="+", default=[32, 64, 128, 64])
    p.add_argument("--cvae_spec_enc_kernel", type=int, default=7)
    p.add_argument("--cvae_spec_enc_fc", type=int, nargs="+", default=[256, 256])
    return p.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

args = get_args()

print("--- Forward Models ---")
# Instantiate with typical parameters (from train_forward.py)
n_harmonics = 5
nx = 128
n_continuous = 12
n_wavelengths = 322
N_MATERIALS = 3

mlp = ForwardMLP(n_wavelengths=n_wavelengths, n_materials=N_MATERIALS, embed_dim=8, n_harmonics=n_harmonics, nx=nx, n_continuous=n_continuous, hidden_dims=tuple(args.mlp_hidden_dims))
print(f"ForwardMLP: {count_parameters(mlp):,}")

spatial = SpatialCNN(conv_channels=tuple(args.cnn_conv_channels), kernel_size=args.cnn_kernel_size, fc_dims=tuple(args.cnn_fc_dims), n_harmonics=n_harmonics, nx=nx, n_continuous=n_continuous, n_wavelengths=n_wavelengths, n_materials=N_MATERIALS, embed_dim=8)
print(f"SpatialCNN: {count_parameters(spatial):,}")

skip = SkipCNN(conv_channels=tuple(args.skipcnn_conv_channels), kernel_size=args.skipcnn_kernel_size, fc_dims=tuple(args.skipcnn_fc_dims), n_harmonics=n_harmonics, nx=nx, n_continuous=n_continuous, n_wavelengths=n_wavelengths, n_materials=N_MATERIALS, embed_dim=8)
print(f"SkipCNN: {count_parameters(skip):,}")

siren = SIREN(conv_channels=tuple(args.siren_conv_channels), kernel_size=args.siren_kernel_size, siren_hidden=tuple(args.siren_fc_dims), n_harmonics=n_harmonics, nx=nx, n_continuous=n_continuous, n_wavelengths=n_wavelengths, n_materials=N_MATERIALS, embed_dim=8)
print(f"SIREN: {count_parameters(siren):,}")

transformer = TransformerForward(d_model=args.tf_d_model, nhead=args.tf_nhead, dim_feedforward=args.tf_dim_feedforward, num_layers=args.tf_num_layers, n_harmonics=n_harmonics, nx=nx, n_continuous=n_continuous, n_wavelengths=n_wavelengths, n_materials=N_MATERIALS, embed_dim=8)
print(f"TransformerForward: {count_parameters(transformer):,}")

print("\n--- Inverse Models ---")
# Tandem
tandem_decoder = InverseDecoder(latent_dim=0, conv_channels=tuple(args.inv_conv_channels), kernel_size=args.inv_kernel_size, fc_dims=tuple(args.inv_fc_dims), n_geometry=12, n_wavelengths=n_wavelengths, n_materials=N_MATERIALS)
print(f"TandemNetwork (InverseDecoder only): {count_parameters(tandem_decoder):,}")

# Generative Tandem
gen_tandem_decoder = InverseDecoder(latent_dim=32, conv_channels=tuple(args.inv_conv_channels), kernel_size=args.inv_kernel_size, fc_dims=tuple(args.inv_fc_dims), n_geometry=12, n_wavelengths=n_wavelengths, n_materials=N_MATERIALS)
print(f"GenerativeTandemNetwork (InverseDecoder): {count_parameters(gen_tandem_decoder):,}")
gen_tandem = GenerativeTandemNetwork(forward_model=skip, inverse_decoder=gen_tandem_decoder, latent_dim=32)
print(f"GenerativeTandemNetwork (Total Trainable Inverse): {count_parameters(gen_tandem_decoder):,}")

# CVAE
geo_enc = GeometryEncoder(latent_dim=64, conv_channels=tuple(args.cvae_geo_enc_conv), kernel_size=args.cvae_geo_enc_kernel, fc_dims=tuple(args.cvae_geo_enc_fc), n_harmonics=n_harmonics, nx=nx, n_continuous=n_continuous, n_materials=N_MATERIALS, embed_dim=8)
geo_dec = GeometryDecoder(latent_dim=64, hidden_dims=tuple(args.cvae_geo_dec_fc), n_geometry=12)
spec_enc = SpectrumEncoder(latent_dim=64, conv_channels=tuple(args.cvae_spec_enc_conv), kernel_size=args.cvae_spec_enc_kernel, fc_dims=tuple(args.cvae_spec_enc_fc), n_wavelengths=n_wavelengths)
print(f"ContrastiveVAE (GeometryEncoder): {count_parameters(geo_enc):,}")
print(f"ContrastiveVAE (GeometryDecoder): {count_parameters(geo_dec):,}")
print(f"ContrastiveVAE (SpectrumEncoder): {count_parameters(spec_enc):,}")
print(f"ContrastiveVAE (Total Trainable): {count_parameters(geo_enc) + count_parameters(geo_dec) + count_parameters(spec_enc):,}")
