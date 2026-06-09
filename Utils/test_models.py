"""
Example: End-to-end smoke test for all surrogate model architectures.
=====================================================================

This script validates that all five models (ForwardMLP, SpatialCNN,
TandemNetwork, GenerativeTandemNetwork, ContrastiveVAE) can be
instantiated, run forward passes with correct shapes, and that gradients
flow correctly through differentiable Gumbel-Softmax material selection.

Run with:
    cd <project_root>
    uv run python -m Utils.test_models
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from Utils.models import (
    ForwardMLP,
    SpatialCNN,
    InverseDecoder,
    TandemNetwork,
    GenerativeTandemNetwork,
    GeometryEncoder,
    GeometryDecoder,
    SpectrumEncoder,
    ContrastiveVAE,
    Snake,
    polar_to_cartesian,
    N_MATERIALS,
    MATERIAL_LIBRARY,
)


def test_materials():
    print("=" * 60)
    print("TEST: Material Library")
    assert "Si" in MATERIAL_LIBRARY
    assert "TiO2" in MATERIAL_LIBRARY
    assert "Si3N4" in MATERIAL_LIBRARY
    assert N_MATERIALS == 3
    print(f"  ✓ Grating materials: {MATERIAL_LIBRARY}")
    assert "Ag" not in MATERIAL_LIBRARY, "Ag should NOT be in grating materials!"
    print("  ✓ Ag correctly excluded from grating library.")


def test_snake_activation():
    print("=" * 60)
    print("TEST: Snake Activation")
    snake = Snake(in_features=64)
    x = torch.randn(8, 64)
    y = snake(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    # Check gradient flows through learnable 'a'
    loss = y.sum()
    loss.backward()
    assert snake.a.grad is not None, "No gradient on Snake.a"
    print("  ✓ Shape correct, gradients flow through learnable frequency 'a'.")


def test_polar_to_cartesian():
    print("=" * 60)
    print("TEST: polar_to_cartesian")
    B, N = 4, 5
    params = torch.randn(B, N, 2)
    out = polar_to_cartesian(params)
    assert out.shape == (B, 2 * N), f"Shape mismatch: {out.shape}"
    print(f"  ✓ Input (B={B}, N={N}, 2) → Output {out.shape}")


def test_forward_mlp():
    print("=" * 60)
    print("TEST: ForwardMLP")
    B, N_harmonics, N_wl = 16, 5, 161

    model = ForwardMLP(
        n_continuous=2 * N_harmonics + 1,  # cartesian + h
        n_wavelengths=N_wl,
        n_materials=N_MATERIALS,
        embed_dim=8,
        hidden_dims=(128, 256, 128),
        activation="snake",
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Test with integer material IDs
    geo = torch.randn(B, 2 * N_harmonics + 1)
    mat_id = torch.randint(0, N_MATERIALS, (B,))
    out = model(geo, mat_id)
    assert out.shape == (B, N_wl), f"Shape: {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "Output not in [0,1]"
    print(f"  ✓ Integer material ID path: output shape {out.shape}, bounded [0,1]")

    # Test with one-hot material (differentiable path)
    mat_oh = torch.zeros(B, N_MATERIALS)
    mat_oh[torch.arange(B), torch.randint(0, N_MATERIALS, (B,))] = 1.0
    mat_oh.requires_grad_(True)
    out2 = model(geo, mat_oh)
    loss = out2.sum()
    loss.backward()
    assert mat_oh.grad is not None, "No gradient through one-hot path"
    print(f"  ✓ One-hot material path: gradients flow correctly.")


def test_spatial_cnn():
    print("=" * 60)
    print("TEST: SpatialCNN")
    B, N_harmonics, N_wl = 16, 5, 161

    model = SpatialCNN(
        n_harmonics=N_harmonics,
        n_wavelengths=N_wl,
        n_pixels=128,
        conv_channels=(32, 64),
        fc_dims=(128,),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    params_x = torch.rand(B, N_harmonics, 2) * 5  # amps and phases
    h = torch.rand(B, 1) * 4500 + 500
    mat_id = torch.randint(0, N_MATERIALS, (B,))

    out = model(params_x, h, mat_id)
    assert out.shape == (B, N_wl), f"Shape: {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "Output not in [0,1]"
    print(f"  ✓ Output shape {out.shape}, bounded [0,1]")

    # Test profile construction
    profile = model._build_profile(params_x)
    assert profile.shape == (B, 1, 128), f"Profile shape: {profile.shape}"
    print(f"  ✓ Grating profile shape: {profile.shape}")


def test_inverse_decoder():
    print("=" * 60)
    print("TEST: InverseDecoder")
    B, N_wl, N_geo = 16, 161, 11

    decoder = InverseDecoder(
        n_wavelengths=N_wl,
        n_geometry=N_geo,
        n_materials=N_MATERIALS,
        latent_dim=0,
        hidden_dims=(128, 128),
    )

    target = torch.rand(B, N_wl)
    pred_geo, mat_oh, mat_logits = decoder(target, tau=1.0)

    assert pred_geo.shape == (B, N_geo), f"Geometry shape: {pred_geo.shape}"
    assert mat_oh.shape == (B, N_MATERIALS), f"Material shape: {mat_oh.shape}"
    # Check bounds are strictly normalized [0,1]
    assert (pred_geo >= -1e-5).all(), "Geometry below 0"
    assert (pred_geo <= 1 + 1e-5).all(), "Geometry above 1"
    # Check one-hot
    assert torch.allclose(mat_oh.sum(dim=-1), torch.ones(B)), "Not one-hot"
    print(f"  ✓ Geometry: {pred_geo.shape}, bounded [0, 1]")
    print(f"  ✓ Material: {mat_oh.shape}, valid one-hot vectors ({N_MATERIALS} materials)")


def test_tandem_network():
    print("=" * 60)
    print("TEST: TandemNetwork")
    B, N_wl, N_geo = 16, 161, 11

    # Build forward model
    forward = ForwardMLP(
        n_continuous=N_geo,
        n_wavelengths=N_wl,
        hidden_dims=(128, 128),
        activation="gelu",
    )

    decoder = InverseDecoder(
        n_wavelengths=N_wl,
        n_geometry=N_geo,
        latent_dim=0,
        hidden_dims=(128, 128),
    )

    tandem = TandemNetwork(inverse_decoder=decoder, forward_model=forward)

    # Check forward model is frozen
    for p in tandem.forward_model.parameters():
        assert not p.requires_grad, "Forward model should be frozen!"

    target = torch.rand(B, N_wl)
    out = tandem(target, tau=1.0)

    assert out["predicted_curve"].shape == (B, N_wl)
    assert out["pred_geometry"].shape == (B, N_geo)
    assert out["material_onehot"].shape == (B, N_MATERIALS)

    # Test gradient flow: only decoder should have gradients
    loss = torch.nn.functional.mse_loss(out["predicted_curve"], target)
    loss.backward()

    has_decoder_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in tandem.inverse_decoder.parameters()
    )
    has_forward_grad = any(
        p.grad is not None for p in tandem.forward_model.parameters()
    )
    assert has_decoder_grad, "Decoder should receive gradients!"
    assert not has_forward_grad, "Forward model should NOT receive gradients!"
    print(f"  ✓ Predicted curve: {out['predicted_curve'].shape}")
    print(f"  ✓ Gradients flow only through InverseDecoder (forward frozen).")
    print(f"  ✓ Loss = {loss.item():.6f}")


def test_generative_tandem():
    print("=" * 60)
    print("TEST: GenerativeTandemNetwork")
    B, N_wl, N_geo, latent_dim = 16, 161, 11, 32

    forward = ForwardMLP(
        n_continuous=N_geo,
        n_wavelengths=N_wl,
        hidden_dims=(128, 128),
        activation="gelu",
    )

    decoder = InverseDecoder(
        n_wavelengths=N_wl,
        n_geometry=N_geo,
        latent_dim=latent_dim,
        hidden_dims=(128, 128),
    )

    gen_tandem = GenerativeTandemNetwork(
        inverse_decoder=decoder,
        forward_model=forward,
        latent_dim=latent_dim,
    )

    # Normal forward
    target = torch.rand(B, N_wl)
    out = gen_tandem(target, tau=1.0)
    assert out["predicted_curve"].shape == (B, N_wl)
    assert out["z"].shape == (B, latent_dim)
    print(f"  ✓ Forward pass: curve={out['predicted_curve'].shape}, z={out['z'].shape}")

    # Test diverse sampling
    single_target = torch.rand(1, N_wl)
    diverse = gen_tandem.sample_diverse_designs(single_target, n_samples=8, tau=0.1)
    assert diverse["pred_geometry"].shape == (8, N_geo)
    assert diverse["material_onehot"].shape == (8, N_MATERIALS)
    geo_std = diverse["pred_geometry"].std(dim=0).mean()
    print(f"  ✓ Diverse sampling: {diverse['pred_geometry'].shape}")
    print(f"    Mean std across geometry dims: {geo_std:.4f} (should be > 0)")

    # Gradient test
    loss = torch.nn.functional.mse_loss(out["predicted_curve"], target)
    loss.backward()
    has_decoder_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in gen_tandem.inverse_decoder.parameters()
    )
    assert has_decoder_grad, "Decoder should receive gradients!"
    print(f"  ✓ Gradients verified through generative tandem pipeline.")


def test_contrastive_vae():
    print("=" * 60)
    print("TEST: ContrastiveVAE")
    B, N_wl, N_geo, latent_dim = 16, 161, 11, 64
    margin_radius = 1.0

    geo_enc = GeometryEncoder(
        n_continuous=N_geo, latent_dim=latent_dim, hidden_dims=(128, 128)
    )
    geo_dec = GeometryDecoder(
        latent_dim=latent_dim, n_geometry=N_geo, hidden_dims=(128, 128)
    )
    spec_enc = SpectrumEncoder(
        n_wavelengths=N_wl, latent_dim=latent_dim, hidden_dims=(128, 128)
    )

    cvae = ContrastiveVAE(
        geometry_encoder=geo_enc,
        geometry_decoder=geo_dec,
        spectrum_encoder=spec_enc,
        margin_radius=margin_radius,
        beta=1e-3,
        gamma=1.0,
    )

    n_params = sum(p.numel() for p in cvae.parameters())
    print(f"  Parameters: {n_params:,}")

    # --- Forward pass ---
    geometry = torch.randn(B, N_geo)
    material_id = torch.randint(0, N_MATERIALS, (B,))
    target_curve = torch.rand(B, N_wl)

    out = cvae(geometry, material_id, target_curve, tau=1.0)

    assert out["z_x"].shape == (B, latent_dim), f"z_x shape: {out['z_x'].shape}"
    assert out["z_y"].shape == (B, latent_dim), f"z_y shape: {out['z_y'].shape}"
    assert out["mu_x"].shape == (B, latent_dim)
    assert out["logvar_x"].shape == (B, latent_dim)
    assert out["recon_geometry"].shape == (B, N_geo)
    assert out["recon_material_onehot"].shape == (B, N_MATERIALS)
    assert out["recon_material_logits"].shape == (B, N_MATERIALS)
    print(f"  ✓ Forward pass: z_x={out['z_x'].shape}, z_y={out['z_y'].shape}")
    print(f"    recon_geometry={out['recon_geometry'].shape}, "
          f"recon_material={out['recon_material_onehot'].shape}")

    # --- Check geometry bounds ---
    assert (out["recon_geometry"] >= -1e-5).all(), "Below 0"
    assert (out["recon_geometry"] <= 1 + 1e-5).all(), "Above 1"
    print(f"  ✓ Reconstructed geometry bounded [0, 1]")

    # --- Check one-hot material ---
    assert torch.allclose(
        out["recon_material_onehot"].sum(dim=-1), torch.ones(B)
    ), "Material not one-hot"
    print(f"  ✓ Reconstructed material: valid one-hot vectors.")

    # --- Loss computation ---
    losses = cvae.compute_loss(out, geometry, material_id)
    assert "loss" in losses
    assert "loss_recon" in losses
    assert "loss_mat_ce" in losses
    assert "loss_kl" in losses
    assert "loss_margin" in losses
    print(f"  ✓ Loss components:")
    print(f"    total={losses['loss'].item():.4f}  "
          f"recon={losses['loss_recon'].item():.4f}  "
          f"mat_ce={losses['loss_mat_ce'].item():.4f}  "
          f"kl={losses['loss_kl'].item():.4f}  "
          f"margin={losses['loss_margin'].item():.4f}")

    # --- Gradient flow test ---
    losses["loss"].backward()
    has_geo_enc_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in cvae.geometry_encoder.parameters()
    )
    has_geo_dec_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in cvae.geometry_decoder.parameters()
    )
    has_spec_enc_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in cvae.spectrum_encoder.parameters()
    )
    assert has_geo_enc_grad, "Geometry encoder should receive gradients!"
    assert has_geo_dec_grad, "Geometry decoder should receive gradients!"
    assert has_spec_enc_grad, "Spectrum encoder should receive gradients!"
    print(f"  ✓ Gradients flow through all three sub-networks.")

    # --- Margin loss correctness ---
    # When z_x and z_y are very close, margin loss should be 0
    z_close = out["z_y"].clone()
    loss_close = ContrastiveVAE.margin_loss(z_close, out["z_y"], margin_radius)
    assert loss_close.item() < 1e-6, f"Margin loss should be ~0 when z_x=z_y, got {loss_close.item()}"

    # When z_x is far from z_y, margin loss should be > 0
    z_far = out["z_y"] + torch.ones_like(out["z_y"]) * 10.0
    loss_far = ContrastiveVAE.margin_loss(z_far, out["z_y"], margin_radius)
    assert loss_far.item() > 0, f"Margin loss should be > 0 when far apart, got {loss_far.item()}"
    print(f"  ✓ Margin loss semantics verified (0 when close, >0 when far).")

    # --- Inverse design: generate diverse proposals ---
    single_curve = torch.rand(1, N_wl)
    designs = cvae.generate(single_curve, n_samples=8, tau=0.1)
    assert designs["pred_geometry"].shape == (8, N_geo)
    assert designs["material_onehot"].shape == (8, N_MATERIALS)
    assert designs["z_samples"].shape == (8, latent_dim)
    assert designs["z_y"].shape == (1, latent_dim)

    # Check that samples are within margin radius of z_y
    dists = torch.norm(
        designs["z_samples"] - designs["z_y"].expand(8, -1), p=2, dim=-1
    )
    assert (dists <= margin_radius + 1e-5).all(), (
        f"Some samples outside margin radius! max dist={dists.max():.4f}"
    )
    geo_std = designs["pred_geometry"].std(dim=0).mean()
    print(f"  ✓ generate() produced {designs['pred_geometry'].shape[0]} candidates")
    print(f"    All within margin_radius={margin_radius} "
          f"(max dist={dists.max():.4f})")
    print(f"    Mean geometry std: {geo_std:.4f}")


if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  Surrogate Model Architecture Smoke Tests")
    print("═" * 60)

    test_materials()
    test_snake_activation()
    test_polar_to_cartesian()
    test_forward_mlp()
    test_spatial_cnn()
    test_inverse_decoder()
    test_tandem_network()
    test_generative_tandem()
    test_contrastive_vae()

    print("\n" + "═" * 60)
    print("  ALL TESTS PASSED ✓")
    print("═" * 60 + "\n")
