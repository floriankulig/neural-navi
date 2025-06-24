# src/utils/nan_debug.py
import torch
import logging


def check_tensor_health(tensor: torch.Tensor, name: str, batch_indices: list = None):
    """Check tensor for NaN/Inf and report which batch elements are affected."""
    if tensor is None:
        return True, []

    has_nan = torch.isnan(tensor)
    has_inf = torch.isinf(tensor)

    if has_nan.any():
        if len(tensor.shape) >= 2:  # Has batch dimension
            nan_batch_indices = torch.where(
                has_nan.any(dim=tuple(range(1, len(tensor.shape))))
            )[0].tolist()
        else:
            nan_batch_indices = [0] if has_nan.any() else []

        logging.error(f"🚨 NaN in {name}:")
        logging.error(f"   Total NaN elements: {has_nan.sum().item()}")
        logging.error(f"   Affected batch indices: {nan_batch_indices}")

        if batch_indices is not None:
            common_indices = set(nan_batch_indices) & set(batch_indices)
            if common_indices:
                logging.error(
                    f"   Previously affected batch indices: {list(common_indices)}"
                )

        return False, nan_batch_indices

    if has_inf.any():
        logging.warning(f"⚠️ Inf in {name}: {has_inf.sum().item()} elements")

    return True, []


def debug_model_forward(model, telemetry, detections, mask):
    """Debug model forward pass step by step."""
    logging.info("🔍 Starting detailed model forward debugging...")

    nan_batch_indices = []

    # Check inputs first
    check_tensor_health(telemetry, "input_telemetry", nan_batch_indices)
    check_tensor_health(detections, "input_detections", nan_batch_indices)
    check_tensor_health(mask.float(), "input_mask", nan_batch_indices)

    # Hook into model components
    def create_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, dict):
                for key, value in output.items():
                    is_healthy, new_nan_indices = check_tensor_health(
                        value, f"{name}_{key}", nan_batch_indices
                    )
                    if not is_healthy:
                        nan_batch_indices.extend(new_nan_indices)
            elif isinstance(output, (tuple, list)):
                for i, item in enumerate(output):
                    is_healthy, new_nan_indices = check_tensor_health(
                        item, f"{name}_output_{i}", nan_batch_indices
                    )
                    if not is_healthy:
                        nan_batch_indices.extend(new_nan_indices)
            else:
                is_healthy, new_nan_indices = check_tensor_health(
                    output, f"{name}_output", nan_batch_indices
                )
                if not is_healthy:
                    nan_batch_indices.extend(new_nan_indices)

        return hook_fn

    # Register hooks
    hooks = []
    hooks.append(
        model.input_encoder.register_forward_hook(create_hook("input_encoder"))
    )
    hooks.append(
        model.fusion_module.register_forward_hook(create_hook("fusion_module"))
    )
    hooks.append(
        model.output_decoder.register_forward_hook(create_hook("output_decoder"))
    )

    try:
        # Forward pass
        with torch.no_grad():  # Prevent gradient computation during debugging
            output = model(telemetry, detections, mask)

        # Check final output
        check_tensor_health(output["coast_1s"], "final_output", nan_batch_indices)

        return output, list(set(nan_batch_indices))

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()


def debug_cross_modal_attention(
    tel_features, det_features, det_mask, module_name="fusion"
):
    """Debug cross-modal attention step by step."""
    print(f"\n🔍 Debugging CrossModalAttention in {module_name}:")

    batch_size, seq_len, embedding_dim = tel_features.shape
    _, _, max_dets, _ = det_features.shape

    print(
        f"   Input shapes: tel={tel_features.shape}, det={det_features.shape}, mask={det_mask.shape}"
    )
    print(
        f"   Tel range: [{tel_features.min().item():.4f}, {tel_features.max().item():.4f}]"
    )
    print(
        f"   Det range: [{det_features.min().item():.4f}, {det_features.max().item():.4f}]"
    )

    # Check for NaN in inputs
    if torch.isnan(tel_features).any():
        print(f"   🚨 NaN in telemetry features!")
        return False
    if torch.isnan(det_features).any():
        print(f"   🚨 NaN in detection features!")
        return False

    problematic_timesteps = []

    # Process each timestep (wie in deinem CrossModalAttentionFusion)
    for t in range(seq_len):
        tel_t = tel_features[:, t].unsqueeze(1)  # [batch_size, 1, embedding_dim]
        det_t = det_features[:, t]  # [batch_size, max_detections, embedding_dim]
        mask_t = det_mask[:, t]  # [batch_size, max_detections]

        print(f"\n   Timestep {t}:")
        print(f"     Tel_t range: [{tel_t.min().item():.4f}, {tel_t.max().item():.4f}]")
        print(f"     Det_t range: [{det_t.min().item():.4f}, {det_t.max().item():.4f}]")

        # Check attention mask
        attn_mask = ~mask_t  # True = ignore
        valid_ratio = (~attn_mask).float().mean()
        print(f"     Valid detections ratio: {valid_ratio:.3f}")

        # Manual attention computation to find the problem
        # Q = tel_t, K = det_t, V = det_t
        d_k = tel_t.size(-1)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(tel_t, det_t.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32, device=tel_t.device)
        )
        print(
            f"     Raw scores range: [{scores.min().item():.4f}, {scores.max().item():.4f}]"
        )

        if torch.isnan(scores).any():
            print(f"     🚨 NaN in raw attention scores at timestep {t}!")
            problematic_timesteps.append(t)
            continue

        if torch.isinf(scores).any():
            print(f"     🚨 Inf in raw attention scores at timestep {t}!")
            problematic_timesteps.append(t)
            continue

        # Apply mask (set ignored positions to very negative value)
        if attn_mask.any():
            scores = scores.masked_fill(
                attn_mask.unsqueeze(1), -1e9
            )  # Use -1e9 instead of -inf
            print(
                f"     After masking range: [{scores.min().item():.4f}, {scores.max().item():.4f}]"
            )

        # Apply softmax
        try:
            attn_weights = torch.softmax(scores, dim=-1)
            print(
                f"     Attention weights range: [{attn_weights.min().item():.4f}, {attn_weights.max().item():.4f}]"
            )

            if torch.isnan(attn_weights).any():
                print(
                    f"     🚨 NaN in attention weights after softmax at timestep {t}!"
                )
                problematic_timesteps.append(t)

                # Check if all values were masked
                if attn_mask.all(dim=1).any():
                    completely_masked_batch = torch.where(attn_mask.all(dim=1))[0]
                    print(
                        f"     🚨 Completely masked batch elements: {completely_masked_batch.tolist()}"
                    )

                continue

            # Compute output: attention_weights @ V
            output = torch.matmul(attn_weights, det_t)
            print(
                f"     Output range: [{output.min().item():.4f}, {output.max().item():.4f}]"
            )

            if torch.isnan(output).any():
                print(f"     🚨 NaN in attention output at timestep {t}!")
                problematic_timesteps.append(t)

        except Exception as e:
            print(f"     ❌ Exception during softmax/matmul at timestep {t}: {e}")
            problematic_timesteps.append(t)

    if problematic_timesteps:
        print(f"\n🚨 Problematic timesteps: {problematic_timesteps}")
        return False

    return True


def debug_fusion_module_detailed(fusion_module, encoded_inputs):
    """Debug the entire fusion module step by step."""
    tel_features = encoded_inputs["telemetry_features"]
    det_features = encoded_inputs["detection_features"]
    det_mask = encoded_inputs["detection_mask"]

    print(f"\n🔧 Debugging FusionModule: {type(fusion_module).__name__}")

    # Debug cross-modal attention if it's CrossModalAttentionFusion
    if hasattr(fusion_module, "tel_to_det_attention"):
        if not debug_cross_modal_attention(tel_features, det_features, det_mask):
            return False

    return True


# src/utils/fusion_debug.py
import torch
import math


def debug_cross_modal_attention_step(tel_t, det_t, mask_t, timestep):
    """Debug one timestep of cross-modal attention."""
    print(f"\n   🔍 Timestep {timestep}:")

    # Check inputs
    if torch.isnan(tel_t).any() or torch.isnan(det_t).any():
        print(f"     🚨 NaN in inputs!")
        return False

    # Attention computation
    d_k = tel_t.size(-1)
    attn_mask = ~mask_t  # True = ignore

    # Compute scores
    scores = torch.matmul(tel_t, det_t.transpose(-2, -1)) / math.sqrt(d_k)
    print(
        f"     Raw scores range: [{scores.min().item():.4f}, {scores.max().item():.4f}]"
    )

    if torch.isnan(scores).any():
        print(f"     🚨 NaN in raw scores!")
        return False

    # Apply mask
    scores = scores.masked_fill(attn_mask.unsqueeze(1), -1e9)

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    print(
        f"     Attention weights range: [{attn_weights.min().item():.4f}, {attn_weights.max().item():.4f}]"
    )

    if torch.isnan(attn_weights).any():
        print(f"     🚨 NaN in attention weights!")
        # Check if completely masked
        if attn_mask.all(dim=1).any():
            completely_masked = torch.where(attn_mask.all(dim=1))[0]
            print(
                f"     Completely masked batch elements: {completely_masked.tolist()}"
            )
        return False

    # Output
    output = torch.matmul(attn_weights, det_t)
    if torch.isnan(output).any():
        print(f"     🚨 NaN in attention output!")
        return False

    return True
