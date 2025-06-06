import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class NaNDebugger:
    """Utility class to track and debug NaN values in neural networks."""

    def __init__(self, model: nn.Module, logger=None):
        self.model = model
        self.logger = logger or print
        self.hooks = []
        self.nan_locations = []

    def check_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        """Check if tensor contains NaN or Inf values."""
        if tensor is None:
            return False

        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        if has_nan or has_inf:
            self.logger(f"⚠️ {name}: NaN={has_nan}, Inf={has_inf}")
            self.logger(f"   Shape: {tensor.shape}")
            self.logger(f"   Device: {tensor.device}")
            self.logger(f"   Dtype: {tensor.dtype}")

            if has_nan:
                nan_mask = torch.isnan(tensor)
                self.logger(f"   NaN count: {nan_mask.sum().item()}")

            if has_inf:
                inf_mask = torch.isinf(tensor)
                self.logger(f"   Inf count: {inf_mask.sum().item()}")

            # Sample values
            flat = tensor.flatten()
            valid_mask = ~(torch.isnan(flat) | torch.isinf(flat))
            if valid_mask.any():
                valid_values = flat[valid_mask]
                self.logger(
                    f"   Valid range: [{valid_values.min():.3f}, {valid_values.max():.3f}]"
                )

            return True
        return False

    def register_hooks(self):
        """Register forward hooks to track NaN propagation."""

        def make_hook(name):
            def hook(module, input, output):
                # Check inputs
                if isinstance(input, tuple):
                    for i, inp in enumerate(input):
                        if isinstance(inp, torch.Tensor):
                            if self.check_tensor(inp, f"{name}.input[{i}]"):
                                self.nan_locations.append(f"{name}.input[{i}]")

                # Check outputs
                if isinstance(output, tuple):
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            if self.check_tensor(out, f"{name}.output[{i}]"):
                                self.nan_locations.append(f"{name}.output[{i}]")
                elif isinstance(output, torch.Tensor):
                    if self.check_tensor(output, f"{name}.output"):
                        self.nan_locations.append(f"{name}.output")

                # Check module parameters
                for param_name, param in module.named_parameters():
                    if self.check_tensor(param, f"{name}.{param_name}"):
                        self.nan_locations.append(f"{name}.{param_name}")

                    if param.grad is not None:
                        if self.check_tensor(param.grad, f"{name}.{param_name}.grad"):
                            self.nan_locations.append(f"{name}.{param_name}.grad")

            return hook

        # Register hooks for all modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                handle = module.register_forward_hook(make_hook(name))
                self.hooks.append(handle)

        self.logger(f"✅ Registered {len(self.hooks)} debug hooks")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def check_data_batch(self, batch: Dict[str, torch.Tensor]) -> bool:
        """Check a data batch for NaN/Inf values."""
        has_issues = False

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if self.check_tensor(value, f"batch.{key}"):
                    has_issues = True
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        if self.check_tensor(sub_value, f"batch.{key}.{sub_key}"):
                            has_issues = True

        return has_issues

    def summarize(self):
        """Print summary of NaN locations."""
        if self.nan_locations:
            self.logger("\n🔍 NaN/Inf Detection Summary:")
            unique_locations = list(set(self.nan_locations))
            for loc in sorted(unique_locations):
                count = self.nan_locations.count(loc)
                self.logger(f"   - {loc}: {count} occurrences")
        else:
            self.logger("✅ No NaN/Inf values detected")
