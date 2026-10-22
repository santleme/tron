"""Torch model factory shared by Policy-Value training and CPU inference."""
from __future__ import annotations


def model_class(torch):
    """Create the viewer-compatible channel-last flattened v2 policy-value MLP."""
    class PolicyValue(torch.nn.Module):
        def __init__(self, width: int, height: int, hidden_size: int):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(width * height * 12, hidden_size), torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, hidden_size), torch.nn.Tanh(),
            )
            self.policy = torch.nn.Linear(hidden_size, 4)
            self.value = torch.nn.Linear(hidden_size, 1)

        def forward(self, observation):
            features = self.encoder(observation)
            logits = self.policy(features)
            head_plane = observation[..., 0]
            head = head_plane.unsqueeze(-1)
            heading_planes = observation[..., 4:8]
            heading = (heading_planes * head).sum(dim=(1, 2)).argmax(dim=-1)
            reverse = torch.nn.functional.one_hot(
                (heading + 2) % 4, num_classes=4).to(torch.bool)
            height, width = observation.shape[1:3]
            x = (head_plane * torch.arange(
                width, device=observation.device, dtype=observation.dtype
            ).view(1, 1, width)).sum((1, 2)).long()
            y = (head_plane * torch.arange(
                height, device=observation.device, dtype=observation.dtype
            ).view(1, height, 1)).sum((1, 2)).long()
            occupied = (observation[..., 2] + observation[..., 3]).reshape(
                -1, height * width) > 0
            offsets = torch.tensor((-width, 1, width, -1), device=observation.device)
            destination = y[:, None] * width + x[:, None] + offsets[None, :]
            in_bounds = torch.stack(
                (y > 0, x < width - 1, y < height - 1, x > 0), dim=1)
            destination_occupied = torch.gather(
                occupied, 1, destination.clamp(0, height * width - 1))
            unsafe = ~in_bounds | destination_occupied
            has_safe_move = (~unsafe & ~reverse).any(dim=1, keepdim=True)
            illegal = reverse | (unsafe & has_safe_move)
            logits = logits.masked_fill(illegal, torch.finfo(logits.dtype).min)
            return logits, self.value(features).squeeze(-1)

    return PolicyValue
