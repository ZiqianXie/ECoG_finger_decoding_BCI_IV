import torch

from scripts.train_event_grouped_lars_lstm_nested import kinematic_hurdle_loss


def loss_for(position: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    moving = target >= 0.08
    state_logit = torch.where(
        moving, torch.full_like(target, 5.0), torch.full_like(target, -5.0)
    )
    velocity = torch.zeros_like(target)
    velocity[:, 1:] = torch.diff(position, dim=1) / 0.04
    loss, _ = kinematic_hurdle_loss(
        position,
        state_logit,
        velocity,
        target,
        movement_threshold=0.08,
        rest_threshold=0.04,
        state_weight=0.2,
        velocity_weight=0.2,
        consistency_weight=0.05,
        curvature_weight=0.02,
    )
    return loss


def test_matching_kinematics_beats_flat_prediction() -> None:
    target = torch.tensor([[0.0, 0.0, 0.2, 0.7, 0.3, 0.0]])
    exact = loss_for(target.clone(), target)
    flat = loss_for(torch.zeros_like(target), target)
    assert exact < flat


def test_kinematic_hurdle_loss_has_finite_gradients() -> None:
    target = torch.tensor([[0.0, 0.03, 0.06, 0.2, 0.4, 0.1, 0.0]])
    position = torch.full_like(target, 0.1, requires_grad=True)
    state_logit = torch.zeros_like(target, requires_grad=True)
    velocity = torch.zeros_like(target, requires_grad=True)
    loss, parts = kinematic_hurdle_loss(
        position,
        state_logit,
        velocity,
        target,
        movement_threshold=0.08,
        rest_threshold=0.04,
        state_weight=0.2,
        velocity_weight=0.2,
        consistency_weight=0.05,
        curvature_weight=0.02,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    assert position.grad is not None and torch.isfinite(position.grad).all()
    assert state_logit.grad is not None and torch.isfinite(state_logit.grad).all()
    assert velocity.grad is not None and torch.isfinite(velocity.grad).all()
