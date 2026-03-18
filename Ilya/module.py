from typing import Literal

import torch
import torchode as to
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam
from torchmetrics import Accuracy

from .jacde import JaCDE
from .matde import MatCDE
from .nat_cub_spline import fit_cubic_spline


class LitModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        vf: Literal["jac", "mat"],
    ):
        super().__init__()
        match vf:
            case "jac":
                vf = JaCDE(input_dim, hidden_dim)
            case "mat":
                vf = MatCDE(input_dim, hidden_dim)
            case _:
                raise ValueError(f"Unknown vector field type: {vf}")

        term = to.ODETerm(vf, with_args=True)
        stepper = to.Dopri5(term)
        controller = to.IntegralController(1e-3, 1e-3, term=term)
        self.solver = to.AutoDiffAdjoint(
            stepper,
            controller,
            backprop_through_step_size_control=False,
        )

        self.head = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.CrossEntropyLoss()
        metric = Accuracy(task="multiclass", num_classes=output_dim)
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        t = t.expand(x.shape[0], x.shape[1])

        coeffs = fit_cubic_spline(t, x)
        dcoeffs = (
            coeffs.roll(-1, 1) * coeffs.new_tensor([1, 2, 3, 0])[None, :, None, None]
        )
        y0 = torch.zeros(x.shape[0], self.hidden_dim, device=self.device, dtype=x.dtype)

        ivp = to.InitialValueProblem(y0, t_start=t[:, 0], t_end=t[:, -1])  # type: ignore
        solution: to.Solution = self.solver.solve(ivp, args=(coeffs, dcoeffs, t))
        preds: Tensor = self.head(solution.ys[:, -1])
        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        self.val_metric(preds.argmax(dim=-1), y)
        self.log("val_metric", self.val_metric, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        self.test_metric(preds.argmax(dim=-1), y)
        self.log("test_metric", self.test_metric)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
