import torch
import torchode as to
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy
from src_torch.nat_cub_spline import fit_cubic_spline

class CDELitModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        if cfg.model == "torch_baseline":
            from src_torch.models_baseline import BaselineCDE 
            vf = BaselineCDE(cfg.input_dim, cfg.hidden_dim, cfg.cell)
        elif cfg.model == "torch_manual":
            from src_torch.models_manual import JaCDEManual
           
            vf = JaCDEManual(cfg.input_dim, cfg.hidden_dim, cfg.cell, cfg.k_terms, cfg.activation)
        elif cfg.model == "torch_auto":
            from src_torch.models_auto import JaCDEAutograd
            from src_torch.cells import RNNCell, GRUCell, LSTMCell
           
            if cfg.cell == "rnn": cell = RNNCell(cfg.input_dim, cfg.hidden_dim, activation=cfg.activation)
            elif cfg.cell == "gru": cell = GRUCell(cfg.input_dim, cfg.hidden_dim)
            elif cfg.cell == "lstm": cell = LSTMCell(cfg.input_dim, cfg.hidden_dim)
            vf = JaCDEAutograd(cell, cfg.k_terms)

        term = to.ODETerm(vf, with_args=True)
        self.solver = to.AutoDiffAdjoint(to.Dopri5(term), to.IntegralController(1e-3, 1e-3, term=term), backprop_through_step_size_control=False)

        self.head = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.loss = nn.CrossEntropyLoss()
        
        self.val_acc = Accuracy(task="multiclass", num_classes=cfg.output_dim)
        self.test_acc = Accuracy(task="multiclass", num_classes=cfg.output_dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).expand(x.shape[0], x.shape[1])
        coeffs = fit_cubic_spline(t, x)
        dcoeffs = coeffs.roll(-1, 1) * coeffs.new_tensor([1, 2, 3, 0])[None, :, None, None]
        y0 = torch.zeros(x.shape[0], self.cfg.hidden_dim, device=self.device, dtype=x.dtype)
        
        ivp = to.InitialValueProblem(y0, t_start=t[:, 0], t_end=t[:, -1])
        solution = self.solver.solve(ivp, args=(coeffs, dcoeffs, t))
        return self.head(solution.ys[:, -1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        self.val_acc(preds.argmax(dim=-1), y)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        preds = self.forward(batch[0])
        self.test_acc(preds.argmax(dim=-1), batch[1])
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(), lr=1e-3)