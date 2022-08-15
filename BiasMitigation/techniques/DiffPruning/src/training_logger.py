from pathlib import Path
from typing import Union, Optional
import math
from torch.utils.tensorboard import SummaryWriter


class TrainLogger:
    delta: float = 1e-8
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        logging_step: int,
        logger_name: Optional[str] = None
    ):
        assert logging_step > 0, "logging_step needs to be > 0"
        
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        
        self.writer = SummaryWriter(log_dir / logger_name)
        self.logger_name = logger_name
        self.logging_step = logging_step
        self.reset()

    def validation_loss(self, eval_step: int, result: dict): 
        for name, value in sorted(result.items(), key=lambda x: x[0]):
            self.writer.add_scalar(f'val/{name}', value, eval_step)

    def step_loss(self, step: int, loss: float, lr: float):
        
        self.logging_loss += loss
        self.steps += 1
        
        if step % self.logging_step == 0:
            logs = {
                "step_learning_rate": lr,
                "step_loss": self.logging_loss / self.steps
            }
            for key, value in logs.items():
                self.writer.add_scalar(f'train/{key}', value, step)
                
            self.logging_loss = 0.
            self.steps = 0
            
    def non_zero_params(self, step, n_p, n_p_zero, n_p_between):
        d = {
            "zero_ratio": n_p_zero / n_p,
            "between_0_1_ratio": n_p_between / n_p
        }
        for k,v in d.items():
            self.writer.add_scalar(f"train/{k}", v, step)


    def is_best(self, result: dict):
        if result["loss"] < self.best_eval_loss + self.delta:
            self.best_eval_loss = result["loss"]
            return True
        
    def reset(self):
        self.steps = 0
        self.logging_loss = 0.
        self.best_eval_loss = math.inf   
            

