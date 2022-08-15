import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.nn.parameter import Parameter
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModel
)

from typing import Union, Callable, List, Dict, Tuple, Optional
from enum import Enum, auto

from techniques.DiffPruning.src.training_logger import TrainLogger
from techniques.DiffPruning.src.diff_param import DiffWeight, DiffWeightFixmask
from techniques.DiffPruning.src.utils import dict_to_device


class ModelState(Enum):
    FINETUNING = auto()
    DIFFPRUNING = auto()
    FIXMASK = auto()


class DiffNetwork(torch.nn.Module):
    
    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device
    
    @property
    def model_type(self) -> str:
        return self.encoder.config.model_type
    
    @property
    def model_name(self) -> str:
        return self.encoder.config._name_or_path
    
    @property
    def total_layers(self) -> int:
        possible_keys = ["num_hidden_layers", "n_layer"]
        for k in possible_keys:
            if k in self.encoder.config.__dict__:
                return getattr(self.encoder.config, k) + 2 # +2 for embedding layer and last layer
        raise Exception("number of layers of pre trained model could not be determined")
    
    @property
    def _parametrized(self) -> bool:
        return (self._model_state == ModelState.DIFFPRUNING or self._model_state == ModelState.FIXMASK)
        
    @staticmethod
    # TODO log ratio could be removed if only symmetric concrete distributions are possible
    def get_log_ratio(concrete_lower: float, concrete_upper: float) -> int:
        # calculate regularization term in objective
        return 0 if (concrete_lower == 0) else math.log(-concrete_lower / concrete_upper)
    
    @staticmethod
    def get_l0_norm_term(alpha: torch.Tensor, log_ratio: float) -> torch.Tensor:
        return torch.sigmoid(alpha - log_ratio).sum()

    
    def get_encoder_base_modules(self, return_names: bool = False):
        if self._parametrized:
            check_fn = lambda m: hasattr(m, "parametrizations")
        else:
            check_fn = lambda m: len(m._parameters)>0
        return [(n,m) if return_names else m for n,m in self.encoder.named_modules() if check_fn(m)]
          
    
    def __init__(self, num_labels, *args, **kwargs): 
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(*args, **kwargs)
        
        emb_dim = self.encoder.embeddings.word_embeddings.embedding_dim
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(.1),
            torch.nn.Linear(emb_dim, num_labels)
        )
        
        self._model_state = ModelState.FINETUNING
            

    def forward(self, **x) -> torch.Tensor: 
        hidden = self.encoder(**x)[0][:,0]
        return self.classifier(hidden)

    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        num_epochs_finetune: int,
        num_epochs_fixmask: int,
        diff_pruning: bool,
        alpha_init: Union[int, float],
        concrete_lower: float,
        concrete_upper: float,
        structured_diff_pruning: bool,
        sparsity_pen: Union[float,list],
        fixmask_pct: float,
        weight_decay: float,
        learning_rate: float,
        learning_rate_alpha: float,
        adam_epsilon: float,
        warmup_steps: int,
        gradient_accumulation_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike]
    ):
        self.global_step = 0
        
        num_epochs_total = num_epochs_finetune + num_epochs_fixmask
        train_steps_finetune = len(train_loader) // gradient_accumulation_steps * num_epochs_finetune
        train_steps_fixmask = len(train_loader) // gradient_accumulation_steps * num_epochs_fixmask
        
        log_ratio = self.get_log_ratio(concrete_lower, concrete_upper)
        
        if diff_pruning:
            self._init_sparsity_pen(sparsity_pen)
            self._add_diff_parametrizations(
                alpha_init,
                concrete_lower,
                concrete_upper,
                structured_diff_pruning  
            )       
        
        
        self._init_optimizer_and_schedule(
            train_steps_finetune,
            learning_rate,
            weight_decay,
            adam_epsilon,
            warmup_steps,
            learning_rate_alpha,
        )
           
        train_str = "Epoch {}, model_state: {}{}"
        str_suffix = lambda result: ", " + ", ".join([f"validation {k}: {v}" for k,v in result.items()])
        result = {}
        
        train_iterator = trange(num_epochs_total, leave=False, position=0)
        for epoch in train_iterator:
            
            train_iterator.set_description(
                train_str.format(epoch, self._model_state, str_suffix(result)), refresh=True
            )
            
            # switch to fixed mask training
            if epoch == num_epochs_finetune:
                self._finetune_to_fixmask(fixmask_pct)
                self._init_optimizer_and_schedule(
                    train_steps_fixmask,
                    learning_rate,
                    weight_decay,
                    adam_epsilon,
                    warmup_steps
                )
            
            self._step(
                train_loader,
                loss_fn,
                logger,
                log_ratio,
                max_grad_norm,
                gradient_accumulation_steps
            )
                        
            result = self.evaluate(
                val_loader, 
                loss_fn,
                metrics
            )

            logger.validation_loss(epoch, result)
            
            # count non zero
            if diff_pruning:
                n_p, n_p_zero, n_p_between = self._count_non_zero_params()            
                logger.non_zero_params(epoch, n_p, n_p_zero, n_p_between)
            
            # if num_epochs_fixmask > 0 only save during fixmask tuning
            if ((num_epochs_fixmask > 0) and (self._model_state==ModelState.FIXMASK)) or (num_epochs_fixmask == 0):
                if logger.is_best(result):
                    self.save_checkpoint(
                        Path(output_dir),
                        concrete_lower,
                        concrete_upper,
                        structured_diff_pruning
                    )
        
        print("Final results after " + train_str.format(epoch, self._model_state, str_suffix(result)))
        return self.encoder, self.classifier

                
    @torch.no_grad()   
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
    ) -> dict: 
        self.eval()

        output_list = []
        val_iterator = tqdm(val_loader, desc="evaluating", leave=False, position=1)
        for batch in val_iterator:

            inputs, labels = batch
            inputs = dict_to_device(inputs, self.device)
            logits = self(**inputs)
            output_list.append((
                logits.cpu(),
                labels
            ))
                    
        p, l = list(zip(*output_list))
        predictions = torch.cat(p, dim=0)
        labels = torch.cat(l, dim=0)
        
        eval_loss = loss_fn(predictions, labels).item()
        
        result = {metric_name: metric(predictions, labels) for metric_name, metric in metrics.items()}
        result["loss"] = eval_loss

        return result


    def get_layer_idx_from_module(self, n: str) -> int:
        # get layer index based on module name
        if self.model_type == "xlnet":
            search_str_emb = "word_embedding"
            search_str_hidden = "layer"
        else:
            search_str_emb = "embeddings"
            search_str_hidden = "encoder.layer"
        
        if search_str_emb in n:
            return 0
        elif search_str_hidden in n:
            return int(n.split(search_str_hidden + ".")[1].split(".")[0]) + 1
        else:
            return self.total_layers - 1
                                               
                    
    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike],
        concrete_lower: Optional[float] = None,
        concrete_upper: Optional[float] = None,
        structured_diff_pruning: Optional[bool] = None
    ) -> None:
        info_dict = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "model_state": self._model_state,
            "encoder_state_dict": self.encoder.state_dict(),
            "classifier_state_dict": self.classifier.state_dict()
        }
        if self._model_state == ModelState.DIFFPRUNING:
            info_dict = {
                **info_dict,
                "concrete_lower": concrete_lower,
                "concrete_upper": concrete_upper,
                "structured_diff_pruning": structured_diff_pruning
            } 
        filename = f"checkpoint-{self.model_name.split('/')[-1]}.pt"
        filepath = Path(output_dir) / filename
        torch.save(info_dict, filepath)
        return filepath


    @staticmethod
    def load_checkpoint(filepath: Union[str, os.PathLike], remove_parametrizations: bool = False) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=torch.device('cpu'))
            
        diff_network = DiffNetwork(
            info_dict['num_labels'],
            info_dict['model_name']
        )
        diff_network.classifier.load_state_dict(info_dict['classifier_state_dict'])
            
        if info_dict["model_state"] == ModelState.DIFFPRUNING:
            for base_module in diff_network.get_encoder_base_modules():
                for n,p in list(base_module.named_parameters()):
                    parametrize.register_parametrization(base_module, n, DiffWeight(p, 0,
                        info_dict['concrete_lower'],
                        info_dict['concrete_upper'],
                        info_dict['structured_diff_pruning']
                    ))
        elif info_dict["model_state"] == ModelState.FIXMASK:
            for base_module in diff_network.get_encoder_base_modules():
                for n,p in list(base_module.named_parameters()):
                    p.requires_grad = False
                    parametrize.register_parametrization(base_module, n, DiffWeightFixmask(
                        torch.clone(p), torch.clone(p.bool())
                    ))          
    
        diff_network.encoder.load_state_dict(info_dict['encoder_state_dict'])
        diff_network._model_state = info_dict["model_state"]
        
        if remove_parametrizations:
            diff_network._remove_diff_parametrizations()           
        
        diff_network.eval()
        
        return diff_network


    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        log_ratio: float,
        max_grad_norm: float,
        gradient_accumulation_steps: int
    ) -> None:
        self.train()
        
        epoch_str = "training - step {}, loss: {:7.5f}, loss without l0 pen: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):
            
            inputs, labels = batch
            inputs = dict_to_device(inputs, self.device)
            outputs = self(**inputs)
            loss = loss_fn(outputs, labels.to(self.device))
            
            loss_no_pen = loss.item()
            
            if self._model_state == ModelState.DIFFPRUNING:
                l0_pen = 0.
                for module_name, base_module in self.get_encoder_base_modules(return_names=True):
                    layer_idx = self.get_layer_idx_from_module(module_name)
                    sparsity_pen = self.sparsity_pen[layer_idx]
                    module_pen = 0.
                    for par_list in list(base_module.parametrizations.values()):
                        for a in par_list[0].alpha_weights:
                            module_pen += self.get_l0_norm_term(a, log_ratio)
                    l0_pen += (module_pen * sparsity_pen)
                loss += l0_pen                   
                                      
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
                
            loss.backward()
            
            if ((step + 1) % gradient_accumulation_steps == 0) or ((step + 1) == len(epoch_iterator)):

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                
                self.zero_grad()
                
            logger.step_loss(self.global_step, loss, self.scheduler.get_last_lr()[0])
                           
            epoch_iterator.set_description(epoch_str.format(step, loss.item(), loss_no_pen), refresh=True)
            
            self.global_step += 1
        
                
    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        weight_decay: float,
        adam_epsilon: float,
        num_warmup_steps: int = 0,
        learning_rate_alpha: Optional[float] = None
    ) -> None:
        
        if self._model_state == ModelState.DIFFPRUNING:
            optimizer_params = [
                {
                    # diff params (last dense layer is in no_decay list)
                    # TODO needs to be changed when original weight is set to fixed pre trained
                    "params": [p for n,p in self.encoder.named_parameters() if n[-8:]=="finetune"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                },
                {
                "params": [p for n,p in self.encoder.named_parameters() if n[-5:]=="alpha" or n[-11:]=="alpha_group"],
                "lr": learning_rate_alpha
                },
                {
                    "params": self.classifier.parameters(),
                    "lr": learning_rate
                },
            ]
        else:
            optimizer_params = [{
                "params": self.parameters(),
                "lr": learning_rate
            }]            
        
        self.optimizer = AdamW(optimizer_params, eps=adam_epsilon)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        
        
    def _init_sparsity_pen(self, sparsity_pen: Union[float, List[float]]) -> None:        
        if isinstance(sparsity_pen, list):
            self.sparsity_pen = sparsity_pen
            assert len(sparsity_pen) == self.total_layers,  "invalid sparsity penalty per layer: # of layers mismatch"
        else:
            self.sparsity_pen = [sparsity_pen] * self.total_layers
            
                
    def _add_diff_parametrizations(self, *args) -> None:
        assert not self._parametrized, "cannot add diff parametrizations because of existing parametrizations in the model"
        for base_module in self.get_encoder_base_modules():
            for n,p in list(base_module.named_parameters()):
                parametrize.register_parametrization(base_module, n, DiffWeight(p, *args))   
        self._model_state = ModelState.DIFFPRUNING
          

    @torch.no_grad()
    def _finetune_to_fixmask(self, pct: float) -> None:
        
        def _get_cutoff(values, pct):
            k = int(len(values) * pct)
            return torch.topk(torch.abs(values), k, largest=True, sorted=True)[0][-1]            
        
        if self._model_state == ModelState.DIFFPRUNING:  
            diff_weights = torch.tensor([])
            for base_module in self.get_encoder_base_modules():
                for n, par_list in list(base_module.parametrizations.items()):
                    par_list[0].eval()
                    diff_weight = (getattr(base_module, n) - par_list.original).detach().cpu()
                    diff_weights = torch.cat([diff_weights, diff_weight.flatten()])
            cutoff = _get_cutoff(diff_weights, pct)
            for base_module in self.get_encoder_base_modules():
                for n, par_list in list(base_module.parametrizations.items()):
                    pre_trained_weight = torch.clone(par_list.original)
                    parametrize.remove_parametrizations(base_module, n)
                    p = base_module._parameters[n].detach()
                    diff_weight = (p - pre_trained_weight)
                    diff_mask = (torch.abs(diff_weight) >= cutoff)
                    base_module._parameters[n] = Parameter(diff_weight * diff_mask)
                    parametrize.register_parametrization(base_module, n, DiffWeightFixmask(pre_trained_weight, diff_mask))
        
        elif self._model_state == ModelState.FINETUNING:     
            diff_weights = torch.tensor([])
            pre_trained = AutoModel.from_config(self.encoder.config)
            for p, p_pre in zip(self.encoder.parameters(), pre_trained.parameters()):
                diff_weight = (p.cpu() - p_pre).flatten().detach()
                diff_weights = torch.cat([diff_weights, diff_weight])
            cutoff = _get_cutoff(diff_weights, pct)
            base_modules = dict(self.get_encoder_base_modules(return_names=True))
            for (n, p), p_pre in zip(list(self.encoder.named_parameters()), pre_trained.parameters()):
                n_parts = n.split(".")
                module_name, p_name = ".".join(n_parts[:-1]), n_parts[-1]
                base_module = base_modules[module_name]
                diff_weight = (p - p_pre.to(self.device))
                diff_mask = (torch.abs(diff_weight) >= cutoff)
                base_module._parameters[n] = Parameter(diff_weight * diff_mask)
                parametrize.register_parametrization(base_module, p_name, DiffWeightFixmask(p_pre.to(self.device), diff_mask))         
                
        self._model_state = ModelState.FIXMASK
                                        
    
    @torch.no_grad() 
    def _count_non_zero_params(self) -> Tuple[int, int]:
        assert self._parametrized, "Function only implemented for diff pruning"
        self.eval()
        n_p = 0
        n_p_zero = 0
        n_p_one = 0
        for base_module in self.get_encoder_base_modules():
            for par_list in list(base_module.parametrizations.values()):
                if isinstance(par_list[0], DiffWeightFixmask):
                    n_p_ = par_list[0].mask.numel()
                    n_p_zero_ = (~par_list[0].mask).sum()
                    n_p += n_p_
                    n_p_zero += n_p_zero_
                    n_p_one += (n_p_ - n_p_zero_)
                else:
                    z = par_list[0].z.detach()
                    n_p += z.numel()
                    n_p_zero += (z == 0.).sum()
                    n_p_one += (z == 1.).sum()
        self.train()
        n_p_between = n_p - (n_p_zero + n_p_one)
        return n_p, n_p_zero, n_p_between

    
    def _remove_diff_parametrizations(self) -> None:
        for module in self.get_encoder_base_modules():
            for n in list(module.parametrizations):
                parametrize.remove_parametrizations(module, n)
        self._model_state = ModelState.FINETUNING
