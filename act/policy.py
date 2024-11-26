import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import hydra
from act.detr.main import (
    build_ACT_model_and_optimizer,
    build_CNNMLP_model_and_optimizer,
)
from omegaconf import OmegaConf
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
import IPython

e = IPython.embed



class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        cfg = OmegaConf.load(args_override['cfg'])
        encoder, _ = build_CNNMLP_model_and_optimizer(args_override['encoder'])
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy,encoder=encoder)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, : self.model.horizon]
            is_pad = is_pad[:, : self.model.horizon]
            batch = {}
            batch['action'] = actions
            batch['obs'] = {'qpos':qpos,'image':image}
            raw_loss = self.model.compute_loss(batch)
            
            loss_dict = dict()
            all_l1 = raw_loss
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["loss"] = l1
            return loss_dict
        else:  # inference time
            obs_dict = {'qpos':qpos,'image':image}
            result = self.model.predict_action(obs_dict)
            return result['action_pred']

    def configure_optimizers(self):
        return self.optimizer
    
    def get_samples(self, qpos, image, num_samples=10, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        image = torch.tile(image, (num_samples,1,1,1,1))
        qpos = torch.tile(qpos, (num_samples, 1))
        # inference time        
        obs_dict = {'qpos':qpos,'image':image}
        result = self.model.predict_action(obs_dict)
        return result['action_pred']
    
    

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(
                qpos, image, env_state
            )  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
