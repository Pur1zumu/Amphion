# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from diffusers import DDPMScheduler

from models.svc.base import SVCTrainer
from modules.encoder.condition_encoder import ConditionEncoder
from .ditsvc import DiTSVC


class DiTSVCTrainer(SVCTrainer):
    r"""The base trainer for all diffusion models. It inherits from SVCTrainer and
    implements ``_build_model`` and ``_forward_step`` methods.
    """

    def __init__(self, args=None, cfg=None):
        SVCTrainer.__init__(self, args, cfg)


    ### Following are methods only for diffusion models ###
    def _build_model(self):
        r"""Build the model for training. This function is called in ``__init__`` function."""

        # TODO: sort out the config
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        self.acoustic_mapper = DiTSVC(self.cfg)
        model = torch.nn.ModuleList([self.condition_encoder, self.acoustic_mapper])

        num_of_params_encoder = self.count_parameters(self.condition_encoder)
        num_of_params_am = self.count_parameters(self.acoustic_mapper)
        num_of_params = num_of_params_encoder + num_of_params_am
        log = "Diffusion Model's Parameters: #Encoder is {:.2f}M, #Diffusion is {:.2f}M. The total is {:.2f}M".format(
            num_of_params_encoder / 1e6, num_of_params_am / 1e6, num_of_params / 1e6
        )
        self.logger.info(log)

        return model

    def count_parameters(self, model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(p.numel() for p in model[key].parameters())
        else:
            model_param = sum(p.numel() for p in model.parameters())
        return model_param

    def _forward_step(self, batch):
        r"""Forward step for training and inference. This function is called
        in ``_train_step`` & ``_test_step`` function.
        """

        x = batch["latent"]

        cond_dict = self.condition_encoder(batch)
        cond = cond_dict["spk_id"]
        seq_cond_key = [k for k in cond_dict.keys() if k.startswith("frame_") or k.endswith("_feat")]
        # sum all the sequence conditioners
        seq_cond = torch.cat([cond_dict[k].unsqueeze(0) for k in seq_cond_key], dim=0)
        seq_cond = torch.sum(seq_cond, dim=0)
        mask = batch["mask"]

        loss = self.acoustic_mapper(x, mask, cond, seq_cond, infer=False)

        return loss
