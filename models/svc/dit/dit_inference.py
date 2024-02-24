# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from diffusers import DDIMScheduler, DDPMScheduler, PNDMScheduler

from models.svc.base import SVCInference
from models.svc.diffusion.diffusion_inference_pipeline import DiffusionInferencePipeline
from models.svc.diffusion.diffusion_wrapper import DiffusionWrapper
from modules.encoder.condition_encoder import ConditionEncoder
from models.svc.dit.ditsvc import DiTSVC


class DiTSVCInference(SVCInference):
    def __init__(self, args=None, cfg=None, infer_type="from_dataset"):
        SVCInference.__init__(self, args, cfg, infer_type)


    def _build_model(self):
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        self.acoustic_mapper = DiTSVC(self.cfg)
        model = torch.nn.ModuleList([self.condition_encoder, self.acoustic_mapper])
        return model

    def _inference_each_batch(self, batch):
        device = self.accelerator.device
        for k, v in batch.items():
            batch[k] = v.to(device)

        x = batch["latent"]

        cond_dict = self.condition_encoder(batch)
        cond = cond_dict["spk_id"]
        seq_cond_key = [k for k in cond_dict.keys() if k.startswith("frame_") or k.endswith("_feat")]
        # sum all the sequence conditioners
        seq_cond = torch.cat([cond_dict[k].unsqueeze(0) for k in seq_cond_key], dim=0)
        seq_cond = torch.sum(seq_cond, dim=0)
        mask = batch["mask"]

        y_pred = self.acoustic_mapper(x, mask, cond, seq_cond, infer=True)

        return y_pred
