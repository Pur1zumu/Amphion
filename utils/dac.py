# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import dac
from audiotools import AudioSignal


def extract_dac_latent(cfg, wav_path):
    """
    Extracts the latent representation of an audio signal using the DAC model.
    This latent is extracted from the encoder, 
    which is different from the latent output from the DAC's RVQ
    """
    model = dac.DAC.load(cfg.preprocess.dac_model_path)

    signal = AudioSignal(wav_path)

    if torch.cuda.is_available():
        model = model.cuda()
        signal = signal.to(model.device)
    with torch.no_grad():
        x = model.preprocess(signal.audio_data, signal.sample_rate)
        _, _, latents, _, _ = model.encode(x)

        return latents
