import os
import yaml
import torchaudio

#### Experiment #######

from contextlib import suppress

import torch
import numpy as np
from torch import nn
import torchvision.transforms
import torchaudio.transforms as T


from audio_flamingo_2.my_laion_clap.CLAP.src.laion_clap.clap_module.htsat import create_htsat_model


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def int16_to_float32_torch(x):
    return (x / 32767.0).type(torch.float32)

def float32_to_int16_torch(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)


class CLAPAudioConfig:
    model_type: str = "HTSAT"
    model_name: str = "large"
    sample_rate: int = 16000
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 160
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    mel_bins: int = 64
    clip_samples: int = 160000


class CLAP(nn.Module):
    
    def __init__(self, clap_config, device=None):    
        super().__init__()        
        self.clap_config = clap_config
        self.method = clap_config["method"]

        # FIXME: define depending on the local system
        if device is None:
            device_id = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_id = torch.device(device)

        # Hardcoded stuff
        self.finetune = False
        enable_fusion = True
        fusion_type = "aff_2d"
        self.audio_cfg = CLAPAudioConfig()
        
        self.nvclap = create_htsat_model(self.audio_cfg, enable_fusion, fusion_type)

        # NOTE: setting weights_only=False, as PyTorch 2.6 changes the behavior
        clap_state_dict = torch.load(clap_config["checkpoint"], map_location = device_id, weights_only=False)
        clap_state_dict_copy = clap_state_dict['state_dict'].copy()

        # Remove the 'module.audio_branch.' prefix from the keys
        for key in list(clap_state_dict['state_dict'].keys()):
            if 'audio' in key:
                clap_state_dict_copy[key.replace('module.audio_branch.','')] = clap_state_dict_copy[key]
                del clap_state_dict_copy[key]
            else:
                del clap_state_dict_copy[key]
        
        # Load state dict in the model
        self.nvclap.load_state_dict(clap_state_dict_copy, strict=False)
        self.nvclap = self.nvclap.to(device_id)
        
        for param in self.nvclap.parameters():
            param.requires_grad = False

        self.nvclap.eval()
        
        # Mel spectrogram transform
        self.mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.audio_cfg.sample_rate,
            n_fft=self.audio_cfg.window_size,
            win_length=self.audio_cfg.window_size,
            hop_length=self.audio_cfg.hop_size,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm=None,
            onesided=True,
            n_mels=self.audio_cfg.mel_bins,
            f_min=self.audio_cfg.fmin,
            f_max=self.audio_cfg.fmax,
        )

        print('loaded NVCLAP model: {}'.format(clap_config["checkpoint"]))
                
    def get_mel(self, audio_data):
        self.mel_tf.to(audio_data.device)
        mel = self.mel_tf(audio_data) # (n_mels, T)
        mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
        return mel.T  # (T, n_mels)

    def get_audio_features(self, sample, audio_data, max_len, data_truncating, data_filling, require_grad=False):

        """
        FROM CLAP paper:
        We use a fixed chunk duration d = 10 seconds for all audio
        For an audio in T seconds and a fixed chunk duration d = 10 seconds:
        - T ≤ d: inputis repeated and then padded it with zero values.
        - T > d: input is downsamples from T to d-second as global input.
        Then three d-second clips are randomly sliced.
        These 4 × d inputs into the first layer of audio encoder

        # NOTE: I think this may not help with for strong temporal annotations.
        """

        grad_fn = suppress if require_grad else torch.no_grad

        with grad_fn():
            
            if len(audio_data) > max_len:
                
                # Data truncating method: fusion
                mel = self.get_mel(audio_data)
                print("Mel shape:", mel.shape)

                # Split to three parts: why?
                chunk_frames = max_len // 160 + 1  # the +1 related to how the spectrogram is computed
                print("Chunk frames:", chunk_frames)

                total_frames = mel.shape[0] # Temporal dimension
                print("Total frames:", total_frames)

                if chunk_frames == total_frames:
                    # there is a corner case where the audio length is
                    # larger than max_len but smaller than max_len+hop_size.
                    # In this case, we just use the whole audio.
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    print("Mel fusion shape:", mel_fusion.shape)

                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([False])
                
                else:
                    ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                    
                    if len(ranges[1]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[2] = [0]
                    
                    # randomly choose index for each part
                    idx_front = np.random.choice(ranges[0])
                    idx_middle = np.random.choice(ranges[1])
                    idx_back = np.random.choice(ranges[2])

                    print("Front index:", idx_front)
                    print("Middle index:", idx_middle)
                    print("Back index:", idx_back)
                    
                    # Select Mel
                    mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                    mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]

                    # Shrink the Mel
                    mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, 64])(mel[None])[0]

                    # Stack
                    mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([True])
            
                # random crop to max_len (for compatibility)
                overflow = len(audio_data) - max_len
                idx = np.random.randint(0, overflow + 1)
                audio_data = audio_data[idx: idx + max_len]

            else:  # padding if too short
                
                if len(audio_data) < max_len:  # do nothing if equal
                    
                    # Forcing filling method: repeatpad
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat)

                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )            

                # Data truncating method: fusion
                mel = self.get_mel(audio_data)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                sample["mel_fusion"] = mel_fusion
                longer = torch.tensor([False])

        sample["longer"] = longer
        sample["waveform"] = audio_data
        return sample
    
    def load_audio(self, clips):
        processed_clips = []
        
        for clip in clips:
            audio_data = int16_to_float32_torch(float32_to_int16_torch(clip))
            sample = self.get_audio_features({}, audio_data, 160000, "fusion", "repeatpad")
            processed_clips.append(sample)

        waveforms = {}
        waveforms["mel_fusion"] = torch.stack([item["mel_fusion"] for item in processed_clips], dim=0)
        waveforms["longer"] = torch.stack([item["longer"] for item in processed_clips], dim=0)
        waveforms["waveform"] = torch.stack([item["waveform"] for item in processed_clips], dim=0)

        return waveforms

    def forward(self, audio_clips):
        # It will handle various segments
        # 1 audio will have various segments [B X n_segments X time]
        # expand batch dimension during inference
        if len(audio_clips.shape) == 2:
            audio_clips = audio_clips.unsqueeze(0)
        assert len(audio_clips.shape) == 3

        audio_embeds = []
        for audio_clip in audio_clips:
            audio = self.load_audio(audio_clip)
            print("Audio shape:", audio["waveform"].shape)
            print("Mel shape:", audio["mel_fusion"].shape)
            print("Longer: ", audio["longer"])
            audio_embed = self.nvclap(audio) #.reshape(-1, self.clap_config["audio_embed_dim"])
            audio_embeds.append(audio_embed)

        audio_embeds = torch.stack(audio_embeds, dim=0)

        return audio_embeds

#### Experiment ####


def main():

    # Do not allow HF to connect to the internet
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load the config file
    config = yaml.load(open("src/audio_flamingo_2/config/inference.yaml"), Loader=yaml.FullLoader)
    clap_config = config['clap_config']    
    print("CLAP config:", clap_config)

    # Load sample audio
    filename = "/mnt/scratch/tmp/xlopezw00/MMAU/test-mini-audios/81684e06-43bd-4523-bbc3-56e4517f7ed8.wav"
    waveform, sr = torchaudio.load(filename)
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    print("Waveform shape:", waveform.shape)
    print("Sample rate:", sr)

    # get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    clap = CLAP(clap_config, device=device)
    clap.eval()

    # Process the audio
    audio_embeds = clap(waveform.to(device))

    print("Audio embeddings shape:", audio_embeds.shape)


if __name__ == "__main__":
    main()

