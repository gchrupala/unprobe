import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from datasets import load_dataset

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

data = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
inputs = [ feature_extractor(datum["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_values  for datum in data ]

with torch.no_grad():
    outputs = []
    for x in inputs:
      out = model(x, output_hidden_states=True, return_dict=True)['hidden_states']
      means = []
      for z in out:
          print(z.shape)
          means.append(z.mean(dim=(0,1)))
      outputs.append(torch.stack(means))
      

print(outputs[10].shape)

