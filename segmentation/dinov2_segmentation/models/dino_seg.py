import torch
from models.decoder import LinearClassifier
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
torch.autograd.set_detect_anomaly(True)
from torch import nn

class Dinov2ForSemanticSegmentation(torch.nn.Module):
  def __init__(self, config, num_classes):
    super().__init__()
    self.config = config
    self.dinov2 = torch.hub.load(repo_or_dir='facebookresearch/dinov2', model=f"dinov2_{config.arch_name}").cuda()
    self.num_classes = num_classes
    self.classifier = LinearClassifier(config.hidden_size, 32, 32, num_labels=num_classes)
    self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
  def forward(self, input, output_attentions=None, labels=None):
    # use frozen features
    outputs = self.dinov2.forward_features(input)
    patch_embeddings = outputs['x_norm_patchtokens']
    logits = self.classifier(patch_embeddings)
    print(f"shape of logits: {logits.shape}")
    logits = torch.nn.functional.interpolate(logits, size=input.shape[2:], mode="bilinear", align_corners=False)
    loss = None
    if output_attentions:
      self.save_attention_maps(input, patch_embeddings)
    if labels is not None:
      torch.autograd.set_detect_anomaly(True)
      label_map = labels[0].detach().cpu().numpy().squeeze()
      plt.imsave(
          "./processing_label_map.png",
          label_map
      )
      loss = self.loss_fn(logits, labels.squeeze().to(torch.int64))
    return dict(
        loss=loss,
        logits=logits,
    )
    
  def capture_attention_maps(self, output):
    # Assuming the attention maps are in the 'attn_output' attribute of the output tensor
    attentions = output.attn_output
    w, h = img.shape[1] - img.shape[1] % self.config.patch_size, img.shape[2] - img.shape[2] % self.config.patch_size
    img = img[:, :w, :h].unsqueeze(0)
    w_featmap = img.shape[-2] // self.config.patch_size
    h_featmap = img.shape[-1] // self.config.patch_size
    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    # for every patch
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    # weird: one pixel gets high attention over all heads?
    print(torch.max(attentions, dim=1)) 
    attentions[:, 283] = 0 

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=self.config.patch_size, mode="nearest")[0].cpu().numpy()
    
    for j in range(nh):
        fname = os.path.join(self.config.attention_out, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")
    for hook in self.hooks:
      hook.remove()

