import torch
from models.decoder import LinearClassifier
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

class Dinov2ForSemanticSegmentation(torch.nn.Module):
  def __init__(self, config, num_classes):
    super().__init__()

    self.dinov2 = torch.hub.load(repo_or_dir='facebookresearch/dinov2', model=f"dinov2_{config.arch_name}").cuda()
    self.num_classes = num_classes
    self.classifier = LinearClassifier(config.hidden_size, 32, 32, num_labels=num_classes)
    print(self.dinov2)
  def forward(self, input, output_attentions=None, labels=None):
    # use frozen features
    outputs = self.dinov2.forward_features(input)
    patch_embeddings = outputs['x_norm_patchtokens']
    print(f"patch_embeddings shape: {patch_embeddings.shape}")
    self.classifier(patch_embeddings)
    loss = None
    if labels is not None:
      # print(f"logits shape: {logits.shape}")
      # print(f"labels shape: {labels.squeeze().shape}, {labels.dtype}")
      # print(f"label_values: {torch.isnan(labels.squeeze()).any()}")
      # print(f"logits: {torch.isnan(logits).any()}")
      loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
      torch.autograd.set_detect_anomaly(True)
      label_map = labels[0].detach().cpu().numpy().squeeze()
      plt.imsave(
          "./processing_label_map.png",
          label_map
      )
      loss = loss_fct(logits, labels.squeeze().to(torch.int64))
    return dict(
        loss=loss,
        logits=logits,
    )