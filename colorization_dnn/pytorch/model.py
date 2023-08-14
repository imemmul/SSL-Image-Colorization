import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.base = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.base.children())[:-2])  # Remove last layers

    def forward(self, x):
        features = self.feature_extractor(x)
        return features



class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.base = FeatureExtractor()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=1)
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=1)
        self.conv10 = nn.Conv2d(1024, 1280, kernel_size=1)
        self.dropout = nn.Dropout2d(0.25)
        self.fusion = nn.Conv2d(2560, 1280, kernel_size=1)
        
    def forward(self, inputs):
        out_base = self.base(inputs)
        x = F.leaky_relu(self.conv1(inputs))
        x = F.leaky_relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        res_skip_1 = F.relu(self.conv5(x))
        x = res_skip_1
        res_skip_2 = F.relu(self.conv6(x))
        x = res_skip_2
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        # print(f"feature extractor out: {out_base.shape}") # feature extractor out: torch.Size([2, 1280, 7, 7])
        # print(f"enc out: {x.shape}") # enc out: torch.Size([2, 256, 7, 7])
        f = self.fusion(torch.cat((out_base, x), dim=1))
        # print(f"fusion shape: {f.shape}")
        skip_f = self.fusion(torch.cat((f, x), dim=1))
        # print(f"skip_f shape: {skip_f.shape}")
        # print(f"res_skip_1: {res_skip_1.shape}, res_skip_2:{res_skip_2.shape}")
        return skip_f, self.dropout(res_skip_1), self.dropout(res_skip_2)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convt1 = nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1)
        self.dropout = nn.Dropout2d(0.25)
        self.convt2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1)
        self.convt4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.convt5 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=1, padding=1)
        self.convt6 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.output_layer = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        
    def forward(self, skip_f, res_skip_1, res_skip_2):
        dec = F.relu(self.convt1(skip_f))
        dec = self.dropout(dec)
        
        res_skip_2_resized = F.interpolate(res_skip_2, size=dec.size()[2:], mode='nearest')
        # print(f"dec.shape: {dec.shape}, res_skip_2_resized.shape: {res_skip_2_resized.shape}")  # Print shapes for debugging
        dec = F.relu(self.convt2(torch.cat((dec, res_skip_2_resized), dim=1)))
        dec = self.dropout(dec)
        res_skip_1_resized = F.interpolate(res_skip_1, size=dec.size()[2:], mode='nearest')
        # print(f"dec.shape: {dec.shape}, res_skip_2_resized.shape: {res_skip_1_resized.shape}")
        dec = F.relu(self.convt3(torch.cat((dec, res_skip_1_resized), dim=1)))
        dec = self.dropout(dec)
        dec = F.relu(self.convt4(dec))
        dec = self.dropout(dec)
        dec = F.relu(self.convt5(dec))
        dec = self.convt6(dec)
        return self.output_layer(dec)

class Colorization_Model(nn.Module):
    def __init__(self, in_channels):
        super(Colorization_Model, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()
        
    def forward(self, inputs):
        skip_f, dropout_res_skip_1, dropout_res_skip_2 = self.encoder(inputs)
        return self.decoder(skip_f, dropout_res_skip_1, dropout_res_skip_2)

from torchsummary import summary
if __name__ == "__main__":
    model = FeatureExtractor()
    enc = Encoder(in_channels=3)
    model.to("cuda")
    enc.to("cuda")
    # summary(model=model, input_size=(3, 224,224), batch_size=32, device="cuda")
    # summary(model=enc, input_size=(3, 224,224), batch_size=32, device="cuda")