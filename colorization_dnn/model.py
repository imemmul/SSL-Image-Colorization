import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.base = torch.hub.load('pytorch/vision', 'efficientnet_b0', pretrained=True)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout2d(0.25)
        self.fusion = nn.Conv2d(768, 512, kernel_size=1, padding=0)
        
    def forward(self, inputs):
        out_base = self.base.extract_features(inputs)
        x = F.leaky_relu(self.conv1(inputs))
        x = F.leaky_relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        res_skip_1 = F.relu(self.conv5(x))
        x = res_skip_1
        res_skip_2 = F.relu(self.conv6(x))
        x = res_skip_2
        x = F.relu(self.conv7(x))
        f = self.fusion(torch.cat([out_base, x], dim=1))
        skip_f = torch.cat([f, x], dim=1)
        return skip_f, self.dropout(res_skip_1), self.dropout(res_skip_2)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convt1 = nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout2d(0.25)
        self.convt2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(384, 128, kernel_size=3, stride=2, padding=1)
        self.convt4 = nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2, padding=1)
        self.convt5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.convt6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.output_layer = nn.Conv2d(32, 2, kernel_size=1)
        
    def forward(self, skip_f, res_skip_1, res_skip_2):
        dec = F.relu(self.convt1(skip_f))
        dec = self.dropout(dec)
        dec = F.relu(self.convt2(torch.cat([dec, res_skip_2], dim=1)))
        dec = self.dropout(dec)
        dec = F.relu(self.convt3(torch.cat([dec, res_skip_1], dim=1)))
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
