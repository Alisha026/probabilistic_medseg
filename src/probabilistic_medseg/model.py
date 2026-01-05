import torch
import torch.nn as nn


def downConv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def upConv_block(in_channels, out_channels):   
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )



class UNETEncoder(nn.Module):   
    def __init__(self, in_channels=3, init_features=32):
        super(UNETEncoder, self).__init__()
        features = init_features
        self.encoder1 = downConv_block(in_channels, features)
        self.encoder2 = downConv_block(features, features * 2)
        self.encoder3 = downConv_block(features * 2, features * 4)
        self.encoder4 = downConv_block(features * 4, features * 8)

        self.bottleneck = downConv_block(features * 8, features * 16)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        return bottleneck, [enc1, enc2, enc3, enc4]
    

class UNETDecoder(nn.Module):   
    def __init__(self, out_channels=1, init_features=32):
        super(UNETDecoder, self).__init__()
        features = init_features

        self.upconv4 = upConv_block(features * 16, features * 8)
        self.decoder4 = downConv_block(features * 16, features * 8)
        self.upconv3 = upConv_block(features * 8, features * 4)
        self.decoder3 = downConv_block(features * 8, features * 4)
        self.upconv2 = upConv_block(features * 4, features * 2)
        self.decoder2 = downConv_block(features * 4, features * 2)
        self.upconv1 = upConv_block(features * 2, features)
        self.decoder1 = downConv_block(features * 2, features)

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, bottleneck, enc_features):
        enc1, enc2, enc3, enc4 = enc_features
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return dec1


class AxisAlignedConvGaussian(nn.Module):
    def __init__(self, in_channels, latent_dim, init_features=32):
        super(AxisAlignedConvGaussian, self).__init__()
        self.latent_dim = latent_dim
        
        # VAE Encoder
        self.encoder = UNETEncoder(in_channels=in_channels, init_features=init_features)
         
        # Take bottleneck features and map to mu and log_var
        self.mu_log_var_conv = nn.Conv2d(init_features * 16 , 2 * latent_dim, kernel_size=1)                  

    def forward(self, x_input):
        bottleneck, _ = self.encoder(x_input)
        bottleneck = torch.nn.functional.adaptive_avg_pool2d(bottleneck, (1, 1))
        mu_log_var = self.mu_log_var_conv(bottleneck)
        mu_log_var = mu_log_var.squeeze(-1).squeeze(-1)  # Shape: (B, 2*latent_dim)
        mu, log_var = mu_log_var.chunk(2, dim=1)
        
        return mu, log_var
        
class FeatureCombiner(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        super(FeatureCombiner, self).__init__()
        self.latent_dim = latent_dim

        self.f_comb = nn.Sequential(
            nn.Conv2d(in_channels + latent_dim, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        
    def forward(self, features, z):
        # Expand z to match spatial dimensions
        z = z.unsqueeze(-1).unsqueeze(-1)
        z_expanded = z.expand(-1, -1, features.size(2), features.size(3))
        combined = torch.cat((features, z_expanded), dim=1)
        logits = self.f_comb(combined)
        return logits
    
        
   
class ProbabilisticUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, latent_dim=6, init_features=32):
        super(ProbabilisticUNet, self).__init__()
        self.latent_dim = latent_dim
        
        self.unet_encoder = UNETEncoder(in_channels=in_channels, init_features=init_features)
        self.unet_decoder = UNETDecoder(out_channels=out_channels, init_features=init_features)

        self.prior_net = AxisAlignedConvGaussian(in_channels=in_channels, latent_dim=latent_dim, init_features=init_features)
        self.posterior_net = AxisAlignedConvGaussian(in_channels=in_channels + 1, latent_dim=latent_dim, init_features=init_features)

        self.feature_combiner = FeatureCombiner(in_channels= init_features, out_channels=out_channels, latent_dim=latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_input, mask_set=None):
        # Encode input image
        bottleneck, enc_features = self.unet_encoder(x_input)

        unet_logits = self.unet_decoder(bottleneck, enc_features)
        
        # Compute prior
        prior_mu, prior_log_var = self.prior_net(x_input)
        
        if self.training:
            # Compute posterior during training
            mask = mask_set[:,0,:,:,:] # Assuming mask_set shape is (B, num_samples, 1, H, W)
            y_input_combined = torch.cat((x_input, mask), dim=1)
            posterior_mu, posterior_log_var = self.posterior_net(y_input_combined)

            # Sample z from posterior
            z = self.reparameterize(posterior_mu, posterior_log_var)
            
            final_logits = self.feature_combiner(unet_logits, z)
            
            return final_logits, (prior_mu, prior_log_var), (posterior_mu, posterior_log_var)
        else:
            # Sample z from prior during inference
            z = self.reparameterize(prior_mu, prior_log_var)
            final_logits = self.feature_combiner(unet_logits, z)
            return final_logits 
        
        
        
        
# Deterministic U-Net with mc dropout for uncertainty estimation (Epistemic Uncertainty)
def downConv_block_withDropout(in_channels, out_channels, dropout_prob=0.3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout_prob),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout_prob),
    )

class DeterministicUNET(nn.Module):   
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(DeterministicUNET, self).__init__()
        features = init_features
        self.encoder1 = downConv_block_withDropout(in_channels, features)
        self.encoder2 = downConv_block_withDropout(features, features * 2)
        self.encoder3 = downConv_block_withDropout(features * 2, features * 4)
        self.encoder4 = downConv_block_withDropout(features * 4, features * 8)

        self.bottleneck = downConv_block_withDropout(features * 8, features * 16)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.upconv4 = upConv_block(features * 16, features * 8)
        self.decoder4 = downConv_block_withDropout(features * 16, features * 8, dropout_prob=0.1)
        self.upconv3 = upConv_block(features * 8, features * 4)
        self.decoder3 = downConv_block(features * 8, features * 4)
        self.upconv2 = upConv_block(features * 4, features * 2)
        self.decoder2 = downConv_block(features * 4, features * 2)
        self.upconv1 = upConv_block(features * 2, features)
        self.decoder1 = downConv_block(features * 2, features)

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    

