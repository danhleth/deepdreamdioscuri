from dioscuri.base.models.backbones.mobilevit import MobileViT

from torch import nn

class ClassificationMobileViT(MobileViT):
    """MobileViT for classification
        modify output is logit with 8-dim vector
    """
    def __init__(self, image_size, dims, channels, num_classes=1000, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__(image_size, dims, channels, num_classes, expansion, kernel_size, patch_size)
        self.logit = nn.Linear(channels[-1], num_classes, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        
        logit = self.logit(x)
        return {"logit": logit}

def mobilevit_xs(image_size=256, num_classes=1000):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return ClassificationMobileViT((image_size, image_size), dims, channels, num_classes=num_classes)