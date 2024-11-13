import torch
import clip
from torch import nn

class CLIPFeatureExtractor(nn.Module):
    def __init__(self,pretrained=True, model_path = None, device = None):
        super(CLIPFeatureExtractor,self).__init__()

        if pretrained:
            if model_path is None:
                raise ValueError("model_path must be specified for pretrained models")
            
            self.model, self.process = clip.load("ViT-L/14", device=device, jit=False, model_path=model_path)
            
    def forward(self, image, text):
        image = self.process(image).unsqueeze(0).to(self.device)
        text = clip.tokenize([text]).to(self.device)
        # size [1,512]
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        # size [1,512]
        return image_features, text_features