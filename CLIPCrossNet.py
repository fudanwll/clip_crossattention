import torch
from torch import nn
from modules.clipnet import CLIPFeatureExtractor
from modules.crossattention import CrossAttentionLayer
from modules.generatescore import GenerateScoreNet

class CLIPCrossNet(nn.Module):
    def __init__(self, model_path = None, device = None):
        super(CLIPCrossNet,self).__init__()
        #初始化CLIP特征提取器
        self.clip_feature_extractor = CLIPFeatureExtractor(pretrained= True, model_path=model_path, device= device)

        #初始化Cross_Attention
        d_model= 512
        n_head = 8
        dim_feedforward = 2048
        dropout = 0.1
        self.cross_attention_layer = CrossAttentionLayer(d_model,n_head,dim_feedforward,dropout)
        
        #初始化打分层
        self.generate_score_net = GenerateScoreNet(input_size= d_model)

    def forward(self, image1, text1, image2, text2):
        image_features_1, text_features_1 = self.clip_feature_extractor(image1, text1)
        image_features_2, text_features_2 = self.clip_feature_extractor(image2, text2)

        combined_features_1 = self.cross_attention_layer(image_features_1, text_features_1)
        combined_features_2 = self.cross_attention_layer(image_features_2, text_features_2)

        combined_feature = torch.concat([combined_features_1, combined_features_2],dim=1)

        score = self.generate_score_net(combined_feature)

        return score