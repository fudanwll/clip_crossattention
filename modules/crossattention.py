from torch import nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        super(CrossAttentionLayer, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_feature, text_feature):
        #image query text 
        image_query_output, _ = self.multi_head_attention(image_feature, text_feature, text_feature)
        #text query image
        text_query_output, _ = self.multi_head_attention(text_feature, image_feature, image_feature)

        #combine two features
        combined_features = image_query_output + text_query_output
        combined_features = self.layer_norm1(combined_features)

        ff_output = self.feed_forward(combined_features)

        final_output = self.layer_norm2(combined_features + self.dropout(ff_output))

        return final_output