import torch
import torch
import torch.nn as nn

from gains.decoder import construct_decoder
from gains.encoder import construct_encoder
from feature_env import FeatureEvaluator

SOS_ID = 0
EOS_ID = 0


# gradient based automatic feature selection
class GAINS(nn.Module):
    def __init__(self,
                 fe:FeatureEvaluator,
                 args
                 ):
        super(GAINS, self).__init__()
        self.style = args.method_name
        self.gpu = args.gpu
        self.encoder = construct_encoder(fe, args)
        self.decoder = construct_decoder(fe, args)
        if self.style == 'rnn':
            self.flatten_parameters()

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, feat_emb, predict_value = self.encoder.forward(input_variable)
                     #encoder的输出 return encoder_outputs, encoder_hidden, seq_emb, predict_value
        decoder_hidden = (feat_emb.unsqueeze(0), feat_emb.unsqueeze(0))#unsqueeze(0)在第0维增加维度
        decoder_outputs, decoder_hidden, ret = self.decoder.forward(target_variable, decoder_hidden, encoder_outputs)
        decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
        feat = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return predict_value, decoder_outputs, feat

    def generate_new_feature(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, feat_emb, predict_value, new_encoder_outputs, new_feat_emb = \
            self.encoder.infer(input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_feat_emb.unsqueeze(0), new_feat_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder.forward(None, new_encoder_hidden, new_encoder_outputs)
        new_feat_seq = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return new_feat_seq

