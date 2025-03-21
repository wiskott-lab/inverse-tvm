from modules.detr.models.transformer import TransformerDecoder, TransformerDecoderLayer
from torch import nn


class InverseTransformerDecoder(TransformerDecoder):

    def __init__(self, d_model=256, nhead=8, num_layers=6, norm=None, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False, num_queries=100):
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        super().__init__(decoder_layer, num_layers, norm)
        self.query_embed = nn.Embedding(num_queries, d_model)


if __name__ == '__main__':
    inv_decoder = InverseTransformerDecoder()
