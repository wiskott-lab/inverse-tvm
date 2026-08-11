from modules.detr.models.transformer import TransformerEncoder, TransformerEncoderLayer


class InverseTransformerEncoder(TransformerEncoder):

    def __init__(self, d_model=256, nhead=8, num_layers=6, norm=None, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        super().__init__(encoder_layer, num_layers, norm)


if __name__ == '__main__':
    ite = InverseTransformerEncoder()
    pass
