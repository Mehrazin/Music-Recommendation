from .gru import *


def build_model(config):
    enc = Encoder(config).to(config.device)
    dec = Decoder(config).to(config.device)
    model = Seq2Seq(config, enc, dec).to(config.device)
    print('Model created')
    return model
