from config import get_args
from dataset import ForexDataset
from models import ae, vae
from trainer import Trainer
from evaluator import evaluate
import pdb

def get_data(config):
    if config.datatype == 'forex':
        ForexDataset(config)

        train = ForexDataset(config, 'train')
        val = ForexDataset(config, 'val')
        test = ForexDataset(config, 'test')

    elif config.datatype == 'equity':
        pass

    return [train, val, test]

def get_model(config):
    if config.model == 'ae' :
        model = ae.VanillaAE(config)
    elif config.model == 'vae' :
        model = vae.VanillaVAE(config)
    else :
        raise NotImplementedError

    return model

def main():
    config = get_args()

    if config.mode == 'eval' :
        evaluate(config)
    else :
        datasets = get_data(config)

        model = get_model(config)
        trainer = Trainer(config, datasets)
        trainer.train(model)


if __name__ == "__main__" :
    main()