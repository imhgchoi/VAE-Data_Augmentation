import argparse
import torch

def get_args():
    argp = argparse.ArgumentParser(description='Variational Data Augmentor',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument('--debug', action='store_true', default=False)
    argp.add_argument('--mode', type=str, default='train', choices=['train','eval'])
    argp.add_argument('--device', type=str, default='cpu', choices=['cuda','cpu'])

    # directories
    argp.add_argument('--raw_datadir', type=str, default='D:/data_repository/KRX_price_data/')
    argp.add_argument('--datadir', type=str, default='./data/')
    argp.add_argument('--outdir', type=str, default='./out/')

    # data
    argp.add_argument('--datatype', type=str, default='forex', choices=['forex','equity'])
    argp.add_argument('--download_data', action='store_true', default=False)
    argp.add_argument('--preprocess', action='store_true', default=False)
    argp.add_argument('--renew_data', action='store_true', default=False)
    argp.add_argument('--val_years', nargs='+', default=[2016, 2017])
    argp.add_argument('--test_years', nargs='+', default=[2018, 2019])

    # model
    argp.add_argument('--model', type=str, default='vae', choices=['ae','vae'])
    argp.add_argument('--reset_model', action='store_true', default=False)
    argp.add_argument('--window', type=int, default=100)
    argp.add_argument('--dropout', type=float, default=0.1)
    argp.add_argument('--init', type=str, default='none', choices=['none','xavier'])

    argp.add_argument('--ae_enc_dim', nargs='+', default=[256])
    argp.add_argument('--ae_dec_dim', nargs='+', default=[256])
    argp.add_argument('--ae_repr_dim', type=int, default=500)
    argp.add_argument('--ae_porify', type=float, default=0)

    argp.add_argument('--vae_enc_dim', nargs='+', default=[])
    argp.add_argument('--vae_dec_dim', nargs='+', default=[])
    argp.add_argument('--vae_repr_dim', type=int, default=64)
    argp.add_argument('--vae_logvar_factor', type=float, default=10)
    argp.add_argument('--vae_porify', type=float, default=0)
    argp.add_argument('--vae_start_std', type=int, default=3000)
    argp.add_argument('--vae_logvar_clip', type=float, default=0.5)

    argp.add_argument('--start_kld', type=float, default=200)
    argp.add_argument('--lstm_dec_dim', type=int, default=8)
    argp.add_argument('--cnn_layers', nargs='+', default=[64,32])
    argp.add_argument('--cnn_repr_dim', type=int, default=32)
    argp.add_argument('--cnn_beta', type=float, default=10)

    # training
    argp.add_argument('--model_name', type=str, default='basic')
    argp.add_argument('--epoch_num', type=int, default=500)
    argp.add_argument('--min_epochs', type=int, default=300)
    argp.add_argument('--min_steps', type=int, default=3000)
    argp.add_argument('--batchsize', type=int, default=1000)
    argp.add_argument('--lr', type=float, default=3e-5)
    argp.add_argument('--no_early_stop', action='store_true', default=False)
    argp.add_argument('--augment_num', type=int, default=5)
    argp.add_argument('--plot_step', type=int, default=100)

    config = argp.parse_args()
    if torch.cuda.is_available :
        config.device = 'cuda'
    print(config)
    return config