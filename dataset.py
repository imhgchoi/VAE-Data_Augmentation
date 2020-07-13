import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pdb

CCY = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','HKD','SGD','INR','THB','TWD']

class ForexDataset(Dataset):
    def __init__(self, config, mode=None):
        super(ForexDataset, self).__init__()
        self.config = config
        self.device = config.device
        self.batchsize = config.batchsize
        self.data_categories = [x+'KRW' for x in CCY]
        self.data_cat_num = len(CCY)

        if mode is None :
            if self.config.download_data :
                self.download()
            if self.config.preprocess :
                self.preprocess()
        else :
            self.read(mode)

            self.dataloader = DataLoader(dataset=self,
                                         batch_size=config.batchsize,
                                         shuffle=True,
                                         drop_last=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.items[idx]

    def download(self):
        print("\n downloading data...")
        for currency in tqdm(CCY) :
            ticker = '{}KRW=X'.format(currency)
            handle = yf.Ticker(ticker)
            data = handle.history(period='max', start='2004-01-02', end='2020-01-01', auto_adjust=True, actions=False)
            data.to_csv(self.config.datadir+'forex/raw/{}KRW.csv'.format(currency))

    def preprocess(self):
        files = os.listdir(self.config.datadir+'forex/raw/')

        # get all dates available
        dates = set()
        print('\n getting available dates...')
        for file in tqdm(files) :
            date = np.array(pd.read_csv(self.config.datadir+'forex/raw/{}'.format(file), usecols=['Date'])).flatten()
            dates = dates.union(date)
        dates = sorted(list(dates))
        template = pd.DataFrame({'Date':dates})

        print("\n processing data...")
        for file in tqdm(files) :
            data = pd.read_csv(self.config.datadir+'forex/raw/{}'.format(file))
            data = template.merge(data, how='left', on='Date').iloc[:,:-1]
            data = data.ffill(axis=0)
            data = data.dropna()

            # preprocess
            idx_low = data['Low'][0]
            data.iloc[:,1:] = data.iloc[:,1:] / idx_low

            # split
            train_years = list(range(2004,2020))
            test = pd.DataFrame()
            for year in self.config.test_years :
                mask = data['Date'].apply(lambda x : x>=str(year) and x<str(year+1))
                test = test.append(data.loc[mask, :])
                train_years.remove(year)
            val = pd.DataFrame()
            for year in self.config.val_years :
                mask = data['Date'].apply(lambda x : x>=str(year) and x<str(year+1))
                val = val.append(data.loc[mask, :])
                train_years.remove(year)
            train = pd.DataFrame()
            for year in train_years :
                mask = data['Date'].apply(lambda x : x>=str(year) and x<str(year+1))
                train = train.append(data.loc[mask, :])

            # save data
            train.to_csv(self.config.datadir+'forex/processed/train/{}'.format(file), index=False)
            val.to_csv(self.config.datadir+'forex/processed/val/{}'.format(file), index=False)
            test.to_csv(self.config.datadir+'forex/processed/test/{}'.format(file), index=False)


    def read(self, mode):
        print("\n reading {} data...".format(mode))
        files = os.listdir(self.config.datadir+'forex/processed/{}/'.format(mode))
        if self.config.debug :
            files = files[:3]
        data, items = [], []
        for file in tqdm(files) :
            item = file.split('.')[0]
            df = pd.read_csv(self.config.datadir+'forex/processed/{}/{}'.format(mode, file))
            arr = np.array(df['Close'])
            for i in range(len(arr) - self.config.window):
                data.append(torch.Tensor(arr[i:i+self.config.window]).to(self.device))
                items.append(item)
        self.data = data
        self.items = items



