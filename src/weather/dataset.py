import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import requests

class WeatherDataset(Dataset):
    def __init__(self, data: pd.DataFrame, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.features = torch.tensor(data.drop(columns=['Original_Temp', 'Target_Diff']).values, dtype=torch.float32)
        self.target = torch.tensor(data['Target_Diff'].values, dtype=torch.float32)
        self.actual_temp = torch.tensor(data['Original_Temp'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        seq = self.features[idx:idx+self.sequence_length]
        target_diff = self.target[idx+self.sequence_length]
        last_actual_temp = self.actual_temp[idx+self.sequence_length-1]
        next_actual_temp = self.actual_temp[idx+self.sequence_length]
        return seq, target_diff, last_actual_temp, next_actual_temp

def get_weather_dataloaders(batch_size=32, sequence_length=30):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": -37.814,
        "longitude": 144.9633,
        "start_date": "2010-01-01",
        "end_date": "2023-12-31",
        "daily": ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"],
        "timezone": "Australia/Sydney"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data['daily'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.rename(columns={'temperature_2m_mean': 'Temp'}, inplace=True)
    df.ffill(inplace=True)
    
    df['Original_Temp'] = df['Temp'].copy()
    df['Target_Diff'] = df['Original_Temp'].diff().shift(-1)
    
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['lag_1'] = df['Temp'].shift(1)
    df['lag_2'] = df['Temp'].shift(2)
    df['lag_3'] = df['Temp'].shift(3)
    df['rolling_mean_7'] = df['Temp'].rolling(window=7).mean()
    df['rolling_std_7'] = df['Temp'].rolling(window=7).std()
    
    df = df.dropna()
    
    diff_mean = df['Target_Diff'].mean()
    diff_std = df['Target_Diff'].std()
    df['Target_Diff'] = (df['Target_Diff'] - diff_mean) / diff_std
    
    cols_to_normalize = [c for c in df.columns if c not in ['Original_Temp', 'Target_Diff']]
    for col in cols_to_normalize:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    train_dataset = WeatherDataset(train_data, sequence_length)
    test_dataset = WeatherDataset(test_data, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = len(cols_to_normalize)
    
    return train_loader, test_loader, diff_mean, diff_std, input_size