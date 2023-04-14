import pandas as pd

class NJCleaner():

    def __init__(self, path:str) -> None:
        self.data = pd.read_csv(path)

    def order_by_scheduled_time(self) -> pd.DataFrame:
        return self.data.sort_values(by=['scheduled_time'])
    
    def drop_columns_and_nan(self) -> pd.DataFrame:
        df = self.data.copy()
        df.drop(columns=['from', 'to'], inplace=True)
        df.dropna(how='any', inplace=True)
        return df
    
    def convert_date_to_day(self) -> pd.DataFrame:
        df = self.data.copy()
        df['day'] = pd.to_datetime(df['date']).dt.day_name()
        df.drop(columns=['date'], inplace=True)
        return df
    
    def convert_scheduled_time_to_part_of_the_day(self) -> pd.DataFrame:
        df = self.data.copy()

        times = [0, 4, 8, 12, 16, 20, 24]
        labels = ['late_night', 'early_morning', 'morning', 'afternoon', 'evening', 'night']

        df['part_of_day'] = pd.cut(pd.to_datetime(df['scheduled_time']).dt.hour, bins=times, labels=labels, include_lowest=True)
        df.drop(columns=['scheduled_time'], inplace=True)
        return df
    
    def convert_delay(self) -> pd.DataFrame:
        df = self.data.copy()
        df.loc[df['delay_minutes'] < 5, 'delay'] = 0
        df.loc[df['delay_minutes'] >= 5, 'delay'] = 1
        df['delay'] = df['delay'].astype(int)

        return df
    
    def drop_unnecessary_columns(self) -> pd.DataFrame:
        df = self.data.copy()
        df.drop(columns=['train_id', 'actual_time', 'delay_minutes'], inplace=True)
        return df

    def save_first_60k(self, path:str) -> None:
        self.data.head(60000).to_csv(path, index=False)

    def prep_df(self, path:str = 'data/NJ.csv') -> None:
        self.data = self.order_by_scheduled_time()
        self.data = self.drop_columns_and_nan()
        self.data = self.convert_date_to_day()
        self.data = self.convert_scheduled_time_to_part_of_the_day()
        self.data = self.convert_delay()
        self.data = self.drop_unnecessary_columns()
        self.save_first_60k(path)