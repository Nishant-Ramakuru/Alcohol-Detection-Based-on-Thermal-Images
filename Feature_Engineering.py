import pandas as pd
import numpy as np
import time

class Feature_Engineering():
    
    def __init__ (self,Sensors_dataframe):
        self.Sensors_dataframe = Sensors_dataframe

    def preprocess_data(self):

        self.Sensors_dataframe.index=pd.to_datetime(self.Sensors_dataframe.index, format='%Y-%m-%d %H:%M:%S') 
        self.Sensors_dataframe= self.Sensors_dataframe.sort_index(ascending=True) 

        '''converting time index to minute level with nearest fill method''' 
        time_delta='1Min'
        filling_method='nearest'
        start_time=self.Sensors_dataframe.index.min()
        end_time = self.Sensors_dataframe.index.max()
        self.Sensors_dataframe.groupby(self.Sensors_dataframe.index.map(lambda t: t.minute)).mean()
        self.Sensors_dataframe = self.Sensors_dataframe.groupby(self.Sensors_dataframe.index).mean().reindex(pd.date_range(start=start_time,end=end_time,freq="1min"),method=filling_method, tolerance=pd.Timedelta(time_delta))
        self.Sensors_dataframe.index = self.Sensors_dataframe.index.map(lambda x: x.replace(second=0))
      
    def stdev_df(self,x):
    '''callable function to creat standard deviation rolling window features'''

        return np.nanstd(x)
        
    def mean_df(self,x):
    '''callable function  to create rolling window features on variable mean,moving avarage '''
    
        return np.nanmean(x)

    def feature1(self,col,fun,window):
    '''base function for feauture creation'''
    '''feature is created based on the function selected'''

        return self.Sensors_dataframe[col].rolling(window).apply(fun,raw=False)

    def Create_features(self):
    '''creating features for 5,10,15,20,30 mints window for current temp and velocity'''

        sensor_columns = Sensors_dataframe.columns
        
        for win in [5,10,15,20,30]:
            for col in sensor_columns:
                self.Sensors_dataframe[col+'_'+'W'+str(win)+'_mean']=feature1(col,mean_df,win)
           
                self.Sensors_dataframe[col+'_'+'W'+str(win)+'_stdev']=feature1(col,stdev_df,win)
            
        self.Sensors_dataframe.drop(sensor_columns,axis=1,inplace=True)
            
        return self.Sensors_dataframe


