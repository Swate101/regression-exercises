import pandas as pd
import env
import os


def get_connection(db, user=env.user, host=env.host, password=env.password):
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def load_zillow_data():
    
    db = 'zillow'
    sql_query = '''
        select *
        from properties_2017
        join predictions_2017 as pred using(parcelid)
        where pred.transactiondate between '2017-05-01' and '2017-06-30';
        '''
    file = 'zillow.csv'
    
    if os.path.isfile(file):
        return pd.read_csv('zillow.csv')
    else:
        df = pd.read_sql(sql_query, get_connection(db))
        df.to_csv('zillow.csv', index=False)
        return df
