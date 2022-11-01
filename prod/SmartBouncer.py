import pandas as pd
import pickle
import sys
import json
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder

from pip import main

def main():
    df=data_import()
    df=dataPreparation(df)
    df=url_preparation(df)
    df=method_preparation(df)
    predict=import_use_ml(df)
    return predict

def data_import():
    df_base = pd.read_json(sys.argv[1])
    return df_base

def dataPreparation(df_base):
    del df_base['_id']
    del df_base['instance']
    del df_base['date']
    del df_base['network']
    del df_base['rdns_name']
    del df_base['headers']
    del df_base['rdns_spoofed']
    del df_base['as_org']
    del df_base['c_code']
    del df_base['rdns']
    df_base['url_%'] = df_base['url']
    df_base['url_shell'] = df_base['url']
    df_base['url_jpg'] = df_base['url']
    df_base['url_png'] = df_base['url']
    df_base['url_jpeg'] = df_base['url']
    df_base['url_gif'] = df_base['url']
    df_base['url_css'] = df_base['url']
    df_base['url_svg'] = df_base['url']
    df_base['url_/tmp'] = df_base['url']
    df_base['url_env'] = df_base['url']
    return df_base

def url_preparation(df):
    #admin
    df.loc[df['url'].str.contains('admin'), 'url'] = 'admin'
    df['url'].mask(df['url'] != 'admin', 0, inplace=True)
    df['url'].mask(df['url'] == 'admin', 1, inplace=True)

    # %
    df.loc[df['url_%'].str.contains('%'), 'url_%'] = '%'
    df['url_%'].mask(df['url_%'] != '%', 0, inplace=True)
    df['url_%'].mask(df['url_%'] == '%', 1, inplace=True)

    #shell
    df.loc[df['url_shell'].str.contains('shell'), 'url_shell'] = 'shell'
    df['url_shell'].mask(df['url_shell'] != 'shell', 0, inplace=True)
    df['url_shell'].mask(df['url_shell'] == 'shell', 1, inplace=True)

    # .jpg
    df.loc[df['url_jpg'].str.contains('.jpg'), 'url_jpg'] = '.jpg'
    df['url_jpg'].mask(df['url_jpg'] != '.jpg', 0, inplace=True)
    df['url_jpg'].mask(df['url_jpg'] == '.jpg', 1, inplace=True)

    # .jpeg
    df.loc[df['url_jpeg'].str.contains('.jpeg'), 'url_jpeg'] = '.jpeg'
    df['url_jpeg'].mask(df['url_jpeg'] != '.jpeg', 0, inplace=True)
    df['url_jpeg'].mask(df['url_jpeg'] == '.jpeg', 1, inplace=True)

    # .png
    df.loc[df['url_png'].str.contains('.png'), 'url_png'] = '.png'
    df['url_png'].mask(df['url_png'] != '.png', 0, inplace=True)
    df['url_png'].mask(df['url_png'] == '.png', 1, inplace=True)

    # .gif
    df.loc[df['url_gif'].str.contains('.gif'), 'url_gif'] = '.gif'
    df['url_gif'].mask(df['url_gif'] != '.gif', 0, inplace=True)
    df['url_gif'].mask(df['url_gif'] == '.gif', 1, inplace=True)

    # .svg
    df.loc[df['url_svg'].str.contains('.svg'), 'url_svg'] = '.svg'
    df['url_svg'].mask(df['url_svg'] != '.svg', 0, inplace=True)
    df['url_svg'].mask(df['url_svg'] == '.svg', 1, inplace=True)

    # .css
    df.loc[df['url_css'].str.contains('.css'), 'url_css'] = '.css'
    df['url_css'].mask(df['url_css'] != '.css', 0, inplace=True)
    df['url_css'].mask(df['url_css'] == '.css', 1, inplace=True)

    # /tmp
    df.loc[df['url_/tmp'].str.contains('/tmp'), 'url_/tmp'] = '/tmp'
    df['url_/tmp'].mask(df['url_/tmp'] != '/tmp', 0, inplace=True)
    df['url_/tmp'].mask(df['url_/tmp'] == '/tmp', 1, inplace=True)

    # .env
    df.loc[df['url_env'].str.contains('.env'), 'url_env'] = '.env'
    df['url_env'].mask(df['url_env'] != '.env', 0, inplace=True)
    df['url_env'].mask(df['url_env'] == '.env', 1, inplace=True)
    return df

def method_preparation(df):
    df['method_GET']=df['method']
    df['method_HEAD']=df['method']
    df['method_OPTIONS']=df['method']
    df['method_POST']=df['method']
    df['method_PATCH']=df['method']
    del df['method']

    #GET
    df['method_GET'].mask(df['method_GET'] != 'GET', 0, inplace=True)
    df['method_GET'].mask(df['method_GET'] == 'GET', 1, inplace=True)

    # HEAD
    df['method_HEAD'].mask(df['method_HEAD'] != 'HEAD', 0, inplace=True)
    df['method_HEAD'].mask(df['method_HEAD'] == 'HEAD', 1, inplace=True)

    #OPTIONS
    df['method_OPTIONS'].mask(df['method_OPTIONS'] != 'OPTIONS', 0, inplace=True)
    df['method_OPTIONS'].mask(df['method_OPTIONS'] == 'OPTIONS', 1, inplace=True)

    # POST
    df['method_POST'].mask(df['method_POST'] != 'POST', 0, inplace=True)
    df['method_POST'].mask(df['method_POST'] == 'POST', 1, inplace=True)

    #PATCH
    df['method_PATCH'].mask(df['method_PATCH'] != 'PATCH', 0, inplace=True)
    df['method_PATCH'].mask(df['method_PATCH'] == 'PATCH', 1, inplace=True)
    return df

def import_use_ml(X_test):
    with open("Pickle_RL_Model.pkl", 'rb') as file:  
        Pickled_LR_Model = pickle.load(file)

    Ypredict = Pickled_LR_Model.predict(X_test)  
    return Ypredict

if __name__ == '__main__':
   print(main())
