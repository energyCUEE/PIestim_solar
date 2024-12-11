import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split


def cloudcondition_splitter(df, valratio = 0.2, testratio = None, irr_colname = 'I', clearirr_colname = 'Iclr'
                            , site_id_colname = 'Site_id', datetime_colname = 'Datetime', method = 'k_bar'
                            , clear_threshold_k = 0.8, partly_threshold_k = 0.6, min_threshold_k = 0.6
                            , returncond = False
                            , countconcave_threshold = 1
                            , count_lowk_threshold = 3
                            , randomseed = 42):

    ## Find the splitted datetime, site_id which meet the cloud conditions
    df_split = df.copy()
    df_split = df_split[[site_id_colname, irr_colname, clearirr_colname]]
    df_split['k'] = df_split[irr_colname]/df_split[clearirr_colname]
    df_split = df_split.between_time('8:00','16:00')

    # Group by 'Site_id' and the index (which is Datetime) at daily frequency, then calculate the daily average for 'k'
    df_split['kbar'] = df_split.groupby([site_id_colname, pd.Grouper(freq='D')])['k'].transform('mean')
    
    df_split['is_low_k'] = df_split['k']<= min_threshold_k
    df_split['count_low_k'] = df_split.groupby(['Site_id', pd.Grouper(freq='D')])['is_low_k'].transform('sum').astype(int)
    
    df_split['increasing'] = df_split.groupby(['Site_id', pd.Grouper(freq='D')])[irr_colname].transform('diff')
    df_split['is_increasing'] = (df_split['increasing'] > 0).astype(int)
    df_split['concave_point'] = df_split.groupby(['Site_id', pd.Grouper(freq='D')])['is_increasing'].transform('diff')
    df_split['is_concave_point'] = df_split['concave_point'] == -1
    df_split['sum_concave_point'] = df_split.groupby(['Site_id', pd.Grouper(freq='D')])['is_concave_point'].transform('sum')

    # Step 1: Remove the time component from the Datetime index by using .date
    df_sitedateindex = df_split.copy()
    df_sitedateindex.index = df_sitedateindex.index.normalize()  # This keeps only the date, removing the timestamp

    # Step 2: Drop duplicate entries to have one row per date for each 'Site_id'
    df_sitedateindex = df_sitedateindex.groupby([site_id_colname, df_sitedateindex.index]).agg({
        'kbar':'mean', 
        'count_low_k': 'mean',
        'sum_concave_point':'mean'}).reset_index()

    # Step 3: Set the 'Date' as the index
    df_sitedateindex.set_index([datetime_colname], inplace = True)    
    
    if method == 'k_bar':
        # Dictionary for conditions and their labels
        conditions = {
        'clearsky': df_sitedateindex['kbar'] >= clear_threshold_k,
        'partlycloudy': (df_sitedateindex['kbar'] >= partly_threshold_k) & (df_sitedateindex['kbar'] < clear_threshold_k),
        'cloudy': df_sitedateindex['kbar'] < partly_threshold_k
        }
    elif method == 'smoothness':
        clearskycond = (df_sitedateindex['sum_concave_point'] <= countconcave_threshold) & (df_sitedateindex['count_low_k'] <= count_lowk_threshold)

        conditions = {
        'clearsky': clearskycond,
        'partlycloudy': ~clearskycond & (df_sitedateindex['kbar'] >= partly_threshold_k),
        'cloudy': ~clearskycond & (df_sitedateindex['kbar'] < partly_threshold_k)
        }
        
    else:
        raise ValueError(f'Invalid method {method} was specified')
    
    if testratio is not None:

        df_train_index, df_val_index, df_test_index = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # Iterate through each condition and split data
        for condition_name, condition in conditions.items():
            if returncond:
#                 df_condition = df_sitedateindex[condition].copy()
#                 df_condition = df_sitedateindex[condition].copy().drop(columns = ['count_low_k', 'sum_concave_point'])
                df_condition = df_sitedateindex[condition].copy()
                df_condition['skycondition'] = condition_name
            else:
#                 df_condition = df_sitedateindex[condition].copy().drop(columns = ['kbar'])
                df_condition = df_sitedateindex[condition].copy().drop(columns = ['kbar','count_low_k', 'sum_concave_point'])

            # Split into 80% train and 20% temp (val + test)
            train, temp = train_test_split(df_condition, test_size = (valratio + testratio), random_state=randomseed, stratify=df_condition[site_id_colname])

            # Split temp into 50% val and 50% test (i.e., 10% each of total)
            val, test = train_test_split(temp, test_size = testratio/(valratio+testratio), random_state=randomseed, stratify=temp[site_id_colname])

            # Concatenate to overall train/val/test sets
            df_train_index = pd.concat([df_train_index, train])
            df_val_index = pd.concat([df_val_index, val])
            df_test_index = pd.concat([df_test_index, test])
        
        df_train = df.copy()
        df_train['date'] = pd.to_datetime(df_train.index.date)
        df_train.reset_index(inplace = True)
        df_train = pd.merge(df_train, df_train_index, left_on = ['date', site_id_colname], right_on = [datetime_colname, site_id_colname], how = 'inner')
        df_train.drop(columns = ['date'], inplace = True)
        df_train.set_index(datetime_colname, inplace = True)
        
        df_val = df.copy()
        df_val['date'] = pd.to_datetime(df_val.index.date)
        df_val.reset_index(inplace = True)
        df_val = pd.merge(df_val, df_val_index, left_on = ['date', site_id_colname], right_on = [datetime_colname, site_id_colname], how = 'inner')
        df_val.drop(columns = ['date'], inplace = True)
        df_val.set_index(datetime_colname, inplace = True)
        
        df_test = df.copy()
        df_test['date'] = pd.to_datetime(df_test.index.date)
        df_test.reset_index(inplace = True)
        df_test = pd.merge(df_test, df_test_index, left_on = ['date', site_id_colname], right_on = [datetime_colname, site_id_colname], how = 'inner')
        df_test.drop(columns = ['date'], inplace = True)
        df_test.set_index(datetime_colname, inplace = True)
        
        
        return df_train, df_val, df_test
    else:
        
        df_train_index, df_val_index = pd.DataFrame(), pd.DataFrame()
        # Iterate through each condition and split data
        for condition_name, condition in conditions.items():
            df_condition = df_sitedateindex[condition].copy().drop(columns = ['kbar'])

            # Split into 80% train and 20% temp (val + test)
            train, val = train_test_split(df_condition, test_size = valratio, random_state=randomseed, stratify=df_condition['Site_id'])

            # Concatenate to overall train/val/test sets
            df_train_index = pd.concat([df_train_index, train])
            df_val_index = pd.concat([df_val_index, val])
            
        df_train = df.copy()
        df_train['date'] = pd.to_datetime(df_train.index.date)
        df_train.reset_index(inplace = True)
        df_train = pd.merge(df_train, df_train_index, left_on = ['date', site_id_colname], right_on = [datetime_colname, site_id_colname], how = 'inner')
        df_train.drop(columns = ['date'], inplace = True)
        df_train.set_index(datetime_colname, inplace = True)
        
        df_val = df.copy()
        df_val['date'] = pd.to_datetime(df_val.index.date)
        df_val.reset_index(inplace = True)
        df_val = pd.merge(df_val, df_val_index, left_on = ['date', site_id_colname], right_on = [datetime_colname, site_id_colname], how = 'inner')
        df_val.drop(columns = ['date'], inplace = True)
        df_val.set_index(datetime_colname, inplace = True)
        
        return df_train, df_val
    
def cloudcondition_ratio(df_train, df_val, df_test, method = 'k_bar', clear_threshold_k = 0.85
                         , partly_threshold_k = 0.75):
    if method == 'k_bar':
        fig, ax = plt.subplots(ncols = 2, figsize = (10, 4))
        ax[0].hist(df_train['kbar'], bins = 50, alpha = 0.7, label = r'kbar')
        ax[0].set_ylabel('Frequency')
        ax[0].set_title('Histogram of kbar on training dataset')
        ax[1].hist(df_train['kbar'], bins = 50, density = True, histtype = 'step'
                   , cumulative = True, label = 'kbar')
        ax[1].set_ylabel('Probability')
        ax[1].set_title('CDF of kbar on training dataset')

        for i in range(2):
            ax[i].grid()
            ax[i].set_xlabel('kbar')
            ax[i].axvline(clear_threshold_k, color = 'red', linestyle = 'dashed', linewidth = 1,
                         label = f'clear: kbar > {clear_threshold_k}')
            ax[i].axvline(partly_threshold_k, color = 'red', linestyle = 'dashed', linewidth = 1,
                         label = f'partly: {clear_threshold_k} < kbar < {partly_threshold_k}')
            ax[i].legend(fontsize = 8)
    else:
        pass

    df_ana_list = [df_train, df_val, df_test]
    df_name = ['Train set', 'Validation set', 'Test set']

    cond_list = ['clearsky', 'partlycloudy', 'cloudy']
    fig, ax = plt.subplots(ncols = 3, figsize = (15, 5))
    for i, df_ana in enumerate(df_ana_list):
        num_samples_list = []
        for cond in cond_list:
            num_cond = len(df_ana[df_ana['skycondition'] == cond])
            num_samples_list.append(num_cond)

        ax[i].pie(num_samples_list, labels = cond_list, autopct='%1.1f%%')
        ax[i].set_title(df_name[i])

def add_traces(fig, df, row, name, custom_columns):
    # Include datetime as a custom column
    customdata = np.column_stack([df[col] for col in custom_columns] + [df.index.astype(str)])
    
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['I'],
        mode='lines',
        name=f'I: {name}',
        customdata=customdata,
        hovertemplate='I: %{y}<br>' + 
                      '<br>'.join([f'{col}: %{{customdata[{i}]}}' for i, col in enumerate(custom_columns)]) + 
                      f'<br>Datetime: %{{customdata[{len(custom_columns)}]}}<extra></extra>'),
        row=row, col=1)
    
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['Iclr'],
        mode='lines',
        name=f'Iclr: {name}',
        line=dict(dash='dash'),
        customdata=customdata,
        hovertemplate='Iclr: %{y}<br>' + 
                      '<br>'.join([f'{col}: %{{customdata[{i}]}}' for i, col in enumerate(custom_columns)]) + 
                      f'<br>Datetime: %{{customdata[{len(custom_columns)}]}}<extra></extra>'),
        row=row, col=1)