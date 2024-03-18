import os
import time

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from gspread_pandas import Spread
import pickle

os.environ['GSPREAD_PANDAS_CONFIG_DIR'] = os.getcwd()


def get_tempLog():
    ss = Spread('1Nssf_r5YN-Epvu92NN8tgZvOe68Ys6E2_MWt3lA3jNc')
    df = ss.sheet_to_df(index=None)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', format='%m/%d/%Y %H:%M:%S')
    df['Unix Timestamp'] = df['Timestamp'].apply(lambda x: int(time.mktime(x.timetuple())))
    print("Retrieved temperature logs.")
    return df[['Timestamp', 'UnitID', 'Temperature', 'Unix Timestamp']]


def get_lotsBatches():
    ss = Spread('1dLGeD-G-PAHbaIuUXHeOXxIx07jUCy4dvN84WPvifWM')
    lots = ss.sheet_to_df(index=None, sheet='Vaccine Lots')
    batches = ss.sheet_to_df(index=None, sheet='Vaccine Batches').rename(columns={'Vaccine Type': 'Vaccine'})
    all_inventory = pd.concat([lots, batches], join='outer')
    print('Retrieved inventory.')
    return all_inventory[['LotID', 'BatchID', 'Vaccine', 'Lot #', 'Source', 'I-CARE PIN', 'Expiration Date', 'Storage Location']]


def get_storageUnits():
    ss = Spread('1dLGeD-G-PAHbaIuUXHeOXxIx07jUCy4dvN84WPvifWM')
    units = ss.sheet_to_df(index=None, sheet='Storage Locations')
    units = units.loc[~units['UnitID'].isin(['USAGE', 'WASTE'])]
    units['Maximum Temperature (C)'] = pd.to_numeric(units['Maximum Temperature (C)'])
    units['Minimum Temperature (C)'] = pd.to_numeric(units['Minimum Temperature (C)'])
    units['Last Recorded Temperature (C)'] = pd.to_numeric(units['Last Recorded Temperature (C)'])
    units['Last Recorded Temperature Timestamp'] = pd.to_datetime(units['Last Recorded Temperature Timestamp'],
                                                                  errors='coerce', format='%m/%d/%Y %H:%M:%S')
    print('Retrieved storage units')
    return units[['UnitID', 'Name', 'Storage Temperature', 'Minimum Temperature (C)', 'Maximum Temperature (C)',
                  'Last Recorded Temperature (C)', 'Last Recorded Temperature Timestamp']]


def get_transactions():
    ss = Spread('1dLGeD-G-PAHbaIuUXHeOXxIx07jUCy4dvN84WPvifWM')

    receptions_df = ss.sheet_to_df(index=None, sheet='Vaccine Reception')
    receptions_df = (receptions_df[['LotID', 'Quantity Received', 'Timestamp']]
                     .rename(columns={'LotID': 'Destination LotID', 'Quantity Received': 'Quantity'}))
    receptions_df['Timestamp'] = pd.to_datetime(receptions_df['Timestamp'], errors='coerce', format='%m/%d/%Y %H:%M:%S')
    transactions_df = ss.sheet_to_df(index=None, sheet='Transactions')
    transactions_df['Timestamp'] = pd.to_datetime(transactions_df['Timestamp'], errors='coerce', format='%m/%d/%Y %H:%M:%S')
    transactions_df = pd.concat([transactions_df, receptions_df], join='outer')
    transactions_df['Quantity'] = pd.to_numeric(transactions_df['Quantity'], 'coerce')
    transactions_df = transactions_df.replace(r'\s+( +\.)|#',np.nan,regex=True).replace('',np.nan)
    print('Retrieved transactions.')
    return transactions_df


def filter_transactions(transactions_df=None, lotID='', batchID=None, before=datetime.now()):
    if transactions_df is None:
        transactions_df = get_transactions()
    if before != datetime.now():
        transactions_df = transactions_df.loc[transactions_df['Timestamp'] < before]
    if lotID != '':
        transactions_df = transactions_df.loc[(transactions_df['LotID'] == lotID) | (transactions_df['Destination LotID'] == lotID)]
    if batchID is not None:
        if batchID == '':
            # Include rows where BatchID is NaN or empty
            transactions_df = transactions_df.loc[
                (transactions_df['BatchID'].isna()) | (transactions_df['Destination BatchID'].isna())]
        else:
            # Filter by specific BatchID
            transactions_df = transactions_df.loc[
                (transactions_df['BatchID'] == batchID) | (transactions_df['Destination BatchID'] == batchID)]
    return transactions_df


def calculate_inventory(dt=datetime.now(), unitID='', inventory_df=None, transactions_df=None):
    if inventory_df is None:
        inventory_df = get_lotsBatches()
    if transactions_df is None:
        transactions_df = get_transactions()

    def calculate_lotInventory(lotID, batchID=''):
        filtered = filter_transactions(transactions_df, lotID, batchID, before=dt)
        if batchID == "":
            additions = filtered.loc[filtered['Destination LotID'] == lotID, 'Quantity'].sum()
            subtractions = filtered.loc[(filtered['LotID'] == lotID) & (~filtered['Destination'].isin(["USAGE", "WASTE"])), 'Quantity'].sum()
            current_inventory = additions - subtractions
        else:
            additions = filtered.loc[filtered['Destination BatchID'] == batchID, 'Quantity'].sum()
            subtractions = filtered.loc[filtered['BatchID'] == batchID, 'Quantity'].sum()
            current_inventory = additions - subtractions
        return current_inventory

    inventory_df['Inventory'] = inventory_df.fillna("").apply(lambda row: calculate_lotInventory(row['LotID'], row['BatchID']), axis=1)
    if unitID != '':
        inventory_df = inventory_df.loc[inventory_df['Storage Location'] == unitID]
    return inventory_df


def aggregate_data(tempLog_df=pd.DataFrame(), inventory_df=pd.DataFrame(), transactions_df=pd.DataFrame()):
    if tempLog_df.empty:
        tempLog_df = get_tempLog()
    if inventory_df.empty:
        inventory_df = get_lotsBatches()
    if transactions_df.empty:
        transactions_df = get_transactions()

    def try_calculate_inventory(row):
        try:
            return calculate_inventory(row['Timestamp'], row['UnitID'], inventory_df, transactions_df).to_dict('records')
        except Exception as e:
            print(f"Error processing row: {row}, Error: {e}")
            return None

    tempLog_df['Inventory Snapshot'] = tempLog_df.apply(lambda row: try_calculate_inventory(row), axis=1)
    tempLog_df['Unix Timestamp'] = tempLog_df['Timestamp'].apply(lambda x: int(time.mktime(x.timetuple())))
    print('Completed aggregating data.')
    return tempLog_df


def expand_rows(agg_data):
    df = pd.DataFrame(columns=['Timestamp', 'UnitID', 'Temperature', 'LotID', 'BatchID', 'Vaccine', 'Lot #', 'Source', 'I-CARE PIN', 'Expiration Date', 'Storage Location', 'Inventory'])
    for row, data in agg_data.iterrows():
        for record in data['Inventory Snapshot']:
            add_row = {'Timestamp': data["Timestamp"],
                       'UnitID': data["UnitID"],
                       'Temperature': data["Temperature"],
                       'LotID': record['LotID'],
                       'BatchID': record['BatchID'],
                       'Vaccine': record["Vaccine"],
                       'Lot #': record["Lot #"],
                       'Source': record["Source"],
                       'I-CARE PIN': record["I-CARE PIN"],
                       'Expiration Date': record["Expiration Date"],
                       'Storage Location': record["Storage Location"],
                       'Inventory': record["Inventory"]}
            df = pd.concat([df, pd.DataFrame([add_row])])
    print('Completed expanding rows.')
    return df


def current_inventory(lots_df=None, temps_df=None):
    pass


if __name__ == '__main__':
    # x = aggregate_data()
    # y = expand_rows(x)
    z = get_storageUnits().to_dict('records')
    d = get_tempLog()
    e = get_lotsBatches()
    f = get_transactions()
    # g = aggregate_data(d, e, f)
    data_file = open('data/data.pickle', 'wb')
    pickle.dump({'timestamp': datetime.now(), 'df': d, 'temps': d, 'units': z, 'lots': e, 'transactions': f}, data_file)
    data_file.close()
    # data = pickle.load(data_file)
    print('x')

