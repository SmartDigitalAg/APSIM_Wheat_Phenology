import os
import pandas as pd
from tqdm import tqdm

def main():
    result_filenames = [f for f in os.listdir('./output/parameter_predict') if f.endswith('.csv')]

    result_files = []
    for filename in tqdm(result_filenames):
        site_name = filename.split('_')[0]
        Rp = filename.split('_')[3].replace('Rp', '')
        Rv = filename.split('_')[4].split('.')[0].replace('Rv', '')
        tt_emergence = filename.split('_')[5].split('.')[0].replace('tt_emergence', '')
        tt_end_of_juvenile = filename.split('_')[6].split('.')[0].replace('tt_end_of_juvenile', '')
        tt_floral_initiation = filename.split('_')[7].split('.')[0].replace('tt_floral_initiation', '')
        tt_flowering = filename.split('_')[8].split('.')[0].replace('tt_flowering', '')
        tt_start_grain_fill = filename.split('_')[9].split('.')[0].replace('tt_start_grain_fill', '')
        tt_end_grain_fill = filename.split('_')[10].split('.')[0].replace('tt_end_grain_fill', '')
        df = pd.read_csv(os.path.join('./output/parameter_predict', filename), parse_dates=["Date"])

        first_date = df.loc[0, 'Date']
        doy = first_date.day_of_year
        df['sowing_date'] = doy
        df['Rp'] = float(Rp)
        df['Rv'] = float(Rv)
        df['tt_emergence'] = float(tt_emergence)
        df['tt_end_of_juvenile'] = float(tt_end_of_juvenile)
        df['tt_floral_initiation'] = float(tt_floral_initiation)
        df['tt_flowering'] = float(tt_flowering)
        df['tt_start_grain_fill'] = float(tt_start_grain_fill)
        df['tt_end_grain_fill'] = float(tt_end_grain_fill)

        df = df.iloc[[-1]]
        df['Site'] = site_name
        result_files.append(df)

    result_df = pd.concat(result_files, ignore_index=True, axis=0)
    result_df.to_csv('parameter_scenario_output_very_early16.csv', index=False)
    # print(result_df)




if __name__ == '__main__':
    main()