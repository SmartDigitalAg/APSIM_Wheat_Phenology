import os
import pandas as pd
from tqdm import tqdm

def main():
    result_filenames = [f for f in os.listdir('./output/parameter_predict') if f.endswith('.csv')]

    result_files = []
    for filename in tqdm(result_filenames):
        site_name = filename.split('_')[0]
        Rp = filename.split('_')[3].replace('Rp', '')
        Rv = filename.split('_')[4].replace('Rv', '')
        tt_emergence = filename.split('_tt')[1].split('.')[0].replace('_emergence', '')
        tt_end_of_juvenile = filename.split('_tt')[2].split('.')[0].replace('_end_of_juvenile', '')
        tt_floral_initiation = filename.split('_tt')[3].split('.')[0].replace('_floral_initiation', '')
        tt_flowering = filename.split('_tt')[4].split('.')[0].replace('_flowering', '')
        tt_start_grain_fill = filename.split('_tt')[5].split('.')[0].replace('_start_grain_fill', '')
        tt_end_grain_fill = filename.split('_tt')[6].split('.')[0].replace('_end_grain_fill', '')
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
    result_df.to_csv('parameter_scenario_output_very_early20.csv', index=False)
    # print(result_df)




if __name__ == '__main__':
    main()