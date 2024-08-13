import pandas as pd
from datetime import datetime, timedelta
from thermal_time import APSIMWheatPhenology
from tqdm import tqdm
import os
from multiprocessing import Pool


def calculate_stage(df, stage_div, previous_stage_col, current_stage_tt, current_stage_col, temperature_col):
    stage = df.dropna(subset=[previous_stage_col])
    first_index = stage[previous_stage_col].first_valid_index()

    if first_index is not None:
        stage = df.loc[first_index + 1:].copy()
        stage[f'{current_stage_col}_tt'] = stage[temperature_col].cumsum()

        tt_threshold = stage_div[current_stage_tt]
        stage_exceedance = stage[stage[f'{current_stage_col}_tt'] >= tt_threshold]

        if not stage_exceedance.empty:
            first_exceedance_index = stage_exceedance.index[0]
            previous_day_index = first_exceedance_index - 1 if first_exceedance_index > stage.index[
                0] else first_exceedance_index

            prev_day_tt = stage.loc[previous_day_index, f'{current_stage_col}_tt']
            next_day_tt = stage_exceedance.iloc[0][f'{current_stage_col}_tt']

            if abs(prev_day_tt - tt_threshold) <= abs(next_day_tt - tt_threshold):
                stage_date = pd.to_datetime(stage.loc[previous_day_index, 'Date'])
            else:
                stage_date = pd.to_datetime(stage_exceedance.iloc[0]['Date'])

            stage_date_doy = stage_date.dayofyear
            df.loc[df['Date'] >= stage_date, current_stage_col] = stage_date_doy
        else:
            df[current_stage_col] = None
    else:
        df[current_stage_col] = None

    return df


def wheat_stage_process(df, stage_div):
    df = calculate_stage(df, stage_div, 'Emergence_date', 'tt_emergence', 'End_of_juvenile_date', 'delta_TT')
    df = calculate_stage(df, stage_div, 'End_of_juvenile_date', 'tt_end_of_juvenile', 'floral_initiation_date',
                         'delta_TT')
    df = calculate_stage(df, stage_div, 'floral_initiation_date', 'tt_floral_initiation', 'flowering_date',
                         'Crown temperature (T_c)')
    df = calculate_stage(df, stage_div, 'flowering_date', 'tt_flowering', 'heading_date', 'Crown temperature (T_c)')
    df = calculate_stage(df, stage_div, 'heading_date', 'tt_start_grain_fill', 'end_grain_fill_date',
                         'Crown temperature (T_c)')
    df = calculate_stage(df, stage_div, 'end_grain_fill_date', 'tt_end_grain_fill', 'maturity_date',
                         'Crown temperature (T_c)')

    floral_initiation_date = df['floral_initiation_date'].dropna().unique()
    if len(floral_initiation_date) > 0:
        floral_initiation_date = \
            pd.to_datetime(df[df['floral_initiation_date'] == floral_initiation_date[0]]['Date']).values[0]

        df['TT_prime'] = df.apply(
            lambda row: row['delta_TT'] if row['Date'] < floral_initiation_date else row['Crown temperature (T_c)'],
            axis=1)
        df['TT_prime'] = df['TT_prime'].cumsum().round(3)

    maturity_date = df['maturity_date'].dropna().unique()
    if len(maturity_date) > 0:
        maturity_date = pd.to_datetime(df[df['maturity_date'] == maturity_date[0]]['Date']).values[0]
        next_day = maturity_date + pd.Timedelta(days=1)
        df = df[df['Date'] <= next_day]

    return df


def assign_combination_number(Rp, Rv, Rp_values, Rv_values):
    rp_index = Rp_values.index(Rp)
    rv_index = Rv_values.index(Rv)
    return rp_index * len(Rv_values) + rv_index + 1


def stage_div_to_str(stage_div):
    return '_'.join([f'{k}{v}' for k, v in stage_div.items()])

def process_location_data(args):
    file_path, location, sowing_date, stage_div, latitude, R_p, R_v, output_folder, Rp_values, Rv_values = args
    daily_data = pd.read_csv(file_path)
    apsim_wheat = APSIMWheatPhenology(R_p=R_p, R_v=R_v, sowing_date=sowing_date)
    results_df = apsim_wheat.accumulate_daily_values(daily_data, latitude)
    result = wheat_stage_process(results_df, stage_div)

    result = result.copy()

    combination_number = assign_combination_number(R_p, R_v, Rp_values, Rv_values)
    result.loc[:, 'Parameter_set'] = combination_number

    stage_div_str = stage_div_to_str(stage_div)
    output_filename = f'{location}_{sowing_date.strftime("%Y%m%d")}_Rp{R_p}_Rv{R_v}_{stage_div_str}.csv'
    result.to_csv(os.path.join(output_folder, output_filename), index=False)

def main():
    input_folder = './input/weather'
    output_folder = './output/parameter_predict'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    stage_div = {
        'tt_emergence': 1,
        'tt_end_of_juvenile': 400.0,
        'tt_floral_initiation': 380.0,
        'tt_flowering': 60.0,
        'tt_start_grain_fill': 700,
        'tt_end_grain_fill': 35,
    }

    location_latitudes = {
        'Daegu_weather': 35.97742,
        'Jeonju_weather': 35.84092,
        'Naju_weather': 35.17294,
        'Jinju_weather': 35.16378,
        'Miryang_weather': 35.49147,
        'Suwon_weather': 37.25746,
    }

    locations = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    sowing_start = datetime(1976, 10, 10)
    sowing_end = datetime(1976, 11, 5)

    Rp_values = [round(2.0 + 0.1 * i, 1) for i in range(1)]
    Rv_values = [round(2.6 + 0.1 * i, 1) for i in range(1)]

    tasks = []
    for location in locations:
        location_name = location.split('.')[0]
        latitude = location_latitudes[location_name]

        file_path = os.path.join(input_folder, location)
        for current_sowing_date in (sowing_start + timedelta(days=n) for n in
                                    range((sowing_end - sowing_start).days + 1)):
            start_year = 1976
            end_year = 2022
            for year in range(start_year, end_year + 1):
                sowing_date = current_sowing_date.replace(year=year)
                for R_p in Rp_values:
                    for R_v in Rv_values:
                        tasks.append((file_path, location_name, sowing_date, stage_div, latitude, R_p, R_v,
                                      output_folder, Rp_values, Rv_values))

    with Pool(processes=18) as pool:
        for _ in tqdm(pool.imap_unordered(process_location_data, tasks), total=len(tasks)):
            pass

if __name__ == '__main__':
    main()
