import pandas as pd
from datetime import datetime
from thermal_time import APSIMWheatPhenology
import os

def emerg2endjuv(df, stage_div):
    juvenile = df.dropna(subset=['Emergence_date'])
    first_emergence_index = juvenile['Emergence_date'].first_valid_index()

    if first_emergence_index is not None:
        juvenile = df.loc[first_emergence_index + 1:].copy()
        juvenile['juvenile_tt'] = juvenile['delta_TT'].cumsum()

        tt_emergence = stage_div['tt_emergence']
        emergence_exceedance = juvenile[juvenile['juvenile_tt'] >= tt_emergence]

        if not emergence_exceedance.empty:
            first_exceedance_index = emergence_exceedance.index[0]
            previous_day_index = first_exceedance_index - 1 if first_exceedance_index > juvenile.index[0] else first_exceedance_index

            prev_day_tt = juvenile.loc[previous_day_index, 'juvenile_tt']
            next_day_tt = emergence_exceedance.iloc[0]['juvenile_tt']

            if abs(prev_day_tt - tt_emergence) <= abs(next_day_tt - tt_emergence):
                end_of_juvenile_date = pd.to_datetime(juvenile.loc[previous_day_index, 'Date'])
            else:
                end_of_juvenile_date = pd.to_datetime(emergence_exceedance.iloc[0]['Date'])

            end_of_juvenile_date_doy = end_of_juvenile_date.dayofyear
            df.loc[df['Date'] >= end_of_juvenile_date, 'End_of_juvenile_date'] = end_of_juvenile_date_doy
        else:
            df['End_of_juvenile_date'] = None
    else:
        df['End_of_juvenile_date'] = None

    return df

def endjuv2floral(df, stage_div):
    df = emerg2endjuv(df, stage_div)
    floral = df.dropna(subset=['End_of_juvenile_date'])
    first_index = floral['End_of_juvenile_date'].first_valid_index()

    if first_index is not None:
        floral = df.loc[first_index + 1:].copy()
        floral['floral_tt'] = floral['delta_TT'].cumsum()

        tt_end_of_juvenile = stage_div['tt_end_of_juvenile']
        juvenile_exceedance = floral[floral['floral_tt'] >= tt_end_of_juvenile]

        if not juvenile_exceedance.empty:
            first_exceedance_index = juvenile_exceedance.index[0]
            previous_day_index = first_exceedance_index - 1 if first_exceedance_index > floral.index[0] else first_exceedance_index

            prev_day_tt = floral.loc[previous_day_index, 'floral_tt']
            next_day_tt = juvenile_exceedance.iloc[0]['floral_tt']

            if abs(prev_day_tt - tt_end_of_juvenile) <= abs(next_day_tt - tt_end_of_juvenile):
                floral_initiation_date = pd.to_datetime(floral.loc[previous_day_index, 'Date'])
            else:
                floral_initiation_date = pd.to_datetime(juvenile_exceedance.iloc[0]['Date'])

            floral_initiation_date_doy = floral_initiation_date.dayofyear
            df.loc[df['Date'] >= floral_initiation_date, 'floral_initiation_date'] = floral_initiation_date_doy
        else:
            df['floral_initiation_date'] = None
    else:
        df['floral_initiation_date'] = None

    return df

def floral2flowering(df, stage_div):
    df = endjuv2floral(df, stage_div)
    flowering = df.dropna(subset=['floral_initiation_date'])
    first_index = flowering['floral_initiation_date'].first_valid_index()

    if first_index is not None:
        flowering = df.loc[first_index + 1:].copy()
        flowering['flowering_tt'] = flowering['Crown temperature (T_c)'].cumsum()

        tt_floral_initiation = stage_div['tt_floral_initiation']
        floral_exceedance = flowering[flowering['flowering_tt'] >= tt_floral_initiation]

        if not floral_exceedance.empty:
            first_exceedance_index = floral_exceedance.index[0]
            previous_day_index = first_exceedance_index - 1 if first_exceedance_index > flowering.index[0] else first_exceedance_index

            prev_day_tt = flowering.loc[previous_day_index, 'flowering_tt']
            next_day_tt = floral_exceedance.iloc[0]['flowering_tt']

            if abs(prev_day_tt - tt_floral_initiation) <= abs(next_day_tt - tt_floral_initiation):
                flowering_date = pd.to_datetime(flowering.loc[previous_day_index, 'Date'])
            else:
                flowering_date = pd.to_datetime(floral_exceedance.iloc[0]['Date'])

            flowering_date_doy = flowering_date.dayofyear
            df.loc[df['Date'] >= flowering_date, 'flowering_date'] = flowering_date_doy
        else:
            df['flowering_date'] = None
    else:
        df['flowering_date'] = None

    return df

def flowering2heading(df, stage_div):
    df = floral2flowering(df, stage_div)
    heading = df.dropna(subset=['flowering_date'])
    first_index = heading['flowering_date'].first_valid_index()

    if first_index is not None:
        heading = df.loc[first_index + 1:].copy()
        heading['heading_tt'] = heading['Crown temperature (T_c)'].cumsum()

        tt_flowering = stage_div['tt_flowering']
        flowering_exceedance = heading[heading['heading_tt'] >= tt_flowering]

        if not flowering_exceedance.empty:
            first_exceedance_index = flowering_exceedance.index[0]
            previous_day_index = first_exceedance_index - 1 if first_exceedance_index > heading.index[0] else first_exceedance_index

            prev_day_tt = heading.loc[previous_day_index, 'heading_tt']
            next_day_tt = flowering_exceedance.iloc[0]['heading_tt']

            if abs(prev_day_tt - tt_flowering) <= abs(next_day_tt - tt_flowering):
                heading_date = pd.to_datetime(heading.loc[previous_day_index, 'Date'])
            else:
                heading_date = pd.to_datetime(flowering_exceedance.iloc[0]['Date'])

            heading_date_doy = heading_date.dayofyear
            df.loc[df['Date'] >= heading_date, 'heading_date'] = heading_date_doy
        else:
            df['heading_date'] = None
    else:
        df['heading_date'] = None

    return df

def heading2endgf(df, stage_div):
    df = flowering2heading(df, stage_div)
    end_grain_fill = df.dropna(subset=['heading_date'])
    first_index = end_grain_fill['heading_date'].first_valid_index()

    if first_index is not None:
        end_grain_fill = df.loc[first_index + 1:].copy()
        end_grain_fill['end_grain_fill_tt'] = end_grain_fill['Crown temperature (T_c)'].cumsum()

        tt_start_grain_fill = stage_div['tt_start_grain_fill']
        heading_exceedance = end_grain_fill[end_grain_fill['end_grain_fill_tt'] >= tt_start_grain_fill]

        if not heading_exceedance.empty:
            first_exceedance_index = heading_exceedance.index[0]
            previous_day_index = first_exceedance_index - 1 if first_exceedance_index > end_grain_fill.index[0] else first_exceedance_index

            prev_day_tt = end_grain_fill.loc[previous_day_index, 'end_grain_fill_tt']
            next_day_tt = heading_exceedance.iloc[0]['end_grain_fill_tt']

            if abs(prev_day_tt - tt_start_grain_fill) <= abs(next_day_tt - tt_start_grain_fill):
                end_grain_fill_date = pd.to_datetime(end_grain_fill.loc[previous_day_index, 'Date'])
            else:
                end_grain_fill_date = pd.to_datetime(heading_exceedance.iloc[0]['Date'])

            end_grain_fill_date_doy = end_grain_fill_date.dayofyear
            df.loc[df['Date'] >= end_grain_fill_date, 'end_grain_fill_date'] = end_grain_fill_date_doy
        else:
            df['end_grain_fill_date'] = None
    else:
        df['end_grain_fill_date'] = None

    return df

def endgf2maturity(df, stage_div):
    df = heading2endgf(df, stage_div)
    maturity = df.dropna(subset=['end_grain_fill_date'])
    first_index = maturity['end_grain_fill_date'].first_valid_index()

    if first_index is not None:
        maturity = df.loc[first_index + 1:].copy()
        maturity['maturity_tt'] = maturity['Crown temperature (T_c)'].cumsum()

        tt_end_grain_fill = stage_div['tt_end_grain_fill']
        end_grain_fill_exceedance = maturity[maturity['maturity_tt'] >= tt_end_grain_fill]

        if not end_grain_fill_exceedance.empty:
            first_exceedance_index = end_grain_fill_exceedance.index[0]
            previous_day_index = first_exceedance_index - 1 if first_exceedance_index > maturity.index[0] else first_exceedance_index

            prev_day_tt = maturity.loc[previous_day_index, 'maturity_tt']
            next_day_tt = end_grain_fill_exceedance.iloc[0]['maturity_tt']

            if abs(prev_day_tt - tt_end_grain_fill) <= abs(next_day_tt - tt_end_grain_fill):
                maturity_date = pd.to_datetime(maturity.loc[previous_day_index, 'Date'])
            else:
                maturity_date = pd.to_datetime(end_grain_fill_exceedance.iloc[0]['Date'])

            maturity_date_doy = maturity_date.dayofyear
            df.loc[df['Date'] >= maturity_date, 'maturity_date'] = maturity_date_doy
        else:
            df['maturity_date'] = None
    else:
        df['maturity_date'] = None

    return df

def wheat_stage_process(df, stage_div):
    df = endgf2maturity(df, stage_div)
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


def main():
    latitude = 35.7281
    file_path = './input/input_weather.csv'
    daily_data = pd.read_csv(file_path)

    sowing_date = datetime(1976, 11, 5)
    apsim_wheat = APSIMWheatPhenology(R_p=1.5, R_v=1.5, sowing_date=sowing_date)
    results_df = apsim_wheat.accumulate_daily_values(daily_data, latitude)

    stage_div = {
        'tt_emergence': 1,
        'tt_end_of_juvenile': 400.0,
        'tt_floral_initiation': 555.0,
        'tt_flowering': 120.0,
        'tt_start_grain_fill': 545,
        'tt_end_grain_fill': 35,
    }

    result = wheat_stage_process(results_df, stage_div)
    output_path = './output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_filename = 'result.csv'
    result.to_csv(os.path.join(output_path, output_filename), index=False)

if __name__ == '__main__':
    main()
