import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def reference_ob_preprocess(df):
    df['지역'] = df['지역'].replace({
        '수원(중부작물부)': 'Suwon',
        '밀양(남부작물부)': 'Miryang',
        '대구(경북도원)': 'Daegu',
        '전주(국립식량과학원)': 'Jeonju',
        '나주(전남도원)': 'Naju',
        '진주(경남도원)': 'Jinju',
    })

    df['출수기'] = pd.to_datetime(df['출수기'])
    df['파종기'] = pd.to_datetime(df['파종기'])
    df['성숙기'] = pd.to_datetime(df['성숙기'])
    df['생육재생기'] = pd.to_datetime(df['생육재생기'])
    df['최고분얼기'] = pd.to_datetime(df['최고분얼기'])
    df['파종기_DOY'] = df['파종기'].dt.dayofyear
    df['성숙기_DOY'] = df['성숙기'].dt.dayofyear
    df['출수기_DOY'] = df['출수기'].dt.dayofyear
    df['생육재생기_DOY'] = df['생육재생기'].dt.dayofyear
    df['최고분얼기_DOY'] = df['최고분얼기'].dt.dayofyear

    def classify_sowing_season(sowing_date):
        if 274 <= sowing_date <= 283:
            return 'oct_first'
        elif 284 <= sowing_date <= 295:
            return 'oct_mid'
        elif 296 <= sowing_date <= 305:
            return 'oct_last'
        elif 306 <= sowing_date <= 315:
            return 'nov_first'
        elif 316 <= sowing_date <= 325:
            return 'nov_mid'
        else:
            return 'other'

    df['sowing_season'] = df['파종기_DOY'].apply(classify_sowing_season)
    df['last_day_doy'] = df['year'].apply(lambda x: 366 if (x % 4 == 0 and x % 100 != 0) or (x % 400 == 0) else 365)
    df['sow2flowering'] = df['last_day_doy'] - df['파종기_DOY'] + df['출수기_DOY']
    df['sow2maturing'] = df['last_day_doy'] - df['파종기_DOY'] + df['성숙기_DOY']

    return df


def preprocessing(df, ob):
    observed = reference_ob_preprocess(ob)
    observed = observed[observed['지역'] != 'Jinju']
    # observed = observed.drop(columns=['생육재생기', '최고분얼기', '생육재생기_DOY', '최고분얼기_DOY'])
    observed = observed.drop_duplicates(subset=['지역', 'year', '파종기_DOY', '성숙기_DOY', '출수기_DOY', '생육재생기_DOY'])
    data = pd.merge(df, observed, left_on=['Site', 'Year', 'sowing_date'],right_on=['지역', 'year', '파종기_DOY'], how='right')

    return data


def plot_results(data):
    varieties = data['품종'].unique()
    num_varieties = len(varieties)
    fig, axes = plt.subplots(num_varieties, 3, figsize=(14, 7 * num_varieties), dpi=150)  # 품종 수에 따른 서브플롯 생성

    if num_varieties == 1:
        axes = np.expand_dims(axes, axis=0)  # 품종이 하나인 경우에도 2D 배열로 처리

    for i, variety in enumerate(varieties):
        variety_data = data[data['품종'] == variety]

        # 첫 번째 서브플롯: 최고분얼기_DOY vs floral_initiation_date
        floral_data = variety_data.dropna(subset=['최고분얼기_DOY', 'floral_initiation_date'])
        x1 = floral_data['최고분얼기_DOY']
        y1 = floral_data['floral_initiation_date']
        model1 = np.polyfit(x1, y1, 1)
        y1_pred = np.polyval(model1, x1)

        r2_1 = r2_score(y1, y1_pred)
        rmse_1 = np.sqrt(mean_squared_error(y1, y1_pred))

        sns.regplot(data=floral_data, x='최고분얼기_DOY', y='floral_initiation_date', ax=axes[i, 0],
                    scatter_kws={'color': 'black', 's': 100, 'alpha': 0.6},
                    line_kws={'color': 'red'})

        min_val_1 = min(x1.min() - 5, y1.min() - 5)
        max_val_1 = max(x1.max() + 5, y1.max() + 5)
        axes[i, 0].plot([min_val_1, max_val_1], [min_val_1, max_val_1], 'k--')

        axes[i, 0].set_xlim(min_val_1, max_val_1)
        axes[i, 0].set_ylim(min_val_1, max_val_1)
        axes[i, 0].set_aspect('equal', 'box')

        text_str_1 = f'R² = {r2_1:.2f}\nRMSE = {rmse_1:.2f}'
        axes[i, 0].text(0.05, 0.95, text_str_1, transform=axes[i, 0].transAxes, fontsize=12, verticalalignment='top')

        axes[i, 0].set_title(f'{variety} - 최고분얼기')
        axes[i, 0].set_xlabel('최고분얼기_DOY')
        axes[i, 0].set_ylabel('floral_initiation_date')

        # 두 번째 서브플롯: 출수기_DOY vs heading_date
        heading_data = variety_data.dropna(subset=['출수기_DOY', 'heading_date'])
        x2 = heading_data['출수기_DOY']
        y2 = heading_data['heading_date']
        model2 = np.polyfit(x2, y2, 1)
        y2_pred = np.polyval(model2, x2)

        r2_2 = r2_score(y2, y2_pred)
        rmse_2 = np.sqrt(mean_squared_error(y2, y2_pred))

        sns.regplot(data=heading_data, x='출수기_DOY', y='heading_date', ax=axes[i, 1],
                    scatter_kws={'color': 'black', 's': 100, 'alpha': 0.6},
                    line_kws={'color': 'red'})

        min_val_2 = min(x2.min() - 5, y2.min() - 5)
        max_val_2 = max(x2.max() + 5, y2.max() + 5)
        axes[i, 1].plot([min_val_2, max_val_2], [min_val_2, max_val_2], 'k--')

        axes[i, 1].set_xlim(min_val_2, max_val_2)
        axes[i, 1].set_ylim(min_val_2, max_val_2)
        axes[i, 1].set_aspect('equal', 'box')

        text_str_2 = f'R² = {r2_2:.2f}\nRMSE = {rmse_2:.2f}'
        axes[i, 1].text(0.05, 0.95, text_str_2, transform=axes[i, 1].transAxes, fontsize=12, verticalalignment='top')

        axes[i, 1].set_title(f'{variety} - 출수기')
        axes[i, 1].set_xlabel('출수기_DOY')
        axes[i, 1].set_ylabel('heading_date')

        # 세 번째 서브플롯: 성숙기_DOY vs maturity_date
        maturity_data = variety_data.dropna(subset=['성숙기_DOY', 'maturity_date'])
        x3 = maturity_data['성숙기_DOY']
        y3 = maturity_data['maturity_date']
        model3 = np.polyfit(x3, y3, 1)
        y3_pred = np.polyval(model3, x3)

        r2_3 = r2_score(y3, y3_pred)
        rmse_3 = np.sqrt(mean_squared_error(y3, y3_pred))

        sns.regplot(data=maturity_data, x='성숙기_DOY', y='maturity_date', ax=axes[i, 2],
                    scatter_kws={'color': 'black', 's': 100, 'alpha': 0.6},
                    line_kws={'color': 'red'})

        min_val_3 = min(x3.min() - 5, y3.min() - 5)
        max_val_3 = max(x3.max() + 5, y3.max() + 5)
        axes[i, 2].plot([min_val_3, max_val_3], [min_val_3, max_val_3], 'k--')

        axes[i, 2].set_xlim(min_val_3, max_val_3)
        axes[i, 2].set_ylim(min_val_3, max_val_3)
        axes[i, 2].set_aspect('equal', 'box')

        text_str_3 = f'R² = {r2_3:.2f}\nRMSE = {rmse_3:.2f}'
        axes[i, 2].text(0.05, 0.95, text_str_3, transform=axes[i, 2].transAxes, fontsize=12, verticalalignment='top')

        axes[i, 2].set_title(f'{variety} - 성숙기')
        axes[i, 2].set_xlabel('성숙기_DOY')
        axes[i, 2].set_ylabel('maturity_date')

    plt.tight_layout()
    plt.show()




def main():
    df = pd.read_csv('./parameter_scenario_output_very_early16.csv', parse_dates=['Date'])
    observed = pd.read_csv('./input/reference_observed.csv')



    data = preprocessing(df, observed)
    plot_results(data)


if __name__ == '__main__':
    main()