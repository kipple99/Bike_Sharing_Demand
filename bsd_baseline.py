#!/usr/bin/python
# -*- coding : utf-8 -*-
'''
 @author : yjjo
'''
''' install '''

''' import '''
import warnings
from logging.config import dictConfig
import logging
import inspect
import datetime
import traceback
import time
import sys
import os
import re
import pandas as pd
import numpy as np
import pandas_profiling
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from prophet.utilities import regressor_coefficients

''' log '''
warnings.filterwarnings(action = 'ignore') 
filePath = os.getcwd()
fileName = re.split('[.]', inspect.getfile(inspect.currentframe()))[0]

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s --- %(message)s',
        }
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': '{}\\logs\\{}_{}.log'.format(filePath, fileName, re.sub('-', '', str(datetime.date.today()))),
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file']
    }
})

# 로그 함수
def log(msg):
    logging.info(msg)
    print(msg)

''' main function '''
def main():
    # 경로(path) 설정
    path = 'C:\\Users\\yjjo\\OneDrive - 데이터누리\\yjjo\\8_내부 과제\\6_비즈팀 역량증진 프로젝트'  # 기본 경로
    data_path = '{}\\2_data'.format(path)                                                           # 데이터 경로
    csv_path = '{}\\1_csv'.format(data_path)                                                        # csv 파일 경로
    plt_path = '{}\\2_plt'.format(data_path)                                                        # 시각화 자료 경로
    
    # csv 파일 수집
    train_df = read_csv('{}\\train.csv'.format(csv_path), parse_date = ['datetime'])            # train csv
    test_df = read_csv('{}\\test.csv'.format(csv_path), parse_date = ['datetime'])              # test csv
    submission_df = read_csv('{}\\sampleSubmission.csv'.format(csv_path), parse_date = ['datetime'])  # test csv
    
    proc_df = pd.concat([train_df, test_df], axis = 0).sort_values('datetime').reset_index(drop = True) # axis = 0, axis = 1
    
    # datetime 정리
    proc_df['date'] = proc_df['datetime'].apply(lambda x : re.split(' ', str(x))[0])    # YYYY-MM-DD 설정
    proc_df['year'] = proc_df['datetime'].dt.year                                       # year 설정
    proc_df['month'] = proc_df['datetime'].dt.month                                     # month 설정
    proc_df['day'] = proc_df['datetime'].dt.day                                         # day 설정
    proc_df['weekday'] = proc_df['datetime'].apply(lambda x : x.weekday())              # 요일 설정
    proc_df['weekday'] = proc_df['weekday'].apply(set_weekday)                          # 숫자 -> 글자 변경
    proc_df['hour'] = proc_df['datetime'].dt.hour                                       # 숫자 -> 글자 변경
    proc_df = proc_df[['datetime', 'date', 'year', 'month', 'day', 'weekday', 'hour', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']]
    
    # 숫자 변수 pandas profiling
    profiling = proc_df.iloc[:, 11:].profile_report()
    profiling.to_file('{}\\{}.html'.format(plt_path, 'profiling'))
    
    # 칼럼 설명
    # datetime      : 일자 + 시간
    # date          : 일자
    # year          : 연도
    # month         : 월
    # day           : 일
    # weekday       : 요일
    # season        : 시즌 분류
    # holiday       : 휴일 여부
    # workingday    : 근무일 여부
    # weather       : 날씨
    # temp          : 기온
    # atemp         : 체감온도
    # humidity      : 습도
    # windspeed     : 풍속
    # casual        : 비등록 회원 이용자 수
    # registered    : 등록 회원 이용자 수
    # count         : 이용자 수
    
    # 칼럼별 시각화
    # x : 시간흐름, y : 칼럼
    x = 'datetime'
    for jdx, col in enumerate(proc_df.columns[11:]):
        plt.plot(proc_df[x], proc_df[col], label = col)
        plt.xticks(rotation = 20)
        plt.legend()
        plt.savefig('{}\\{}_{}.png'.format(plt_path, x, col))
        plt.cla()
    
    # x : 요일, y : 이용자 수
    x = 'hour'
    for jdx, col in enumerate(proc_df.columns[-3:]):
        plt.plot(proc_df[x].unique(), proc_df[proc_df['workingday'] == 0].groupby(x).mean()[col], label = 'not workingday')
        plt.plot(proc_df[x].unique(), proc_df[proc_df['workingday'] == 1].groupby(x).mean()[col], label = 'workingday')
        plt.xticks(rotation = 20)
        plt.legend()
        plt.savefig('{}\\workingday_{}_{}.png'.format(plt_path, x, col))
        plt.cla()
    
    # month별 season 분포
    proc_df[['month', 'season']].drop_duplicates(subset = ['month', 'season']).reset_index(drop = True)
    
    proc_df[(proc_df['month'] == 12) & (proc_df['season'] == 1)]
    
    # weekday별 workingday 분포
    proc_df[['weekday', 'workingday']].drop_duplicates(subset = ['weekday', 'workingday']).reset_index(drop = True)
    
    # date별 weather 분포
    proc_df[['date', 'weather']].drop_duplicates(subset = ['date', 'weather']).reset_index(drop = True)
    
    # 데이터 가공
    # season 데이터프레임 생성
    season_df = proc_df[['date', 'season']].drop_duplicates(subset = ['date', 'season']).reset_index(drop = True)
    season_df[season_df['date'].str.contains('2011-04-15')]
    season_df['season'] = season_df['season'].apply(lambda x : 'season_{}'.format(x))
    season_df['upper_window'] = 0
    season_df['lower_window'] = 0
    season_df = season_df.rename({'date' : 'ds', 'season' : 'holiday'}, axis = 1)
    
    # holiday 데이터프레임 생성
    hldy_list = list(proc_df[proc_df['holiday'] == 1]['date'].unique())
    
    hldy_df = pd.DataFrame({
        'ds' : hldy_list,
        'holiday' : 'holiday', 
        'upper_window' : 0,
        'lower_window' : 0,
        })
    
    # holidays 데이터프레임 생성
    holidays_df = pd.concat([season_df, hldy_df], axis = 0).reset_index(drop = True)
    holidays_df.to_csv('{}\\{}.csv'.format(csv_path, 'holidays_dataframe'), index = False, encoding = 'UTF-8-SIG')
    
    # casual, registered 데이터 분리 및 불필요 칼럼 drop
    casual_df = proc_df.drop('registered', axis = 1).rename({'datetime' : 'ds', 'casual' : 'y'}, axis = 1)
    registered_df = proc_df.drop('casual', axis = 1).rename({'datetime' : 'ds', 'registered' : 'y'}, axis = 1)
    
    drop_list = ['date', 'year', 'month', 'day', 'weekday', 'hour', 'season', 'holiday', 'workingday', 'count'] # drop 리스트
    # casual
    train_df_casual = casual_df[casual_df['day'] <= 19].drop(drop_list, axis = 1).reset_index(drop = True)
    test_df_casual = casual_df[casual_df['day'] >= 20].drop(drop_list + ['y'], axis = 1).reset_index(drop = True)
    
    # registered
    train_df_registered = registered_df[registered_df['day'] <= 19].drop(drop_list, axis = 1).reset_index(drop = True)
    test_df_registered = registered_df[registered_df['day'] >= 20].drop(drop_list + ['y'], axis = 1).reset_index(drop = True)
    
    # # 튜닝
    # # casual
    # optimum_df = pd.DataFrame([], columns = ['num', 'changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode', 'holidays_prior_scale', 'rmse'])
    
    # ttrain = train_df_casual[train_df_casual['day'] <= 14]
    # ttest = pd.DataFrame(train_df_casual[train_df_casual['day'] >= 15].drop(['y'], axis = 1).reset_index(drop = True))
    # ttest_y = train_df_casual[train_df_casual['day'] >= 15]['y'].reset_index(drop = True)

    # rmse_df, min_rmse = tuning(ttrain, ttest, ttest_y, [0.001, 0.01, 0.1, 0.5], [0.01, 0.1, 1, 10], ['additive', 'multiplicative'], [0.01, 0.1, 1, 10], holidays_df)
    # print(min_rmse)
    # rmse_df.to_csv('{}\\{}.csv'.format(csv_path, 'rmse_casual'), encoding = 'UTF-8', index = False)
    
    # train_df_casual = train_df_casual.drop(drop_list, axis = 1)
    
    # # registered
    # optimum_df = pd.DataFrame([], columns = ['num', 'changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode', 'holidays_prior_scale', 'rmse'])
    
    # ttrain = train_df_registered[train_df_registered['day'] <= 14]
    # ttest = pd.DataFrame(train_df_registered[train_df_registered['day'] >= 15].drop(['y'], axis = 1).reset_index(drop = True))
    # ttest_y = train_df_registered[train_df_registered['day'] >= 15]['y'].reset_index(drop = True)

    # rmse_df, min_rmse = tuning(ttrain, ttest, ttest_y, [0.001, 0.01, 0.1, 0.5], [0.01, 0.1, 1, 10], ['additive', 'multiplicative'], [0.01, 0.1, 1, 10], holidays_df)
    # print(min_rmse)
    # rmse_df.to_csv('{}\\{}.csv'.format(csv_path, 'rmse_registered'), encoding = 'UTF-8', index = False)
    
    # train_df_registered = train_df_registered.drop(drop_list, axis = 1)
    
    # Prophet 모델 생성
    # casual 예측
    model_casual = Prophet(
        # growth
        # linear or logistic
        # growth = 'linear', # default : 'linear'

        # trend
        # 값이 높을수록 trend를 유연하게 감지
        changepoint_prior_scale = 0.1,
        # change point 명시
        # changepoints=['2021-02-08', '2021-05-13'],

        # seasonality
        # # 연 계절성
        # yearly_seasonality = 10, # default : 10
        # # 주 계절성
        # weekly_seasonality = 10, # default : 10
        # # 일 계절성
        # daily_seasonality = 10,
        # 계절성 반영 강도
        seasonality_prior_scale = 1,
        # additive or multiplicative
        seasonality_mode = 'multiplicative',

        # holidays
        holidays = holidays_df,
        # holiday 반영 강도
        holidays_prior_scale = 0.01,
    ).add_regressor('temp')\
        .add_regressor('atemp')\
            .add_regressor('humidity')\
                .add_regressor('windspeed')
    
    # model fit
    model_casual.fit(train_df_casual)
    
    # 예측
    forecast_casual = model_casual.predict(test_df_casual)
    forecast_casual['yhat'] = forecast_casual['yhat'].apply(lambda x : 0 if x < 0 else int(x))  # 예측 이용자 수 0 미만 -> 0 변경
    model_casual.plot(forecast_casual)
    plt.savefig('{}\\{}.png'.format(plt_path, 'forecast_casual'))   #  결과 시각화
    model_casual.plot_components(forecast_casual)
    plt.savefig('{}\\{}.png'.format(plt_path, 'model_plt_componets_casual'))
    
    # regressor_coefficients(train_df_casual)
    
    # registered 예측
    model_registered = Prophet(
        # growth
        # linear or logistic
        # growth = 'linear', # default : 'linear'

        # trend
        # 값이 높을수록 trend를 유연하게 감지
        changepoint_prior_scale = 0.5,
        # change point 명시
        # changepoints=['2021-02-08', '2021-05-13'],

        # seasonality
        # # 연 계절성
        # yearly_seasonality = 10, # default : 10
        # # 주 계절성
        # weekly_seasonality = 10, # default : 10
        # # 일 계절성
        # daily_seasonality = 10,
        # 계절성 반영 강도
        seasonality_prior_scale = 1,
        # additive or multiplicative
        seasonality_mode = 'multiplicative',

        # holidays
        holidays = holidays_df,
        # holiday 반영 강도
        holidays_prior_scale = 0.1,
    ).add_regressor('temp')\
        .add_regressor('atemp')\
            .add_regressor('humidity')\
                .add_regressor('windspeed')
    
    # model fit
    model_registered.fit(train_df_registered)
    
    # 예측
    forecast_registered = model_registered.predict(test_df_registered)
    forecast_registered['yhat'] = forecast_registered['yhat'].apply(lambda x : 0 if x < 0 else int(x))  # 예측 이용자 수 0 미만 -> 0 변경
    model_registered.plot(forecast_registered)
    plt.savefig('{}\\{}.png'.format(plt_path, 'forecast_registered'))   #  결과 시각화
    model_registered.plot_components(forecast_registered)
    plt.savefig('{}\\{}.png'.format(plt_path, 'model_plt_componets_registered'))
    
    # regressor_coefficients(model_registered)
    
    # casual + predict
    final_df = forecast_casual[['ds', 'yhat']].merge(forecast_registered[['ds', 'yhat']], 'left', 'ds')
    final_df['count'] = final_df['yhat_x'] + final_df['yhat_y']
    
    # 제출용 파일 생성
    submission_df['count'] = final_df['count']
    submission_df.to_csv('{}\\{}.csv'.format(csv_path, 'submission'), index = False, encoding = 'UTF-8-SIG')
    
''' functions '''
# csv 파일 수집 함수
def read_csv(path, parse_date):
    try:
        log('#### Read File {}'.format(path))
        return pd.read_csv(path, parse_dates = parse_date)
    except:
        log('######## Read File Error')
        log(traceback.format_exc())

# 요일 숫자 -> 글자 변경 함수
def set_weekday(num):
    try:
        dateDict = {0 : '월요일', 1 : '화요일', 2 : '수요일', 3 : '목요일', 4 : '금요일', 5 : '토요일', 6 : '일요일'}
        
        if num in dateDict.keys():
            return dateDict[num]
    except:
        log('######## Set weekday Error')
        log(traceback.format_exc())

# 튜닝
def tuning(train, test, test_y, changepoint_prior_scale, seasonality_prior_scale, seasonality_mode, holidays_prior_scale, holidays_df):
    headers = ['changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode', 'holidays_prior_scale', 'rmse']
    rmse_df = pd.DataFrame([], columns = headers)

    for cps in changepoint_prior_scale:
        for sps in seasonality_prior_scale:
            for sm in seasonality_mode:
                for hps in holidays_prior_scale:
                    model = Prophet(
                        changepoint_prior_scale = cps,
                        seasonality_prior_scale = sps,
                        seasonality_mode = sm,
                        holidays_prior_scale = hps,
                        holidays = holidays_df
                    ).add_regressor('temp')\
                        .add_regressor('atemp')\
                            .add_regressor('humidity')\
                                .add_regressor('windspeed')

                    model.fit(train)

                    past = model.predict(test)
                    rmse = RMSE(test_y, past['yhat'])
                    rmse_list = [cps, sps, sm, hps, rmse]
                    rmse_df = rmse_df.append(pd.Series(rmse_list, index = headers), ignore_index = True)
                    print('rmse_df')
                    print(rmse_df)

    min_rmse = rmse_df[rmse_df['rmse'] == rmse_df['rmse'].min()].reset_index(drop = True)

    return rmse_df, min_rmse

def RMSE(test, pred):
    try:
        mse = mean_squared_error(test, pred)
        return np.sqrt(mse)
    except:
        log('######## RMSE Error')
        log(traceback.format_exc())

''' main '''
if __name__ == '__main__':
    # 시간 계산
    start_time = time.time()

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    main()

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    log('#### ===================== Time =====================')
    log('#### {:.3f} seconds'.format(time.time() - start_time))
    log('#### ================================================')