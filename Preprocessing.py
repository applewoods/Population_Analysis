import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from pickle import dump

class Preprocessing():
    def __init__(self, total_urban= 229):
        self.total_urban = total_urban      # 전국 시군구 개수 변수

    ## 연령 범위 설정 함수
    ## start: 시작 나이, end: 미지막 나이
    def age_range(self, start, end):
        age_list = list()

        if end == 100:
            age_list = ['{}세'.format(age) for age in range(start, end)]
            age_list.append('100세 이상')

        else:
            age_list = ['{}세'.format(age) for age in range(start, end + 1)]

        return age_list
    
    ## 분석하는 도시갯수 확인 함수
    def check_num_urban(self, dataframe, columns= ['행정구역(시도)', '행정구역(시군구)별']):
        num_urban = len(dataframe[columns].value_counts().index)

        try:
            if num_urban != self.total_urban:
                raise Exception('시군구의 개수가 {}개가 아닙니다. (현재: {})'.format(self.total_urban, num_urban))
            
        except Exception as e:
            print(e)

    ## 연령 기준별 인구 추출 함수
    def population_split_age_range(self, dataframe, age_guide, group_col= ['행정구역(시도)', '행정구역(시군구)별']):
        age_df = dataframe[dataframe['연령별'].isin(age_guide)]
        age_df = age_df.groupby(group_col).sum()
        age_df = age_df.drop(['연령별'], axis= 1)
        age_df = age_df.reset_index(drop= False)

        self.check_num_urban(dataframe= age_df, columns= group_col)

        return age_df
    
    ## 도시별 인구비율 추출 함수
    ## numerator: 분자 / denominator: 분모
    def population_rate(self, numerator, denominator, guid_col= ['행정구역(시도)', '행정구역(시군구)별'], pre= 100):
        numerator = numerator.merge(denominator[guid_col], how= 'inner', on= guid_col)
        denominator = denominator.merge(numerator[guid_col], how= 'inner', on= guid_col)

        numerator = numerator.sort_values(guid_col)
        denominator = denominator.sort_values(guid_col)

        numerator = numerator.reset_index(drop= True)
        numerator_values = numerator.drop(guid_col, axis= 1)

        denominator = denominator.reset_index(drop= True)
        denominator_values = denominator.drop(guid_col, axis= 1)

        pop_rate_df = numerator_values.div(denominator_values)

        try:
            if str(pre).isnumeric():
                pop_rate_df = np.round(pop_rate_df * pre, 3)
                pop_rate_df = pd.concat([numerator[guid_col], pop_rate_df], axis= 1)
                return pop_rate_df

            else:
                raise Exception('숫자를 입력하세요.')
        
        except Exception as e:
            print(e)

    ## 연평균인구증감율
    def cagr_func(dataframe, start_year, end_year, columns= ['행정구역(시도)', '행정구역(시군구)별']):
        start_year = str(start_year) + ' 년'
        end_year = str(end_year) + ' 년'

        tmp_df = dataframe[columns]
        tmp_df[end_year] = dataframe[[start_year, end_year]].apply(lambda x: np.round(np.power(x[end_year] / x[start_year], 1 / (int(end_year.split(' ')[0]) - int(start_year.split(' ')[0]))) - 1, 4), axis= 1)

        return tmp_df

    ## 분석용 데이터프레임 만드는 함수
    def mk_analysis_dataframe(self, original_dataframe, dataframe, cat_name, columns= ['행정구역(시도)', '행정구역(시군구)별'], how= 'inner'):
        add_dataframe = self.series_to_dataframe(dataframe= dataframe, cat_name= cat_name, columns_= columns)
        result_df = self.merge_dataframe(original_dataframe= original_dataframe, add_dataframe= add_dataframe, columns= columns, how= how)

        return result_df

    def series_to_dataframe(self, dataframe, cat_name, columns_= ['행정구역(시도)', '행정구역(시군구)별']):
        cat_len = len(dataframe.drop(columns_, axis= 1).columns)
        series_frame = pd.Series(dataframe[dataframe.columns[cat_len * -1:]].values.tolist(), name= cat_name).to_frame()
        col_frame = dataframe[columns_]

        return pd.concat([col_frame, series_frame], axis= 1)
    
    def merge_dataframe(self, original_dataframe, add_dataframe, columns= ['행정구역(시도)', '행정구역(시군구)별'], how= 'inner'):
        if len(original_dataframe) == 0:
            return add_dataframe
        
        else:
            new_dataframe = original_dataframe.merge(add_dataframe, on= columns, how= how)
            self.check_num_urban(original_dataframe, columns= columns)

            return new_dataframe
        
    def MinMaxScaler_func(self, dataframe, title, save_path = './', colums= ['행정구역(시도)', '행정구역(시군구)별'], min_value= 0, max_value= 1):
        tmp_df = dataframe.drop(colums, axis= 1)
        X = np.array(tmp_df.values)

        scaler = MinMaxScaler(feature_range=(min_value, max_value))
        scaler.fit(X)
        scaler_values = scaler.transform(X)

        scaler_df = pd.DataFrame(scaler_values, columns= tmp_df.columns)
        dump(scaler, open(save_path + 'MinMaxScaler_{}.pkl'.format(title), 'wb'))
        
        return pd.concat([dataframe[colums], scaler_df], axis= 1)