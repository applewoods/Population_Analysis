import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from pickle import dump

from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.metrics import dtw
from yellowbrick.cluster import SilhouetteVisualizer

import geopandas as gpd

# from tqdm._tqdm_notebook import tqdm_notebook

import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette('Set3')
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class KMeans_Clustering():
    def __init__(self, timeseries_data, analysis_columns_dict, save_path, total_dataframe, men_dataframe, women_dataframe):
        self.timeseries_data = timeseries_data
        self.analysis_columns_dict = analysis_columns_dict
        self.save_path = save_path

        self.total_df = total_dataframe
        self.men_df = men_dataframe
        self.women_df = women_dataframe

        self.centerios_total_df = pd.DataFrame()

    ## 시계열 클러스터링 분석
    def preprocessing(self):
        timesereis_preprocessing_data = np.array(self.timeseries_data[list(self.analysis_columns_dict.keys())].values.tolist())
        timeseries_flatten = np.reshape(timesereis_preprocessing_data, (timesereis_preprocessing_data.shape[0], -1))

        return timeseries_flatten
    
    def find_best_K(self, min_num= 2, max_num= 20, metric= 'euclidean'):
        self.metric = metric
        self.timeseries_flatten = self.preprocessing()
        self.min_num, self.max_num= min_num, max_num + 1
        self.inertia_, self.silhouette_score_, self.models_ = list(), list(), list()

        for k in range(self.min_num, self.max_num):
            ts_kmeans = TimeSeriesKMeans(n_clusters= k, verbose=False, random_state=42, n_jobs= -1, max_iter= 100, metric= metric)
            y_pred = ts_kmeans.fit_predict(self.timeseries_flatten)

            # K-Means Model
            self.models_.append(ts_kmeans)

            # Inertia Score
            self.inertia_.append(ts_kmeans.inertia_)

            # Silhouette Score
            s_score = silhouette_score(X= self.timeseries_flatten, labels= y_pred, metric= metric, verbose=False, random_state=42, n_jobs= -1)
            self.silhouette_score_.append(s_score)

    def find_best_K_vis(self):
        x_range = [x for x in range(self.min_num, self.max_num)]

        # Elbow Methods
        plt.figure()
        plt.plot(x_range, self.inertia_)
        plt.xticks(x_range)
        plt.xlabel('Num of Cluster')
        plt.ylabel('Inertia')
        plt.savefig(self.save_path + 'Inertia.png')

        # Silhouette Score
        plt.figure()
        plt.bar(x_range, self.silhouette_score_)
        plt.xticks(x_range)
        plt.xlabel('Num of Cluster')
        plt.ylabel('Silhouette Score')
        plt.savefig(self.save_path + 'Silhouette_Score_Bar.png')

    def best_K_silhouettes_vis(self):
        self.best_model = self.models_[self.best_K - self.min_num]
        self.best_silhouettes_score = self.silhouette_score_[self.best_K - self.min_num]

        # 최적의 클러스터의 Silhouette 분포 시각화
        silhouette_vis = SilhouetteVisualizer(
            self.best_model,
            title= 'Silhouette Plot for Cluster {}'.format(self.best_K)
        )
        silhouette_vis.fit(self.timeseries_flatten)
        plt.ylabel('Cluster Label')
        plt.xlabel('Silhouette Coefficient Values')
        plt.savefig(self.save_path + 'Silhouette Plot.png')

    def best_K_distribution_vis(self):
        y_pred = self.best_model.fit_predict(self.timeseries_flatten)
        pred_series = pd.Series(y_pred, name= 'Cluster Pred').to_frame()
        self.pre_df = pd.concat([self.timeseries_data, pred_series], axis= 1)
        self.pre_df['Cluster Pred'] = self.pre_df['Cluster Pred'] + 1
        print(len(self.pre_df[['행정구역(시도)', '행정구역(시군구)별']].value_counts().index))

        labels = self.pre_df['Cluster Pred'].value_counts().sort_index().index
        frequency = self.pre_df['Cluster Pred'].value_counts().sort_index().values

        fig = plt.figure()
        ax = fig.add_subplot()                                      # 프레임 생성

        pie = ax.pie(
            frequency,
            counterclock= False,
            startangle=90,
            wedgeprops= {'width' : 0.5}
        )

        total = np.sum(frequency)                                   # 빈도수 총합
        
        sum_pct = 0                                                 # 백분율 초기값
        for i,l in enumerate(labels):
            ang1, ang2 = pie[0][i].theta1, pie[0][i].theta2         # 각1, 각2
            r = pie[0][i].r                                         # 원의 반지름
            
            x = ((r+0.5)/2)*np.cos(np.pi/180*((ang1+ang2)/2))       # 정중앙 x좌표
            y = ((r+0.5)/2)*np.sin(np.pi/180*((ang1+ang2)/2))       # 정중앙 y좌표
            
            if i < len(labels) - 1:
                sum_pct += float(f'{frequency[i]/total*100:.1f}')   # 백분율을 누적한다.
                ax.text(x,y,f'{frequency[i]}\n({frequency[i]/total*100:.1f}%)',ha='center',va='center')
            else:
                ax.text(x,y,f'{frequency[i]}\n({100-sum_pct:.1f}%)',ha='center',va='center')

        plt.legend(['도시유형{}'.format(idx) for idx in labels])
        plt.savefig(self.save_path + 'Num_of_Cluster.png')

    def best_K_vis(self, best_K= 3):
        self.best_K = best_K
        self.best_K_silhouettes_vis()
        self.best_K_distribution_vis()

        self.cluster_labels = self.best_model.labels_

        self.cluster_center = self.best_model.cluster_centers_
        self.cluster_center = np.array(self.cluster_center).reshape(self.best_K, len(self.analysis_columns_dict), len(self.timeseries_data[self.timeseries_data.columns[-1]][0]))

        dump(self.best_model, open(self.save_path + 'TimeSereies_Cluster_Best_Model.pkl', 'wb'))

    ### 인구 피라미드 시각화
    def gender_population_by_year(self, year= 2023):
        year = str(year) + ' 년'
        p_men_df = self.men_df[['행정구역(시도)', '행정구역(시군구)별', '연령별', year]].merge(self.pre_df[['행정구역(시도)', '행정구역(시군구)별', 'Cluster Pred']], on= ['행정구역(시도)', '행정구역(시군구)별'], how= 'outer')
        self.p_group_men_df = self.gender_preprocess(p_men_df, year, False)

        p_women_df = self.women_df[['행정구역(시도)', '행정구역(시군구)별', '연령별', year]].merge(self.pre_df[['행정구역(시도)', '행정구역(시군구)별', 'Cluster Pred']], on= ['행정구역(시도)', '행정구역(시군구)별'], how= 'outer')
        self.p_group_women_df = self.gender_preprocess(p_women_df, year, False)

        p_total_df = self.total_df[['행정구역(시도)', '행정구역(시군구)별', year]].merge(self.pre_df[['행정구역(시도)', '행정구역(시군구)별', 'Cluster Pred']], on= ['행정구역(시도)', '행정구역(시군구)별'], how= 'outer')
        self.p_group_total_df = self.gender_preprocess(p_total_df, year, True)
        self.p_group_total_df = self.p_group_total_df.rename(columns= {
            year : 'Total'
        })

    def gender_preprocess(self, dataframe, year= 2023, is_total= False):
        year = str(year) + ' 년'
        group_df = pd.DataFrame()

        if is_total:
            group_df = dataframe.groupby(['Cluster Pred']).sum().drop(['행정구역(시도)', '행정구역(시군구)별'], axis= 1)
            group_df = group_df.reset_index(drop= False)

        else:
            group_df = dataframe.groupby(['Cluster Pred', '연령별']).sum().drop(['행정구역(시도)', '행정구역(시군구)별'], axis= 1)
            group_df = group_df.reset_index(drop= False)

        return group_df
    
    def gender_add_total(self, year= 2023):
        year = str(year) + ' 년'

        self.p_group_women_df = self.p_group_women_df.merge(self.p_group_total_df, on= ['Cluster Pred'], how= 'inner')
        self.p_group_women_df['Percentage'] = self.p_group_women_df[[year, 'Total']].apply(lambda x: np.round(x[0]/x[1]*100, 2), axis= 1)
        # p_group_women_df.describe()

        self.p_group_men_df = self.p_group_men_df.merge(self.p_group_total_df, on= ['Cluster Pred'], how= 'inner')
        self.p_group_men_df['Percentage'] = self.p_group_men_df[[year, 'Total']].apply(lambda x: np.round(x[0]/x[1]*100, 2), axis= 1)
        # p_group_men_df.describe()

    def population_pyramid_vis(self, cluster= 1, year= 2023, show_rate= True):
        self.gender_population_by_year(year= year)

        if show_rate:
            self.gender_add_total(year= year)

        age_list = [str(x) + '세' for x in range(100)]
        age_list.append('100세 이상')
        age_series = pd.Series(age_list, name= '연령별').to_frame()

        women_cluster = self.p_group_women_df[self.p_group_women_df['Cluster Pred'] == cluster]
        women_cluster = age_series.merge(women_cluster, on= ['연령별'], how= 'inner')

        men_cluster = self.p_group_men_df[self.p_group_men_df['Cluster Pred'] == cluster]
        men_cluster = age_series.merge(men_cluster, on= ['연령별'], how= 'inner')

        if show_rate:
            men_cluster['Percentage'] = men_cluster['Percentage'] * (-1)

        else:
            men_cluster[str(year) + ' 년'] = men_cluster[str(year) + ' 년'] * (-1)

        plt.figure(figsize=(8,8))
        
        if show_rate:
            plt.barh(range(101), men_cluster['Percentage'], label='남성', color= '#1f77b4')
            plt.barh(range(101), women_cluster['Percentage'], label='여성', color= '#d62728')

        else:
            plt.barh(range(101), men_cluster[str(year) + ' 년'], label='남성', color= '#1f77b4')
            plt.barh(range(101), women_cluster[str(year) + ' 년'], label='여성', color= '#d62728')
        
        plt.grid(axis='y', linestyle= '--')
        plt.grid(axis='x', linestyle= 'None')
        plt.yticks(np.arange(0, 101, 10))
        plt.title('Cluster {}'.format(cluster))
        plt.legend()
        plt.savefig(self.save_path + '인구피라미드_Cluster{}_{}.png'.format(cluster, year))


    ## 지도 기반 시각화
    def geo_vis(self, shp_path = './Data/02Map/행정구역지도(전국)/sig.shp'):
        korea = gpd.read_file(shp_path, encoding= 'euckr')
        korea = korea.astype({'SIG_CD' : int})

        tmp_korea = pd.read_excel('./Data/행정구역Geo2.xlsx')
        tmp_korea.dropna(inplace= True)
        tmp_korea = tmp_korea.astype({'SIG_CD' : int})
        korea = korea[['SIG_CD', 'SIG_ENG_NM', 'SIG_KOR_NM', 'geometry']].merge(tmp_korea[['행정구역(시도)', '행정구역(시군구)별', 'SIG_CD', 'Group']], on= 'SIG_CD', how= 'inner')
        korea = korea[['SIG_CD', '행정구역(시도)', '행정구역(시군구)별', 'Group', 'SIG_ENG_NM', 'SIG_KOR_NM', 'geometry']]

        # raw_pred_df = raw_dataframe[['행정구역(시도)', '행정구역(시군구)별']].merge(self.pre_df, on=['행정구역(시도)', '행정구역(시군구)별'], how= 'outer')
        # raw_pred_df = raw_pred_df.fillna(-1)
        # raw_pred_df = raw_pred_df.astype({'Cluster Pred' : np.int64})

        # raw_pred_df['Group'] = raw_pred_df['행정구역(시군구)별'].apply(lambda x: x.split('(')[0])
        # raw_pred_df['Check'] = raw_pred_df[['행정구역(시군구)별', 'Group']].apply(lambda x: 1 if x[0] == x[1] else np.nan, axis= 1)
        # raw_pred_df.to_csv('./Data/map_csv.csv', index= False, encoding= 'utf-8-sig')

        # raw_pred_df = raw_pred_df.dropna()
        # raw_pred_df.drop('Check', axis= 1, inplace= True)

        urban_short_name2 = {
            '전북' : '전라북도',
            '전남' : '전라남도',
            '충북' : '충청북도',
            '충남' : '충청남도',
            '강원' : '강원특별자치도',
            '경북' : '경상북도',
            '경남' : '경상남도',
            '제주' : '제주특별자치도',
            '경기' : '경기도',
            '서울' : '서울특별시',
            '부산' : '부산광역시',
            '인천' : '인천광역시',
            '대구' : '대구광역시',
            '대전' : '대전광역시',
            '광주' : '광주광역시',
            '울산' : '울산광역시',
            '세종' : '세종특별자치시'
        }

        raw_pred_df = self.pre_df.copy()
        # raw_pred_df['행정구역(시도)'] = raw_pred_df['행정구역(시군구)별'].apply(lambda x: urban_short_name2[x.split(' ')[0]])
        # raw_pred_df['행정구역(시군구)별'] = raw_pred_df['행정구역(시군구)별'].apply(lambda x: x.split(' ')[-1])

        raw_pred_df['Group'] = raw_pred_df['행정구역(시군구)별'].apply(lambda x: x.split('(')[0])
        raw_pred_df['Check'] = raw_pred_df[['행정구역(시군구)별', 'Group']].apply(lambda x: 1 if x[0] == x[1] else np.nan, axis= 1)
        raw_pred_df['Cluster Pred'] = raw_pred_df['Cluster Pred'].apply(lambda x: '도시유형{}'.format(x))
        raw_pred_df.to_csv('./Data/map_csv.csv', index= False, encoding= 'utf-8-sig')

        raw_pred_df = raw_pred_df.dropna()
        raw_pred_df.drop('Check', axis= 1, inplace= True)

        korea = korea.merge(
            raw_pred_df[['행정구역(시도)', 'Group', 'Cluster Pred']],
            on = ['행정구역(시도)', 'Group'],
            how = 'inner'
        )

        korea = korea.dissolve(by=['행정구역(시도)', 'Group']).reset_index()
        ax = korea.plot(figsize=(15, 15), column="Cluster Pred", categorical=True,
                        edgecolor="k", legend=True, legend_kwds={'loc': 'lower right', 'fontsize' : 20})
        ax.set_title("Cluster Result")
        ax.set_axis_off()
        # palette = sns.color_palette('Set3')
        plt.savefig(self.save_path + 'Cluster_Result_Map_vis.png')
        # plt.show()

    ## 각 클러스터 별 중심값 변화
    def cluster_centerios_vis(self, cluster= 1, start_year= 2013, end_year= 2022):
        self.years_col = [x for x in range(start_year, end_year + 1)]

        # print(self.analysis_columns_dict.keys())

        centerios_df = pd.DataFrame(self.cluster_center[cluster - 1].T, columns= self.analysis_columns_dict.keys())
        centerios_df['Cluster'] = '도시유형{}'.format(cluster)
        centerios_df.to_excel(self.save_path + '[cluster{}] Centerios.xlsx'.format(cluster), index= False)

        if len(self.centerios_total_df) == 0:
            self.centerios_total_df = centerios_df

        else:
            self.centerios_total_df = pd.concat([self.centerios_total_df, centerios_df], axis= 0)

        plt.figure(figsize=(20, 8))
        for num_col in range(len(self.analysis_columns_dict.keys())):
            plt.plot(self.years_col, self.cluster_center[cluster - 1][num_col], label= list(self.analysis_columns_dict.keys())[num_col])

        plt.title('Cluster{} Results'.format(cluster))
        plt.xticks(self.years_col, rotation= 45)
        plt.ylabel('Rate')
        plt.xlabel('Year')
        plt.legend(loc= 'upper right')

        plt.savefig(self.save_path + 'Cluster{}_Results.png'.format(cluster))

    def cluster_centerios_columns(self):
        for col in self.analysis_columns_dict.keys():
            tmp_col_df = self.centerios_total_df[[col, 'Cluster']]
            per_columns_df = pd.DataFrame()

            for k in range(self.best_K):
                tmp_cluster_df = tmp_col_df[tmp_col_df['Cluster'] == '도시유형{}'.format(k + 1)]
                tmp_series = pd.Series(tmp_cluster_df[col], name= '도시유형{}'.format(k + 1))

                if len(self.centerios_total_df) == 0:
                    self.centerios_total_df = tmp_series
                    # per_columns_df = tmp_series
                    # print(self.centerios_total_df)

                else:
                    per_columns_df = pd.concat([per_columns_df, tmp_series], axis= 1)
                    
            # print(per_columns_df)

            year_col = pd.Series([2013 + x for x in range(len(per_columns_df))], name= 'year').to_frame()
            per_columns_df = pd.concat([per_columns_df, year_col], axis= 1)
            per_columns_df = per_columns_df.set_index('year')
            per_columns_df = per_columns_df.T

            per_columns_df.to_excel(self.save_path + '[{}] Centerios.xlsx'.format(col))

    ### 각 클러스터 별 상관관계 분석
    def corr_analysis_vis(self, cluster= 1):
        cluster_center_df = pd.DataFrame()

        for num_cluster in range(self.best_K):
            tmp_df = pd.DataFrame()
            for num_col in range(len(self.analysis_columns_dict.keys())):
                col_df = pd.Series(self.cluster_center[num_cluster][num_col], name= list(self.analysis_columns_dict.keys())[num_col]).to_frame()
                tmp_df = pd.concat([tmp_df, col_df], axis= 0)
            
            tmp_df['Cluster'] = num_cluster + 1
            
            if len(cluster_center_df) == 0:
                cluster_center_df = tmp_df
            else:
                cluster_center_df = pd.concat([cluster_center_df, tmp_df], axis= 0)

        cluster_center_df.reset_index(inplace= True, drop= True)

        corr_cluster = cluster_center_df[cluster_center_df['Cluster'] == cluster]
        corr_cluster.drop(['Cluster'], axis= 1, inplace= True)

        plt.figure(figsize=(20, 20))
        sns.heatmap(
            data= corr_cluster.corr(method= 'pearson'),
            annot = True,
            fmt = '.2f',
            vmin= -1, vmax= 1,
            cmap= 'coolwarm'
        )
        plt.title('Cluster{} 상관관계분석'.format(cluster))
        plt.savefig(self.save_path + 'Cluster{}_Corr.png'.format(cluster))

    def pred_result(self):
        return self.pre_df
    
    ### 클러스터 별 중심값과 거리 측정
    def distance_centroid(self):
        distance = list()
        raw_timeseries_data = np.array(self.timeseries_data[list(self.analysis_columns_dict.keys())].values.tolist())

        if self.metric == 'euclidean':
            distance = [euclidean_distances(raw_timeseries_data[i], self.cluster_center[self.cluster_labels[i]]) for i in range(len(self.timeseries_flatten))]

        elif self.metric == 'dtw':
            distance = [dtw(raw_timeseries_data[i], self.cluster_center[self.cluster_labels[i]]) for i in range(len(self.timeseries_flatten))]
            distance_df = pd.Series(distance, name= 'distance').to_frame()
            distance_df = pd.concat([self.pre_df[['행정구역(시도)', '행정구역(시군구)별', 'Cluster Pred']], distance_df], axis= 1)
            return distance_df
        
    ### 모든 클러스터의 중심값과 거리 측정
    def distance_centroid_all(self):
        raw_timeseries_data = np.array(self.timeseries_data[list(self.analysis_columns_dict.keys())].values.tolist())

        if self.metric.lower() == 'dtw':
            total_distance = list()
            for num in range(len(self.timeseries_flatten)):
                urban_distance = list()
                for cluster in range(self.best_K):
                    tmp_distance = dtw(raw_timeseries_data[num], self.cluster_center[self.cluster_labels[cluster]])
                    urban_distance.append(tmp_distance)
                total_distance.append(urban_distance)
                
            distance_df = pd.DataFrame(total_distance, columns=['도시유형{}'.format(x + 1) for x in range(self.best_K)])
            distance_df.to_excel(self.save_path + '클러스터 간 거리.xlsx',index= False)
            return distance_df
        
        else:
            print('Metric이 DTW인지 확인 부탁드립니다.(현재: {})'.format(self.metric))
            return np.nan
        
    def dimensionality_reduction(self):
        markers = ['^', 's', 'o', 'v', 'x', 'd', '+', '*', 'h']
        pca = PCA(n_components= 2)

        distance_df = self.distance_centroid_all()
        distance_pca = pca.fit_transform(distance_df)

        distance_pca_df = pd.DataFrame(distance_pca, columns= ['pca_component_1', 'pca_component_2'])
        distance_pca_df = pd.concat([self.pre_df[['행정구역(시도)', '행정구역(시군구)별', 'Cluster Pred']], distance_pca_df], axis= 1)

        for cluster_num in range(self.best_K):
            cluster_num = cluster_num + 1
            marker  = markers[cluster_num]
            x_data = distance_pca_df[distance_pca_df['Cluster Pred'] == cluster_num]['pca_component_1']
            y_data = distance_pca_df[distance_pca_df['Cluster Pred'] == cluster_num]['pca_component_2']
            plt.scatter(x_data, y_data, marker= marker, label= '도시유형{}'.format(cluster_num))

        plt.legend()
        plt.xlabel('pca_component_1')
        plt.xlabel('pca_component_2')
        plt.grid(False)
        plt.savefig(self.save_path + 'PCA 분석결과.png')

    def dimensionality_each_reduction(self, k_value = 0):
        markers = ['^', 's', 'o', 'v', 'x', 'd', '+', '*', 'h']
        pca = PCA(n_components= 2)

        distance_df = self.distance_centroid_all()
        distance_pca = pca.fit_transform(distance_df)

        distance_pca_df = pd.DataFrame(distance_pca, columns= ['pca_component_1', 'pca_component_2'])
        distance_pca_df = pd.concat([self.pre_df[['행정구역(시도)', '행정구역(시군구)별', 'Cluster Pred']], distance_pca_df], axis= 1)

        marker  = markers[k_value + 1]
        x_data = distance_pca_df[distance_pca_df['Cluster Pred'] == k_value + 1]['pca_component_1']
        y_data = distance_pca_df[distance_pca_df['Cluster Pred'] == k_value + 1]['pca_component_2']
        plt.scatter(x_data, y_data, marker= marker, label= '도시유형{}'.format(k_value + 1))

        plt.legend()
        plt.xlabel('pca_component_1')
        plt.xlabel('pca_component_2')
        plt.grid(False)
        plt.savefig(self.save_path + 'PCA 분석결과.png')
