import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
import streamlit as st
import json
import folium
import altair as alt
import shap
import streamlit.components.v1 as components
import joblib
import requests
import io
import plotly.express as px
from streamlit_folium import st_folium

st.set_page_config(layout = 'wide')
# st.set_option('./deprecation.showPyplotGlobalUse', False)

@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Jeon-doun/food_poissoning_dashboard/refs/heads/main/Foodborne_Region_MasterTable_2.csv')
    return df

@st.cache_data
def load_json():
    state_geo = 'https://raw.githubusercontent.com/Jeon-doun/food_poissoning_dashboard/refs/heads/main/TL_SCCO_CTPRVN.json'
    response = requests.get(state_geo)
    response.raise_for_status() # 요청에 실패하면 오류 발생
    jsonResult = response.json()
    # json_data = open(state_geo, encoding = 'utf-8').read()
    # jsonResult = json.loads(json_data)
    return jsonResult

@st.cache_resource
def load_model(model_path):
    response = requests.get(model_path)
    response.raise_for_status() # 요청에 실패하면 오류 발생
    file = io.BytesIO(response.content) # pkl파일 불러오기
    loaded_model = joblib.load(file)
    return loaded_model

def convert_dash_info(x):
    if x == '종합위험도':
        return 'risk'
    elif x == '발생건수':
        return 'OCCRNC_CNT'
    elif x == '발생환자수':
        return 'PATNT_CNT'
    
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def shap_summary_plot(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    plot = shap.force_plot(explainer.expected_value, shap_values.values[0, :], X.iloc[0, :])
    return st_shap(plot, height=300)

def main():

    # 데이터 로드 및 캐시 저장
    data = load_data()
    data.index = pd.to_datetime(data['OCCRNC_YEAR'].astype(str) + '-' + data['OCCRNC_MM'].astype(str))
    data = data.sort_index()

    # 모델 로드 및 캐시 저장(추후에 지역별로 전부 불러와야함)
    model = load_model('https://github.com/Jeon-doun/food_poissoning_dashboard/raw/refs/heads/main/test_clf_model.pkl')
    model = model.best_estimator_

    # 행정구역 json 로드 및 캐시 저장
    jsonResult = load_json()

    st.sidebar.title('전국 식중독 현황')

    with st.sidebar:
        function = st.radio('기능을 선택하세요'
                        , ['현황 모니터링','예측 시뮬레이션']
                        , horizontal=True)

    if function == '현황 모니터링':

        # 현황 모니터링 화면에서 KPI 표시

        col1, col2 ,col3, col4 = st.columns(4)

        max_yearm = data.index.max()
        max_yearm_1 = max_yearm - datetime.timedelta(days=1)
        year_1 = datetime.datetime(max_yearm_1.year, max_yearm_1.month, 1)
        with col1:

            st.metric(label = '종합위험도'
                        , value = '주의')
            
        with col2:
            
            OCCRNC_CNT_KPI = data.loc[data.index == max_yearm, 'OCCRNC_CNT'].sum()
            OCCRNC_CNT_KPI_1 = data.loc[data.index == year_1, 'OCCRNC_CNT'].sum()
            st.metric(label = '이달의 발생건수'
                        , value = f'{OCCRNC_CNT_KPI:.0f}건'
                        , delta = f'{(OCCRNC_CNT_KPI - OCCRNC_CNT_KPI_1):.0f}건')
            
        with col3:

            PATNT_CNT_KPI = data.loc[data.index == max_yearm, 'PATNT_CNT'].sum()
            PATNT_CNT_KPI_1 = data.loc[data.index == year_1, 'PATNT_CNT'].sum()
            st.metric(label = '이달의 환자수'
                        , value = f'{PATNT_CNT_KPI:.0f}명'
                        , delta = f'{(PATNT_CNT_KPI - PATNT_CNT_KPI_1):.0f}명')
            
        with col4:

            st.metric(label = '이달의 위험물질'
                        , value = '노로바이러스'
                        )
        
        st.write('--'*3)

        # sidebar 설정
            
        with st.sidebar:

            # 조회할 정보 선택
            dash_info = st.selectbox("조회할 정보를 선택하세요"
                                    , ['발생건수'
                                    ,'발생환자수'])
            
            dash_info_eng = convert_dash_info(dash_info)

            # 조회 기간 선택
            st.write('조회 기간을 선택하세요')

            start_year = st.sidebar.selectbox('시작년도'
                                    , range(2002, 2023))
            start_month = st.sidebar.selectbox('시작월'
                                    , range(1, 13))
            end_year = st.sidebar.selectbox('종료연도'
                            , range(2022, 2001, -1))
            end_month = st.sidebar.selectbox('종료월'
                                    , range(12, 0, -1))
            
            function = st.radio('조회 조건을 선택하세요'
                        , ['전체','상위 5개지역','하위 5개지역']
                        , horizontal=False)

        # sidebar 설정값에 따라 데이터 필터링

        target_df = data.loc[(data.index >= datetime.datetime(start_year, start_month, 1))
                              & (data.index <= datetime.datetime(end_year, end_month, 1)),:]
        group_df = target_df.groupby('OCCRNC_REGN')[dash_info_eng].sum().sort_values(ascending = False)
        group_df = group_df.loc[group_df > 0]
             
        if function == '전체':
            region_filter = group_df.index
        
        elif function == '상위 5개지역':
            
            region_filter = group_df[:5].index
            

        elif function == '하위 5개지역':
            region_filter = group_df[-5:].index

        # 지도 시각화

        col1, col2 = st.columns(2)

        with col1:
                with st.container():

                    st.subheader('식중독 지도')

                    map_df = target_df.loc[(target_df[dash_info_eng] > 0) & (target_df['OCCRNC_REGN'].isin(region_filter))]

                    m = folium.Map(location = [36.2, 127.8], tiles = 'Cartodb Positron', zoom_start = 7, zoom_control = False)

                    folium.Choropleth(
                        geo_data = jsonResult
                        , name = dash_info
                        , data = map_df.groupby('CTPRVN_CD', as_index = False)[dash_info_eng].sum()
                        , columns = ['CTPRVN_CD', dash_info_eng]
                        , key_on='feature.properties.CTPRVN_CD'
                        , fill_color = 'YlOrRd'
                        , fill_opacity = 0.7
                        , line_opacity = 0.3
                        , color = 'gray'
                        , legend_name = dash_info
                    ).add_to(m)

                    def on_click(feature):
                        return {
                            # 'fillColor': '#ffaf00',
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.1
                        }

                    folium.GeoJson(
                        data = jsonResult,
                        name="OCCRNC_REGN",
                        style_function=on_click,
                        tooltip=folium.features.GeoJsonTooltip(
                            fields=["CTP_KOR_NM"],  # GeoJSON의 구역 이름 필드명
                        ),
                        highlight_function=lambda x: {"weight": 3, "color": "gray"}
                    ).add_to(m)

                    output = st_folium(m, width=500, height=700, use_container_width=True)

        with col2:
                
                st.subheader('원인물질별 발생 현황')

                st.subheader('식중독 발생 주요 요인')

                if output["last_active_drawing"]:

                    # 선택된 행정구역 코드 반환
                    selected_area = output["last_active_drawing"]["properties"]["CTPRVN_CD"]

                    # 원 데이터의 CTPRVN_CD가 정수형 변수여서 강제로 변환해줌, 이후 데이터타입 변경 필요
                    target_df_2 = target_df.loc[target_df['CTPRVN_CD'] == int(selected_area)]

                    X = target_df_2[model.feature_names_in_]

                    shap_plot = shap_summary_plot(model, X)

                    st.subheader('월별 식중독 발생 현황')

                    # 선형 그래프 도식화
                    st.bar_chart(target_df_2[dash_info_eng], color = '#848484', height=150)

                else:

                    st.bar_chart(target_df[dash_info_eng], color = '#848484', height=150)


    elif function == '예측 시뮬레이션':

        with st.sidebar:
            
            st.write('변화율을 설정하세요')
            st.write(f'(설정 기준월 : {data.index.max().strftime('%Y년 %m월')})')

            option_value = {}

            option_value['기온'] = st.number_input('기온 변화율(%)', value = 0.0, step = 0.1, format = '%.1f')
            option_value['강수량'] = st.number_input('강수량 변화율(%)', value = 0.0, step = 0.1, format = '%.1f')
            option_value['습도'] = st.number_input('습도 변화율(%)', value = 0.0, step = 0.1, format = '%.1f')
            option_value['인구'] = st.number_input('인구 변화율(%)', value = 0.0, step = 0.1, format = '%.1f')
            option_value['학생수'] = st.number_input('학생수 변화율(%)', value = 0.0, step = 0.1, format = '%.1f')

            
        test_X = data.loc[data.index == data.index.max(), model.feature_names_in_]
        
        col1, col2 = st.columns(2)

        with col1:

            st.subheader('식중독 예측 지도')

            기온변수 = ['WTHR_AVG_TEMP','WTHR_AVG_H_TEMP','WTHR_AVG_L_TEMP']
            강수량변수 = ['WTHR_AVG_PRECIP']
            인구변수 = ['POP_GEN_CNT', 'POP_60P_CNT', 'POP_60P_RATIO', 'POP_DENS']
            습도변수 = ['WTHR_AVG_RHUM', 'WTHR_MN_RHUM']
            학생수변수 = ['POP_ELM_CNT', 'POP_MID_CNT','POP_HIGH_CNT','POP_ELM_RATIO', 'POP_MID_RATIO', 'POP_HIGH_RATIO']

            for col in option_value.keys():
                if col == '기온':
                    test_X[기온변수] = test_X[기온변수] * (1 + option_value[col]/100)
                elif col == '강수량':
                    test_X[강수량변수] = test_X[강수량변수] * (1 + option_value[col]/100)
                elif col == '인구':
                    test_X[인구변수] = test_X[인구변수] * (1 + option_value[col]/100)
                elif col == '습도':
                    test_X[습도변수] = test_X[습도변수] * (1 + option_value[col]/100)
                elif col == '학생수':
                    test_X[학생수변수] = test_X[학생수변수] * (1 + option_value[col]/100)

            
            test_X['pred_y'] = model.predict(test_X)
            test_X['CTPRVN_CD'] = data.loc[data.index == data.index.max(), 'CTPRVN_CD']

            m = folium.Map(location = [36.2, 127.8], tiles = 'Cartodb Positron', zoom_start = 7, zoom_control = False)

            folium.Choropleth(
                geo_data = jsonResult
                , name = '식중독 발생확률'
                , data = test_X
                , columns = ['CTPRVN_CD', 'pred_y']
                , key_on='feature.properties.CTPRVN_CD'
                , fill_color = 'YlOrRd'
                , fill_opacity = 0.7
                , line_opacity = 0.3
                , color = 'gray'
                , legend_name = '식중독 발생확률'
            ).add_to(m)

            def on_click(feature):
                return {
                    # 'fillColor': '#ffaf00',
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.1
                }

            folium.GeoJson(
                data = jsonResult,
                name="OCCRNC_REGN",
                style_function=on_click,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=["CTP_KOR_NM"],  # GeoJSON의 구역 이름 필드명
                ),
                highlight_function=lambda x: {"weight": 3, "color": "gray"}
            ).add_to(m)

            output = st_folium(m, width=500, height=700, use_container_width=True)

        # with col2:



if __name__ == '__main__':
    main()