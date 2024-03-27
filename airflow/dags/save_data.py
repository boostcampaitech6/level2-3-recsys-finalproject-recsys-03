from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from crawling import get_new_interaction_track, get_audio_features, get_tags_with_lastfm_api, get_tags_with_crawling_lastfm

#하루에 한번 실행되는 신곡 데이터 수집 및 전처리 작업
daily_args = {
    'owner': 'siwoo',
    'start_date': datetime(2023, 3, 27),
    'schedule_interval':'0 0 * * *',
    'catchup':True,
}

with DAG(
    'crawling_new_track_info', 
    default_args=daily_args,
    description="Daily crawling new track DAG",
    ) as dag:
    
    get_new_interaction_track_task = PythonOperator(
        task_id = 'get_new_interaction_track',
        python_callable = get_new_interaction_track,
        provide_context = True,
    )
    
    get_audio_features_task = PythonOperator(
        task_id = 'get_audio_features',
        python_callable = get_audio_features,
        provide_context = True,
    )
    
    get_tags_with_lastfm_api_task = PythonOperator(
        task_id = 'get_tags_with_lastfm_api',
        python_callable = get_tags_with_lastfm_api,
        provide_context = True,
    )
    
    get_tags_with_crawling_lastfm_task = PythonOperator(
        task_id = 'get_tags_with_crawling_lastfm',
        python_callable = get_tags_with_crawling_lastfm,
        provide_context = True,
    )
    
    get_new_interaction_track_task >> get_audio_features_task >> get_tags_with_lastfm_api_task >> get_tags_with_crawling_lastfm_task