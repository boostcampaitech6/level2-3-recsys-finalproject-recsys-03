
# 이전 모델의 파일 백업 후 재학습 권장
# 혹은, --filename 파라미터를 변경하면 이름이 다르게 저장됨 (default: "graphsage")

# 모델 파일 : ./model/graphsage.pt
# inference 데이터 파일 : ../data/serving_data_graphsage.pt
# 인덱스 mapping 파일 : mapping ~ .json 8개
# 로그 파일 : ./log/graphsage.log

# 모델 학습
python train_graphsage.py

# inference 작동 확인
python inference_graphsage.py --serving_filename input_data.csv
