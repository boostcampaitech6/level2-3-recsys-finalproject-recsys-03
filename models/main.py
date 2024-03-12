from graphsage_inference import graphsage_inference, GraphSAGE, DotPredictor, Model
from tag_ranking import tag_ranking_load_data, tag_ranking

if __name__ == "__main__":
    recommended_track_id = graphsage_inference(k=1000) # Graphsage 결과값 상위 K개
    recommended_list = tag_ranking_load_data(recommended_track_id)  # recommended_track_id에 side info 추가
    input_tags = "driving, party, upbeat, summer, electronic"
    recommended_ids, recommended_titles, recommended_artists, recommended_uris, recommended_selected_tags = tag_ranking(recommended_list, input_tags,N=20) #Graphsage 결과값에 tag ranking(기준:input tag와 동일한 tag수) 적용
    print("Recommended Track IDs:", recommended_ids)
    print("Recommended Titles:", recommended_titles)
    print("Recommended Artists:", recommended_artists)
    print("Recommended URIs:", recommended_uris)
    print("Recommended Selected Tags:", recommended_selected_tags) #input tag와 동일한 tags
