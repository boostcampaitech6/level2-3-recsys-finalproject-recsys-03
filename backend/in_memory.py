from models.inference_tag_model import load_data_for_tag_model, load_tag_model
from models.inference_cbf_model import load_data_for_cbf_model, load_cbf_model

class Model_memory():
    def __init__(self):
        self.tag_model_dict = None
        self.cbf_model_dict = None

    def load_tag_model_memory(self): 
        mapping_index_track, mapping_tag_index, graph_data = load_data_for_tag_model()
        embeddings = load_tag_model(graph_data)

        tag_model_dict = {'mapping_index_track': mapping_index_track,
                        'mapping_tag_index': mapping_tag_index,
                        'embeddings': embeddings}
        self.tag_model_dict = tag_model_dict


    def load_cbf_model_memory(self):
        mapping_index_track, mapping_track_index, graph_data = load_data_for_cbf_model()
        embeddings = load_cbf_model(graph_data)

        cbf_model_dict = {'mapping_index_track': mapping_index_track,
                        'mapping_track_index': mapping_track_index,
                        'embeddings': embeddings}
        self.cbf_model_dict = cbf_model_dict

model_memory = Model_memory()

