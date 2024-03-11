from models.tag_model import recommend_songs
from filter import filter_model

import pandas as pd
from pymongo import MongoClient


def make_playlist(question, tag_list):
    
    rec_tag = ', '.join(filter_model(question, tag_list))
    print(rec_tag)
    rec_playlist = recommend_songs(rec_tag)
    
    return rec_playlist 
    


if __name__ == "__main__":
    df_tags = pd.read_csv('data/tag.csv')
    tags = df_tags.Tag
    
    
    question1 = "긴 긴 하루였어요. 하나님이 제 인생을 망치려고 작정한 날이에요. 그러지 않고서야 어떻게 이럴 수 있겠어요. 저는 당신이 원하시는 걸 줄 수 있어요. 하지만 그건 진짜가 아닐 거예요. 진짜가 무엇일까요. 사실 다 솔직했는걸요." 
    question2 = "까마득한 날에 하늘이 처음 열리고어데 닭 우는 소리 들렸으랴 모든 산맥들이 바다를 연모해 휘달릴 때도 차마 이곳을 범하던 못하였으리라 끊임없는 광음을 부지런한 계절이 피어선 지고 큰 강물이 비로소 길을 열었다 지금 눈 내리고 매화 향기 홀로 아득하니 내 여기 가난한 노래의 씨를 뿌려라 다시 천고의 뒤에 백마 타고 오는 초인이 있어 이 광야에서 목놓아 부르게 하리라"
    question3= "언제나 취해 있어야 한다. 모든 것이 거기에 있다. 그것이 유일한 문제다. 그대의 어깨를 짓누르고, 땅을 향해 그대 몸을 구부러뜨리는 저 시간의 무서운 짐을 느끼지 않으려면, 쉴새없이 취해야 한다.그러나 무엇에? 술에, 시에 혹은 미덕에, 무엇에나 그대 좋을 대로. 아무튼 취하라.그리하여 때때로, 궁전의 섬돌 위에서, 도랑의 푸른 풀 위에서, 그대의 방의 침울한 고독 속에서, 그대 깨어 일어나, 취기가 벌써 줄어들거나 사라지거든, 물어보라, 바람에, 물결에, 별에, 새에, 시계에, 달아나는 모든 것에, 울부짖는 모든 것에, 흘러가는 모든 것에, 노래하는 모든 것에, 말하는 모든 것에, 물어보라, 지금이 몇시인지. 그러면 바람이, 물결이, 별이, 새가, 시계가, 그대에게 대답하리라. 지금은 취할 시간! 시간의 학대받는 노예가 되지 않으려면, 취하라, 끊임없이 취하라! 술에, 시에 혹은 미덕에, 그대 좋을 대로."
    question4 = "맥주 넘치지 않게 따라줘"
    
    for question in [question1, question2, question3, question4]:
        print(question)
        print(make_playlist(question, tags))
        print("-"*100)