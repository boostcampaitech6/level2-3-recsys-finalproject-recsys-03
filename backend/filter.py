import ast
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import time
from backend.config import config

# tags = ['사랑', '고통', '의도', '위로', '진실', '솔직함', '고난', '자아성찰', '자기계발', '희망']

def filter_model(question, tag_list):
    llm = ChatOpenAI(openai_api_key=config.open_ai_api_key)
    
    propmt_message_1 =  """ 1. 당신은 문장이나 단어에서 태그를 추출하는 태그 추출기가 됩니다. 선택할 수 있는 태그 후보 맨 아래에 드리겠습니다.
    2. 사람이 쓴 문장을 드릴 테니 읽어보시고, 태그 후보 목록에서 화자의 상황, 감정, 기분, 맥락에 가장 잘 맞는 5개의 태그를 고르시면 됩니다.
    3. 태그나 단어를 받은 경우, 반드시 결과물에 태그가 포함되어 있어야 합니다.
    4. 5개의 태그를 선택할 수 있으며 반드시 파이썬의 List 형태로 반환해야 합니다!
    5. kpop, pop, rock 등의 장르나 artist name에 대해서는 높은 가중치를 주면 됩니다!
    6. 만일 아무런 문장을 받지 않을 경우 무작위로 선택해서 보내주세요.
    """
    propmt_message_2 = str(tag_list)
    # 너는 사람이 입력한 문장을 읽고 난 뒤에, 문장의 상황과 감정, 분위기에 가장 어울리는 Tag를 뒤에 내가 주는 리스트에서 5개 정도 골라서 나한테 'python의 리스트 형태'로 돌려주면 돼." + str(tag_list)
    
    # print(tag_list)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", propmt_message_1 + propmt_message_2),
        ("user", "{input}")
    ])
    
    chain = prompt | llm 
    result = str(chain.invoke({"input": question}))
    print(result)
    
    # result 예시 : content="['sad', 'emotional', 'melancholy', 'reflective', 'introspective']" response_metadata={'finish_reason': 'stop', 'logprobs': None}
    start = result.find("[")
    end = result.find("]") + 1  
    
    # result에서 list만 추출
    extracted_result_str = result[start:end]
    
    # string -> list type
    extracted_result_list = eval(extracted_result_str)
        
    return extracted_result_list 
        
if __name__ == "__main__":
    df_tags = pd.read_csv('../data/tag.csv')
    tags = df_tags.Tag
    
    question1 = "긴 긴 하루였어요. 하나님이 제 인생을 망치려고 작정한 날이에요. 그러지 않고서야 어떻게 이럴 수 있겠어요. 저는 당신이 원하시는 걸 줄 수 있어요. 하지만 그건 진짜가 아닐 거예요. 진짜가 무엇일까요. 사실 다 솔직했는걸요." 
    question2 = "까마득한 날에 하늘이 처음 열리고어데 닭 우는 소리 들렸으랴 모든 산맥들이 바다를 연모해 휘달릴 때도 차마 이곳을 범하던 못하였으리라 끊임없는 광음을 부지런한 계절이 피어선 지고 큰 강물이 비로소 길을 열었다 지금 눈 내리고 매화 향기 홀로 아득하니 내 여기 가난한 노래의 씨를 뿌려라 다시 천고의 뒤에 백마 타고 오는 초인이 있어 이 광야에서 목놓아 부르게 하리라"
    question3= "언제나 취해 있어야 한다. 모든 것이 거기에 있다. 그것이 유일한 문제다. 그대의 어깨를 짓누르고, 땅을 향해 그대 몸을 구부러뜨리는 저 시간의 무서운 짐을 느끼지 않으려면, 쉴새없이 취해야 한다.그러나 무엇에? 술에, 시에 혹은 미덕에, 무엇에나 그대 좋을 대로. 아무튼 취하라.그리하여 때때로, 궁전의 섬돌 위에서, 도랑의 푸른 풀 위에서, 그대의 방의 침울한 고독 속에서, 그대 깨어 일어나, 취기가 벌써 줄어들거나 사라지거든, 물어보라, 바람에, 물결에, 별에, 새에, 시계에, 달아나는 모든 것에, 울부짖는 모든 것에, 흘러가는 모든 것에, 노래하는 모든 것에, 말하는 모든 것에, 물어보라, 지금이 몇시인지. 그러면 바람이, 물결이, 별이, 새가, 시계가, 그대에게 대답하리라. 지금은 취할 시간! 시간의 학대받는 노예가 되지 않으려면, 취하라, 끊임없이 취하라! 술에, 시에 혹은 미덕에, 그대 좋을 대로."
    question4 = "맥주 넘치지 않게 따라줘"
    
    time_result = []

    # print(filter_model(question1, tags)
    for question in [question1, question2, question3, question4]:
        print(question)
        print("-"*100)
        # 시작시간
        start = time.time()
        print(filter_model(question, tags))
        # 종료시간
        end = time.time()
        print("-"*100)
        time_result.append(end - start)

    print(time_result)