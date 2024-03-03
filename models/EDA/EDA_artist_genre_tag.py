import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

## Data 불러오기
#os.chdir('/home/jjw/level2-3-recsys-finalproject-recsys-03/models')
#os.getcwd()
data_path = 'data/'
interaction = pd.read_csv(data_path+"interaction.csv",index_col=0)
sideinfo = pd.read_csv(data_path+"sideinfo.csv",index_col=0)
interaction_merged = pd.merge(interaction,sideinfo,on='track_id',how='left')
#interaction_merged.iloc[:,:5]


### Artist EDA ###

# Artist 1093개
print('Artist 수 :', interaction_merged['artist_name'].nunique())

## Top artists by listening count
# 생각보다 Rock 계열이 많이보임
artist_listening_sum = interaction_merged.groupby('artist_name')['total_listening'].sum().reset_index()
top_artists_by_listening = artist_listening_sum.sort_values(by='total_listening', ascending=False).head(50)
# plot
plt.figure(figsize=(20, 12))
plt.barh(top_artists_by_listening['artist_name'], top_artists_by_listening['total_listening'])
plt.xlabel('Total Listening Count')
plt.ylabel('Artist')
plt.title('Top 50 Artists by Total Listening')
plt.gca().invert_yaxis() 
plt.savefig('Top 50 Artists by Total Listening')

## Top artists by Frequency
artist_user_count = interaction_merged.groupby('artist_name')['user_id'].nunique().reset_index()
top_artists_by_user = artist_user_count.sort_values(by='user_id', ascending=False).head(50)
# Plot
plt.figure(figsize=(20, 12))
plt.barh(top_artists_by_user['artist_name'], top_artists_by_user['user_id'])
plt.xlabel('Number of Unique Users')
plt.ylabel('Artist')
plt.title('Top 50 Artists by Frequency')
plt.gca().invert_yaxis()  
plt.savefig('Top 50 Artists by Frequency')
# bin으로 나누기
bins = pd.cut(artist_user_count['user_id'], 5)
print(bins.value_counts())
# Artists by Frequency는 왼족으로 치우친 분포



### Genres EDA ###

sideinfo_genre = sideinfo['genres'].apply(eval)

## Genre의 Weight 분포 확인
genres_over_50 = []
for genre_dict in sideinfo_genre:
    for genre, score in genre_dict.items():
        if score > 50:
            genres_over_50.append(genre)

unique_genres_over_50 = set(genres_over_50)
print(len(unique_genres_over_50))
# Genre의 weight에 따른 Genre갯수
# 50이하: 61335 / 50초과: 70924

# Genre의 unique 갯수
# 전체: 830개 / 50이하: 830개 / 50초과: 189개
# 50초과 List에 없는 top_3_df['Top1 Genre'] 존재

## Genre weight의 sum
genre_counts = {}
for genre_dict in sideinfo_genre:
    for genre, count in genre_dict.items():
        if genre in genre_counts:
            genre_counts[genre] += count
        else:
            genre_counts[genre] = count

sorted_genre_counts = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
genre_counts_list = list(genre_counts.items())
genre_counts_df = pd.DataFrame(genre_counts_list, columns=['Genre', 'Weight Sum'])
genre_counts_df_sorted = genre_counts_df.sort_values(by='Weight Sum', ascending=False)

# top 50의 plot
plt.figure(figsize=(12, 8)) 
plt.bar(genre_counts_df_sorted['Genre'].head(50), genre_counts_df_sorted['Weight Sum'].head(50))
plt.xlabel('Genre')  
plt.ylabel('Weight Sum')  
plt.title('Top 50 Genres by Weight Sum')  
plt.xticks(rotation=90) 
plt.grid(axis='y', linestyle='--', alpha=0.7)  
plt.savefig('Top 50 Genres by Weight Sum')


## Genre weight(1~100)의 Frequency
genre_counts = []
for genre_dict in sideinfo_genre:
    for count in genre_dict.values():
        genre_counts.append(count)

genre_counts_df = pd.DataFrame(genre_counts, columns=['Weights'])
print(genre_counts_df.describe())
genre_value_counts=genre_counts_df.value_counts()

genre_value_counts_df = pd.DataFrame(list(genre_value_counts.items()), columns=['Weights', 'Frequency'])
genre_value_counts_df['Weights'] = genre_value_counts_df['Weights'].apply(lambda x: x[0] if isinstance(x, tuple) else x)
genre_value_counts_df = genre_value_counts_df.sort_values(by='Weights')

# plot : Genre counts
plt.figure(figsize=(20, 12))
plt.bar(genre_value_counts_df['Weights'].astype(str), genre_value_counts_df['Frequency'], width=0.8)
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Genres by Weight Frequency')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.savefig('Genres by Weight Frequency')

# 각 item(행)별로 가장 큰 값 3개
top_3_per_row = []
for genre_dict in sideinfo_genre:
    top_3 = sorted(genre_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    top_3_per_row.append([item[0] for item in top_3] + [item[1] for item in top_3])

columns = ['Top1 Genre', 'Top2 Genre', 'Top3 Genre', 'Top1 Count', 'Top2 Count', 'Top3 Count']
top_3_df = pd.DataFrame(top_3_per_row, columns=columns)
print(top_3_df['Top1 Genre'].value_counts().head(10))
# count 100 : 3677개 


### Tags EDA ###

sideinfo_tag = sideinfo['tags'].apply(eval)

## Tag의 Weight 분포 확인
tags_over_50 = []
for tag_dict in sideinfo_tag:
    for tag, score in tag_dict.items():
        if score > 50:
            tags_over_50.append(tag)

unique_tags_over_50 = set(tags_over_50)
print(len(unique_tags_over_50))
# tag의 unique 갯수
# 전체: 50081개 / 50이하: 50015개 / 50초과: 739개
# 50초과 List에 없는 top_3_df['Top1 tag'] 존재


## Tag weight의 sum
tag_counts = {}
for tag_dict in sideinfo_tag:
    for tag, count in tag_dict.items():
        if tag in tag_counts:
            tag_counts[tag] += count
        else:
            tag_counts[tag] = count

sorted_tag_counts = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
tag_counts_list = list(tag_counts.items())
tag_counts_df = pd.DataFrame(tag_counts_list, columns=['tag', 'Weight Sum'])
tag_counts_df_sorted = tag_counts_df.sort_values(by='Weight Sum', ascending=False)

# top 50의 plot
plt.figure(figsize=(12, 8)) 
plt.bar(tag_counts_df_sorted['tag'].head(50), tag_counts_df_sorted['Weight Sum'].head(50))
plt.xlabel('Tag')  
plt.ylabel('Weight Sum')  
plt.title('Top 50 Tags by Weight Sum')  
plt.xticks(rotation=90) 
plt.grid(axis='y', linestyle='--', alpha=0.7)  
plt.savefig('Top 50 Tags by Weight Sum')


## Tag weight(1~100)의 Frequency
tag_counts = []
for tag_dict in sideinfo_tag:
    for count in tag_dict.values():
        tag_counts.append(count)

tag_counts_df = pd.DataFrame(tag_counts, columns=['Weights'])
print(tag_counts_df.describe())
tag_value_counts=tag_counts_df.value_counts()

tag_value_counts_df = pd.DataFrame(list(tag_value_counts.items()), columns=['Weights', 'Frequency'])
tag_value_counts_df['Weights'] = tag_value_counts_df['Weights'].apply(lambda x: x[0] if isinstance(x, tuple) else x)
tag_value_counts_df = tag_value_counts_df.sort_values(by='Weights')

# plot : Tag counts
plt.figure(figsize=(20, 12))
plt.bar(tag_value_counts_df['Weights'].astype(str), tag_value_counts_df['Frequency'], width=0.8)
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Tags by Weight Frequency')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.savefig('Tags by Weight Frequency')

# 각 item(행)별로 가장 큰 값 3개
top_3_per_row = []
for tag_dict in sideinfo_tag:
    top_3 = sorted(tag_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    top_3_per_row.append([item[0] for item in top_3] + [item[1] for item in top_3])

columns = ['Top1 tag', 'Top2 tag', 'Top3 tag', 'Top1 Count', 'Top2 Count', 'Top3 Count']
top_3_df = pd.DataFrame(top_3_per_row, columns=columns)
print(top_3_df['Top1 Count'].value_counts().head(10))
# Top1 Tag : Weight 100이 5637개 

