import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
from collections import defaultdict
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv('attribution_data.csv', parse_dates=['time'])
df = df.sort_values(['cookie', 'time'],
                    ascending=[False, True])
df['visit_order'] = df.groupby('cookie').cumcount() + 1


df_paths = df.groupby('cookie')['channel'].aggregate(
    lambda x: x.unique().tolist()).reset_index()
    
df_last_interaction = df.drop_duplicates('cookie', keep='last')[['cookie', 'conversion']]
df_paths = pd.merge(df_paths, df_last_interaction, how='left', on='cookie')

df_paths_ass = df_paths.copy()



df_paths['path'] = np.where( df_paths['conversion'] == 0,
['Start, '] + df_paths['channel'].apply(', '.join) + [', Null'],
['Start, '] + df_paths['channel'].apply(', '.join) + [', Conversion'])


df_paths['path'] = df_paths['path'].str.split(', ')

df_paths = df_paths[['cookie', 'path','conversion']]
list_of_paths = df_paths['path']
total_conversions = sum(path.count('Conversion') for path in df_paths['path'].tolist())
base_conversion_rate = total_conversions / len(list_of_paths)


paths_list = list_of_paths.copy().to_frame()
paths_list['conversion'] = paths_list['path'].apply(lambda x: 'Conversion' in x)
paths_list['path'] = paths_list['path'].apply(lambda x: x[1:-1])
paths_list['count'] = paths_list['path'].apply(len)



t = paths_list.iloc[:240108]
g = t.groupby(['conversion', 'count']).count()
IM_conv_count = g.unstack('conversion').plot.bar()


def transition_states(list_of_paths):
    list_of_unique_channels = set(x for element in list_of_paths for x in element)
    transition_states = {x + '>' + y: 0 for x in list_of_unique_channels for y in list_of_unique_channels}

    for possible_state in list_of_unique_channels:
        if possible_state not in ['Conversion', 'Null']:
            for user_path in list_of_paths:
                if possible_state in user_path:
                    indices = [i for i, s in enumerate(user_path) if possible_state in s]
                    for col in indices:
                        transition_states[user_path[col] + '>' + user_path[col + 1]] += 1

    return transition_states


trans_states = transition_states(list_of_paths)



def transition_prob(trans_dict):
    list_of_unique_channels = set(x for element in list_of_paths for x in element)
    trans_prob = defaultdict(dict)
    for state in list_of_unique_channels:
        if state not in ['Conversion', 'Null']:
            counter = 0
            index = [i for i, s in enumerate(trans_dict) if state + '>' in s]
            for col in index:
                if trans_dict[list(trans_dict)[col]] > 0:
                    counter += trans_dict[list(trans_dict)[col]]
            for col in index:
                if trans_dict[list(trans_dict)[col]] > 0:
                    state_prob = float((trans_dict[list(trans_dict)[col]])) / float(counter)
                    trans_prob[list(trans_dict)[col]] = state_prob

    return trans_prob


trans_prob = transition_prob(trans_states)


def transition_matrix(list_of_paths, transition_probabilities):
    trans_matrix = pd.DataFrame()
    list_of_unique_channels = set(x for element in list_of_paths for x in element)

    for channel in list_of_unique_channels:
        trans_matrix[channel] = 0.00
        trans_matrix.loc[channel] = 0.00
        trans_matrix.loc[channel][channel] = 1.0 if channel in ['Conversion', 'Null'] else 0.0

    for key, value in transition_probabilities.items():
        origin, destination = key.split('>')
        trans_matrix.at[origin, destination] = value

    return trans_matrix


trans_matrix = transition_matrix(list_of_paths, trans_prob)


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(12, 10))
IM_heat_map = sns.heatmap(trans_matrix, annot=True)



def removal_effects(df, conversion_rate):
    removal_effects_dict = {}
    channels = [channel for channel in df.columns if channel not in ['Start',
                                                                     'Null',
                                                                     'Conversion']]
    for channel in channels:
        removal_df = df.drop(channel, axis=1).drop(channel, axis=0)
        for column in removal_df.columns:
            row_sum = np.sum(list(removal_df.loc[column]))
            null_pct = float(1) - row_sum
            if null_pct != 0:
                removal_df.loc[column]['Null'] = null_pct
            removal_df.loc['Null']['Null'] = 1.0

        removal_to_conv = removal_df[
            ['Null', 'Conversion']].drop(['Null', 'Conversion'], axis=0)
        removal_to_non_conv = removal_df.drop(
            ['Null', 'Conversion'], axis=1).drop(['Null', 'Conversion'], axis=0)

        removal_inv_diff = np.linalg.inv(
            np.identity(
                len(removal_to_non_conv.columns)) - np.asarray(removal_to_non_conv))
        removal_dot_prod = np.dot(removal_inv_diff, np.asarray(removal_to_conv))
        removal_cvr = pd.DataFrame(removal_dot_prod,
                                   index=removal_to_conv.index)[[1]].loc['Start'].values[0]
        removal_effect = 1 - removal_cvr / conversion_rate
        removal_effects_dict[channel] = removal_effect

    return removal_effects_dict


removal_effects_dict = removal_effects(trans_matrix, base_conversion_rate)


def markov_chain_allocations(removal_effects, total_conversions):
    re_sum = np.sum(list(removal_effects.values()))

    return {k: (v / re_sum) * total_conversions for k, v in removal_effects.items()}


attributions = markov_chain_allocations(removal_effects_dict, total_conversions)

IM_attributions = plt.bar(*zip(*attributions.items()))

header = st.container()
dataset = st.container()
markov = st.container()
associations = st.container()
header.title('Маркетинговая аналитика цифровых клиентов на основе вероятностных моделей и методов машинного обучения')

dataset.header('Данные о взаимодействии пользователей с рекламными каналами')
orig_df = df.drop('visit_order', axis=1)
dataset.write(orig_df)


dataset.header('Соотношение конверсионных событий к неконверсионным')
IM_conversions = df['conversion'].value_counts()
dataset.bar_chart(IM_conversions)

dataset.header('Зависимость конверсии от количества каналов в пути пользователя')
conversion_count = paths_list.groupby(['conversion', 'count']).count().unstack('conversion')
fig, ax = plt.subplots()
conversion_count.plot.bar(ax=ax)
plt.legend(['Нет конверсии', 'Есть конверсия'])
plt.xlabel('Количество каналов в пути')
dataset.pyplot(fig)


dataset.header('Аналитика по дням')

df['day'] = df['time'].apply(lambda t: t.day)
days_count_all = df.query('conversion == 0').groupby('day')['day'].count()
days_count_conv = df.query('conversion > 0').groupby('day')['day'].count()

fig, ax = plt.subplots()
days_count_all.plot(ax=ax, color='blue', label='Без конверсии')
days_count_conv.plot(ax=ax, color='red', label='Конверсия')
ax.set_yscale('log')
plt.legend(loc='best')
dataset.write(fig)

dataset.header('Аналитика по часам')

df['hour'] = df['time'].apply(lambda t: t.hour)
hours_count_all = df.query('conversion == 0').groupby('hour')['hour'].count()
hours_count_conv = df.query('conversion > 0').groupby('hour')['hour'].count()

fig, ax = plt.subplots()
hours_count_all.plot(ax=ax, color='blue', label='Без конверсии')
hours_count_conv.plot(ax=ax, color='red', label='Конверсия')
ax.set_yscale('log')
plt.legend(loc='best')

dataset.write(fig)

markov.header('Предобработка данных')
markov.write(df_paths.head(10))

markov.header('Матрица переходов между каналами')
markov.write(trans_matrix)
markov.header('Тепловая карта')
fig, ax = plt.subplots()
sns.heatmap(trans_matrix, ax=ax)
markov.write(fig)

# markov.header('Эффекты удаления для каждого из каналов')
# src = pd.DataFrame({
#     'Эффекты удаления': [0.3547597674182721, 0.21731366149038445, 0.15435482356041286, 0.2069141165564219, 0.3311037560086154],
# 'Каналы': ["Facebook", "Instagram", "Online Display", "Online Video", "Paid Search"]
# })
# bar_chart = alt.Chart(src).mark_bar().encode(
#         y='Эффекты удаления',
#         x='Каналы',
#     )
# markov.altair_chart(bar_chart, use_container_width=True)

markov.header('Атрибуция каналов')
source = pd.DataFrame({
        'Распределение': [4948.892177847523, 2153.2469267590823, 4618.891257291356, 3031.5215485558915, 2886.4480895461475],
        'Каналы': ["Facebook", "Online Display", "Paid Search", "Instagram", "Online Video"]
     })
 
bar_chart = alt.Chart(source).mark_bar().encode(
        y='Распределение',
        x='Каналы',
    )
 
markov.altair_chart(bar_chart, use_container_width=True)

first_touch = df_paths.query('conversion > 0').path.apply(lambda x: x[1])
last_touch = df_paths.query('conversion > 0').path.apply(lambda x: x[-2])
assert not (first_touch == 'Start').any()
assert not (last_touch == 'Conversion').any()

fig, ax = plt.subplots()
first_touch_count = first_touch.to_frame().groupby('path')['path'].count()
last_touch_count = last_touch.to_frame().groupby('path')['path'].count()

markov_count = {'Facebook': 4948.892177847523,
 'Instagram': 3031.5215485558915,
 'Online Display': 2153.2469267590836,
 'Online Video': 2886.4480895461475,
 'Paid Search': 4618.891257291356}

result = pd.DataFrame({'first_touch': first_touch_count, 'last_touch': last_touch_count})

result['markov'] = markov_count.values()
result.plot.bar(y=['first_touch', 'last_touch', 'markov'], ax=ax)
markov.header('Сравнение результатов атрибуции модели первого и последнего касания с атрибуцией по модели цепей маркова')
markov.write(fig)

df_paths_conv = df_paths_ass.loc[df_paths_ass['conversion'] == 1]
channels = df_paths_conv['channel']
data = []
for ch in channels:
  data.append(ch)


a = TransactionEncoder()
a_data = a.fit(data).transform(data)
table = pd.DataFrame(a_data,columns=a.columns_)
table = table.replace(False,0)
table = table.replace(True,1)

table_ap = apriori(table, min_support=0.04, use_colnames=True)

table_ar = association_rules(table_ap, metric = "confidence", min_threshold = 0.1)
table_ar['title'] = table_ar.apply(lambda c: list(c['antecedents'])[0] + '->' + list(c['consequents'])[0], axis=1)

plt.figure(figsize=(15, 12))


x, y = table_ar['support'], table_ar['confidence']
fig, ax = plt.subplots(figsize=(15,12))
ax.scatter(x, y)
for i, txt in enumerate(table_ar['title']):
    plt.annotate(txt, (x.iloc[i], y.iloc[i]), fontsize=17)

associations.header('Ассоциативные правила')
associations.write(fig)


