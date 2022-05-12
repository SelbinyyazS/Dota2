import pandas as pd
df=pd.read_csv('/dota2_skill_train.csv', index_col='id')
df['player_team']=df['player_team'].map({'dire':0, 'radiant':1})
df['winner_team']=df['winner_team'].map({'dire':0, 'radiant':1})
df.fillna(0)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(df.drop('skilled', axis=1), df.skilled , random_state=42)
from sklearn.tree import DecisionTreeClassifier
tree_gini=DecisionTreeClassifier(criterion='gini')
tree_gini.fit(x_train,y_train)
