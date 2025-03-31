# IPL DATA 

![alt text](https://tse2.mm.bing.net/th?id=OIP.Zcw-IMxjeYWlMPW7NkevzgHaFT&pid=Api&P=0&h=180 "logo title")

Reference data [IPL_DATA](https://drive.google.com/drive/folders/1LYbvR_34eMLGL7K0ASvqTCIBZXvcKn-M)

IPL 2025 [MATCH_SCHEDULE](https://www.google.com/search?q=ipl+matches+2025&sca_esv=fa927a4af76fe362&rlz=1C5CHFA_enIN1072IN1073&sxsrf=AHTn8zpeinaHm1L8Z09COP-jHVeLEkWCxw%3A1742551763133&ei=0zrdZ4jbB4GcseMPorfrwQU&ved=0ahUKEwjIkOCW95qMAxUBTmwGHaLbOlgQ4dUDCBA&uact=5&oq=ipl+matches+2025&gs_lp=Egxnd3Mtd2l6LXNlcnAiEGlwbCBtYXRjaGVzIDIwMjUyERAAGIAEGJECGLEDGIMBGIoFMgYQABgHGB4yBhAAGAcYHjIGEAAYBxgeMgYQABgHGB4yBhAAGAcYHjIGEAAYBxgeMgYQABgHGB4yBhAAGAcYHjIGEAAYBxgeSLcRUPIGWLkQcAJ4AZABAJgBsQSgAaAMqgELMC40LjAuMS4wLjG4AQPIAQD4AQGYAgOgApIBwgIKEAAYsAMY1gQYR5gDAIgGAZAGCJIHAzIuMaAH4CmyBwMwLjG4B4gB&sclient=gws-wiz-serp#cobssid=s&sie=lg;/g/11w9tbcchv;5;/m/03b_lm1;mt;fp;1;;;)


## objective
by analysing the data set we are going to find 
* who will win the toss
* who will win the match

```python
# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```
Loading the data from dataset 
```python
#loading the data
matches_df = pd.read_csv('matches.csv')
deliveries_df = pd.read_csv('deliveries.csv')
  ```

Droping missing values
```python
# Drop missing values
matches_df.dropna(subset=['winner'], inplace=True)
```
```python
# Encode categorical variables
encoder = LabelEncoder()
all_teams = sorted(set(matches_df['team1']).union(set(matches_df['team2'])))
encoder.fit(all_teams)

matches_df['team1'] = encoder.transform(matches_df['team1'])
matches_df['team2'] = encoder.transform(matches_df['team2'])
matches_df['toss_winner'] = encoder.transform(matches_df['toss_winner'])
matches_df['winner'] = encoder.transform(matches_df['winner'])
```
## cleaning the data set
```python
matches['player_of_match'].fillna('Unknown', inplace=True)
matches['city'].fillna('Unknown', inplace=True)
matches['winner'].fillna('No Result', inplace=True)
matches['result_margin'].fillna(0, inplace=True)
matches['target_runs'].fillna(0, inplace=True)
matches['target_overs'].fillna(0, inplace=True)
matches.loc[matches['venue'] == 'Sharjah Cricket Stadium', 'city'] = 'Sharjah'
matches.loc[matches['venue'] == 'Dubai International Cricket Stadium', 'city'] = 'Dubai'

team_fixes = {
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
    'Kings XI Punjab':'Punjab Kings',
    
}
deliveries['batting_team'] = deliveries['batting_team'].replace(team_fixes)
deliveries['bowling_team'] = deliveries['bowling_team'].replace(team_fixes)
deliveries['over'] = deliveries['over'] + 1
```

## Data encoding
```python
team_encoder = LabelEncoder()
venue_encoder = LabelEncoder()
decision_encoder = LabelEncoder()

team_encoder.fit(pd.concat([matches['team1'], matches['team2'], matches['toss_winner'], matches['winner']]))
venue_encoder.fit(matches['venue'])
decision_encoder.fit(matches['toss_decision'])
```
## toss modeling
```python
toss_models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVC": SVC()
}

toss_results = []
print("🔍 Toss Model Comparison:\n")
for name, model in toss_models.items():
    model.fit(X_train_toss, y_train_toss)
    y_pred = model.predict(X_test_toss)
    acc = accuracy_score(y_test_toss, y_pred)
    toss_results.append((name, model, acc))
    print(f"{name} Accuracy: {acc:.4f}")
```


## prerdicting toss winner
```python
# Predict Toss Winner
team1 = 'Kolkata Knight Riders'
team2 = 'Royal Challengers Bangalore'
venue = 'Eden Gardens'


toss, winner = predict_winner(team1, team2,venue)

print(f"\n {team1} vs {team2} at {venue}" )
print(" Predicted Toss Winner:", toss)
print(" Predicted Match Winner:", winner)


mask = ((matches['team1'] == team1) & (matches['team2'] == team2)) | \
       ((matches['team1'] == team2) & (matches['team2'] == team1))
h2h_matches = matches[mask]
win_counts = h2h_matches['winner'].value_counts()

plt.figure(figsize=(6,4))
sns.barplot(x=win_counts.index, y=win_counts.values)
plt.title(f'Head-to-Head: {team1} vs {team2}')
plt.ylabel("Wins")
plt.xlabel("Team")
plt.xticks(rotation=45)
plt.show()
```

## ANALYSIS

Match 1 to 10

![alt text](<Screenshot 2025-03-29 at 14-30-05 ipl matches 2025 - Google Search.png>)


## MATCH 1  
## KKR VS RCB

**KKR vs RCB: A Rivalry Through the Years**

The Kolkata Knight Riders (KKR) vs Royal Challengers Bangalore (RCB) rivalry has been one of the most thrilling in IPL history. Over 32 encounters, KKR has dominated with 18 wins, while RCB has triumphed 14 times, making every match a battle of strategy, power-hitting, and game-changing spells.



📅 **Date:** March 22, 2005

📍**Venue:** Eden Gardens, Kolkata

**Insights from Historical Data**

kkr  won 41.94% of tosses at Eden Gardens since 2010.

Kolkata Knight Riders have won 5 of the last 10 matches against RCB.

Teams batting second have a 67.74% win rate at this venue.

**Model Predictions**

**✔ Toss Winner:** Kolkata Knight Riders🏏

**Match Winner:** Kolkata Knight Riders🎉


## snippet
replace this snippet code in predicting toss winner

```python

team1 = 'Kolkata Knight Riders'
team2 = 'Royal Challengers Bangalore'
venue = 'Eden Gardens
```


## MATCH 2

**SRH vs RR: A Battle of Grit and Glory (2008-2024)**

The Sunrisers Hyderabad (SRH) vs Rajasthan Royals (RR) rivalry has been a closely contested battle, with 20 matches played. SRH holds a slight edge with 11 wins, while RR has claimed 9 victories.


📅 **Date:** March 23, 2005

📍**Venue:** Rajiv Gandhi International Stadium, Hyderabad

**Insights from Historical Data**
srh have won 35.06% of tosses at Rajiv Gandhi since 2010.

Sunrisers Hyderabad have won 5 of the last 10 matches against RR.

Teams batting second have a 54.55% win rate at this venue.

**Model Predictions**

**✔ Toss Winner:** Sunrisers Hyderabad🏏

**Match Winner:** Sunrisers Hyderabad🎉

**Code Snippet**

```python
team1 = 'Sunrisers Hyderabad'
team2 = 'Rajasthan Royals'
venue = 'Rajiv Gandhi International Stadium'
```
## MATCH 3
## MI VS CSK


**MI vs CSK: The El Clásico of IPL (2008-2024)**

The Mumbai Indians (MI) vs Chennai Super Kings (CSK) rivalry is the most celebrated in IPL history, with 37 epic clashes. MI leads the battle with 20 wins, while CSK has secured 17 victories, making every encounter a blockbuster.

📅 **Date:** March 23, 2025

📍**Venue:** MA Chidambaram Stadium, Chennai

**Insights from Historical Data**

csk have won 43.53% of tosses at Chennai since 2010.

Chennai Super Kings have won 6 of the last 10 matches against MI.

Teams batting second have a 43.53% win rate at this venue.

**Model Predictions**
**✔ Toss Winner:** Mumbai Indians🏏

**Match Winner:** Chennai Super Kings🎉

**Code Snippet**
```python
team1 = 'Chennai Super Kings'
team2 = 'Mumbai Indians'
venue = 'MA Chidambaram Stadium'
``` 

## MATCH 4
## LSG VS DC

L**SG vs DC: A New-Age Rivalry (2022-2024)**
The Lucknow Super Giants (LSG) vs Delhi Capitals (DC) rivalry is still in its early stages, with 5 thrilling encounters. LSG has the upper hand with 3 wins, while DC has claimed 2 victories, making it a closely fought battle.

📅 **Date:** March 24, 2025

📍**Venue:** ACA-VDCA Stadium, Visakhapatnam

**Insights from Historical Data**

Teams have won 13.46% of tosses at ACA since 2010.

Lucknow Super Giants have won 3 of the last 10 matches against DC.

Teams batting second have a 67.31% win rate at this venue.

**Model Predictions**

**✔ Toss Winner:** Delhi Capitals🏏

**Match Winner:** Lucknow Super Giants🎉

**Code Snippet**
```python 
team1 = 'Delhi Capitals'
team2 = 'Lucknow Super Giants'
venue = 'ACA-VDCA Cricket Stadium'
```
## MATCH 5
## PBKS VS GT

**PBKS vs GT: A New-Age Battle (2022-2024)**

The Punjab Kings (PBKS) vs Gujarat Titans (GT) rivalry is still developing, with 5 intense encounters. GT leads with 3 wins, while PBKS has secured 2 victories, making it a close contest.

**📅 Date:** March 25, 2025

**📍 Venue:** Narendra Modi Stadium, Ahmedabad

**Insights from Historical Data**

Teams have won 55.6% of tosses at Narendra Modi Stadium since 2010.

Punjab Kings have won 4 of the last 10 matches against Gujarat Titans.

Teams batting second have a 62.1% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**
**✔ Toss Winner:** Gujarat Titans 🏏
**🏆 Match Winner:** Gujarat Titans
## code snippet
```python
team1 = 'Gujarat Titans'
team2 = 'Punjab Kings'
venue = 'Narendra Modi Stadium'
```
## MATCH 6
## RR VS KKR

The Rajasthan Royals (RR) vs Kolkata Knight Riders (KKR) rivalry has been one of the most evenly contested battles in IPL history, with 28 thrilling encounters. Both teams have won 14 matches each, making it a true neck-and-neck contest.

**📅 Date:** March 26, 2025

**📍 Venue:** Barsapara Cricket Stadium, Guwahati

**Insights from Historical Data**

Teams have won 48.2% of tosses at Barsapara Cricket Stadium since 2010.

Rajasthan Royals have won 6 of the last 10 matches against Kolkata Knight Riders.

Teams batting second have a 58.4% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Rajasthan Royals 🏏
**🏆 Match Winner:** Rajasthan Royals 🎉

## code snippet
```python
team1 = 'Rajasthan Royals'
team2 = 'Kolkata Knight Riders'
venue = 'Barsapara Stadium'
```

## MATCH 7
## SRH VS LSG

**SRH vs LSG: A Budding Rivalry 

The Sunrisers Hyderabad (SRH) vs Lucknow Super Giants (LSG) clash is still young, with 4 encounters so far. LSG leads with 3 wins, while SRH has claimed 1 victory.

**📅 Date:**March 27, 2025

**📍 Venue:** Rajiv Gandhi International Stadium, Hyderabad

**Insights from Historical Data**

Teams have won 52.3% of tosses at Rajiv Gandhi International Stadium since 2010.

Sunrisers Hyderabad have won 5 of the last 10 matches against Lucknow Super Giants.

Teams batting second have a 60.7% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Sunrisers Hyderabad 🏏
**🏆 Match Winner:** Sunrisers Hyderabad 🎉

## code snippet
```python
team1 = 'Sunrisers Hyderabad'
team2 = 'Lucknow Super Giants'
venue = 'Rajiv Gandhi International Stadium'
```
## MATCH 8
## RCB VS CSK

**RCB vs CSK: A Rivalry of Titans** 

The Royal Challengers Bangalore (RCB) vs Chennai Super Kings (CSK) rivalry has seen 30 epic clashes, with CSK leading the battle with 20 wins, while RCB has won 10 times.

**📅 Date:** March 28, 2025

**📍 Venue:** M. A. Chidambaram Stadium, Chennai

**Insights from Historical Data**

Teams have won 58.2% of tosses at M. A. Chidambaram Stadium since 2010.

Chennai Super Kings have won 7 of the last 10 matches against Royal Challengers Bangalore.

Teams batting second have a 55.4% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Chennai Super Kings 🏏
**🏆 Match Winner:** Chennai Super Kings 🎉

## code snippet
```python
team1 = 'Chennai Super Kings'
team2 = 'Royal Challengers Bangalore'
venue = 'MA Chidambaram Stadium'
```

## MATCH 9
## GT VS MI

**GT vs MI: The New-Age Rivalry**

The Gujarat Titans (GT) vs Mumbai Indians (MI) rivalry is a fresh but exciting contest, with 5 encounters so far. GT leads with 3 wins, while MI has won 2 times.

**📅 Date:**March 29, 2025

**📍 Venue:** Narendra Modi Stadium, Ahmedabad

**Insights from Historical Data**

Teams have won 55.6% of tosses at Narendra Modi Stadium since 2010.

Mumbai Indians have won 6 of the last 10 matches against Gujarat Titans.

Teams batting second have a 62.1% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Mumbai Indians 🏏
**🏆 Match Winner:**Mumbai Indians 🎉
## code snippet
```python
team1 = 'Gujarat Titans'
team2 = 'Mumbai Indians'
venue = 'Narendra Modi Stadium'
```

## MATCH 10
## DC VS SRH 
**DC vs SRH: A Battle of Consistency**

The Delhi Capitals (DC) vs Sunrisers Hyderabad (SRH) rivalry has seen 12 matches, with DC leading the contest with 7 wins, while SRH has won 5 times.

**📅 Date:** March 30, 2025

**📍 Venue:** Arun Jaitley Stadium, Delhi

**Insights from Historical Data**

Teams have won 48.3% of tosses at Arun Jaitley Stadium since 2010.

Sunrisers Hyderabad have won 7 of the last 10 matches against Delhi Capitals.

Teams batting second have a 58.7% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Delhi Capitals 🏏
**🏆 Match Winner:** Sunrisers Hyderabad 🎉
## code snippet
```python 
team1 = 'Delhi Capitals'
team2 = 'Sunrisers Hyderabad'
venue = 'ACA-VDCA Cricket Stadium'
```

![lt text](image.png))

## MATCH 11 TO 20

## MATCH 11
## RR VS CSK

**RR vs CSK: A Clash of Titans (2008-2024)**

The Rajasthan Royals (RR) vs Chennai Super Kings (CSK) rivalry has seen 29 matches, with CSK leading the contest with 16 wins, while RR has secured 13 victories.

**📅 Date:** March 30, 2025

**📍 Venue:** Barsapara Cricket Stadium, Guwahati

**Insights from Historical Data**

Teams have won 50.2% of tosses at Barsapara Cricket Stadium since 2010.

Chennai Super Kings have won 6 of the last 10 matches against Rajasthan Royals.

Teams batting second have a 64.5% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Rajasthan Royals 🏏
**🏆 Match Winner:** Chennai Super Kings 🎉

## code snippet
```python
team1 = 'Rajasthan Royals'
team2 = 'Chennai Super Kings'
venue = 'Barsapara Stadium
```

## MATCH 12
## MI VS KKR 

**MI vs KKR: A Battle of Champions**

The Mumbai Indians (MI) vs Kolkata Knight Riders (KKR) rivalry has seen 34 matches, with MI dominating the contest with 23 wins, while KKR has managed 11 victories.

**📅 Date:** March 31, 2025

**📍 Venue:** Wankhede Stadium, Mumbai

**Insights from Historical Data**

Teams have won 53.7% of tosses at Wankhede Stadium since 2010.

Mumbai Indians have won 7 of the last 10 matches against Kolkata Knight Riders.

Teams batting second have a 61.8% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Mumbai Indians 🏏
**🏆 Match Winner:** Mumbai Indians 🎉
## code snippet
```python
team1 = 'Mumbai Indians'
team2 = 'Kolkata Knight Riders'
venue = 'Wankhede Stadium'
```
## MATCH 13
## LSG VS PBKS 

**LSG vs PBKS: A New Rivalry (2022-2024)**

The Lucknow Super Giants (LSG) vs Punjab Kings (PBKS) rivalry is still in its early days, with 4 matches played. LSG has dominated so far with 3 wins, while PBKS has won 1 match.

**📅 Date:** April 1, 2025

**📍 Venue:** Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur

**Insights from Historical Data**

Teams have won 50.2% of tosses at this venue since 2010.

Lucknow Super Giants have won 3 of the last 5 matches against Punjab Kings.

Teams batting second have a 58.4% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Punjab Kings 🏏
**🏆 Match Winner:** Lucknow Super Giants 🎉
## code snippet 
```python
team1 = 'Lucknow Super Giants'
team2 = 'Punjab Kings'
venue = 'Ekana Cricket Stadium'
```


## MATCH 14
## RCB VS GT

**RCB vs GT: A New Clash**

The Royal Challengers Bangalore (RCB) vs Gujarat Titans (GT) rivalry is relatively new, with 3 matches played. GT has the upper hand with 2 wins, while RCB has won 1 match.

**📅 Date:**April 2, 2025

**📍 Venue:** M. Chinnaswamy Stadium, Bengaluru

**Insights from Historical Data**

Teams have won 48.7% of tosses at this venue since 2010.

Royal Challengers Bangalore have won 6 of the last 10 matches against Gujarat Titans.

Teams batting second have a 64.3% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Gujarat Titans 🏏
**🏆 Match Winner:** Royal Challengers Bangalore 🎉
## code snippet
```python
team1 = 'Royal Challengers Bangalore'
team2 = 'Gujarat Titans'
venue = 'M. Chinnaswamy Stadium'
```

## MATCH 15
## KKR VS SRH 

**KKR vs SRH: A Battle of Titans**

The Kolkata Knight Riders (KKR) vs Sunrisers Hyderabad (SRH) rivalry has seen 28 matches, with KKR leading the head-to-head with 19 wins, while SRH has won 9 matches.

**📅 Date:** April 3, 2025

**📍 Venue:** Eden Gardens, Kolkata

**Insights from Historical Data**

Teams have won 52.8% of tosses at Eden Gardens since 2010.

Kolkata Knight Riders have won 7 of the last 10 matches against Sunrisers Hyderabad.

Teams batting second have a 58.6% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Kolkata Knight Riders 🏏
**🏆 Match Winner:**Kolkata Knight Riders 🎉
## code snippet
```python 
team1 = 'Kolkata Knight Riders'
team2 = 'Sunrisers Hyderabad'
venue = 'Eden Gardens'
```

## MATCH 16
## LSG VS mi

**📅 Date:** April 4, 2025

**📍 Venue:** Ekana Cricket Stadium, Lucknow

**Insights from Historical Data**

Teams have won 50.2% of tosses at Ekana Cricket Stadium since 2010.

Lucknow Super Giants have won 3 of the last 7 matches against Mumbai Indians.

Teams batting second have a 61.4% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Mumbai Indians 🏏
**🏆 Match Winner:** Lucknow Super Giants 🎉
## code snippet
```python
team1 = 'Lucknow Super Giants'
team2 = 'Mumbai Indians'
venue = 'Ekana Cricket Stadium'
```


## MATCH 17 
## CSK VS DC
**CSK vs DC: A Clash of Generations**

The Chennai Super Kings (CSK) vs Delhi Capitals (DC) rivalry has seen 12 matches, with CSK leading the head-to-head with 7 wins, while DC has won 5 matches.

**📅 Date:** April 5, 2035

**📍 Venue:** MA Chidambaram Stadium, Chennai

**Insights from Historical Data**

Teams have won 48.3% of tosses at MA Chidambaram Stadium since 2010.

Chennai Super Kings have won 7 of the last 10 matches against Delhi Capitals.

Teams batting first have a 64.7% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Chennai Super Kings 🏏
**🏆 Match Winner:** Chennai Super Kings 🎉

## snippet 
```python

team1 = 'Chennai Super Kings'
team2 = 'Delhi Capitals'
venue = 'MA Chidambaram Stadium'
```

## MATCH 18
## PBKS VS RR
**PBKS vs RR: A Battle of Unpredictability (2008-2024)**

The Punjab Kings (PBKS) vs Rajasthan Royals (RR) rivalry has seen 7 matches, with RR leading the head-to-head with 4 wins, while PBKS has won 3 matches.

**📅 Date:** April 5, 2025
**📍 Venue:** Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur

**Insights from Historical Data**

Teams have won 52.3% of tosses at this venue since 2010.

Punjab Kings have won 6 of the last 10 matches against Rajasthan Royals.

Teams batting second have a 58.7% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Punjab Kings 🏏
**🏆 Match Winner:** Punjab Kings

## snippet
```python

team1 = 'The Punjab Kings'
team2 = 'Rajasthan Royals'
venue = 'Maharaja Yadavindra Singh International Cricket Stadium'
```



## MATCH 19
## SRH VS GT
**SRH vs GT: The New-Age Rivalry**

The Sunrisers Hyderabad (SRH) vs Gujarat Titans (GT) rivalry has seen 4 matches, with GT leading the head-to-head with 3 wins, while SRH has won 1 match.

**📅 Date:** April 6, 2025

**📍 Venue:** Rajiv Gandhi International Stadium, Hyderabad

**Insights from Historical Data**

Teams have won 48.9% of tosses at this venue since 2010.

Sunrisers Hyderabad have won 5 of the last 10 matches against Gujarat Titans.

Teams batting second have a 61.4% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**
**✔ Toss Winner:** Gujarat Titans 🏏
**🏆 Match Winner:** Sunrisers Hyderabad

## snippet 
```python

team1 = 'he Sunrisers Hyderabad'
team2 = 'Gujarat Titans'
venue = 'Rajiv Gandhi International Stadium'
```

## MATCH 20
## MI VS RCB

**MI vs RCB: A Clash of Titans**

The Mumbai Indians (MI) vs Royal Challengers Bangalore (RCB) rivalry has seen 32 matches, with MI leading the head-to-head with 18 wins, while RCB has won 14 matches.

**📅 Date:** April 7, 2025

**📍 Venue:** Wankhede Stadium, Mumbai
Insights from Historical Data
Teams have won 51.2% of tosses at Wankhede Stadium since 2010.

Mumbai Indians have won 6 of the last 10 matches against Royal Challengers Bangalore.

Teams batting second have a 59.8% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Mumbai Indians 🏏

**🏆 Match Winner:** Mumbai Indians 🎉
## snippet 
```python

team1 = 'The Mumbai Indians'
team2 = 'Royal Challengers Bangalore'
venue = 'Wankhede Stadium, Mumbai'
```

## MATCH 11 TO 20
![alt text](image-1.png))
## MATCH 21
## KKR VS LSG
**KKR vs LSG: A Growing Rivalry (2022-2024)**

The Kolkata Knight Riders (KKR) vs Lucknow Super Giants (LSG) rivalry has seen 5 matches, with LSG leading the head-to-head with 3 wins, while KKR has won 2 matches.

**📅 Date:** April 8, 2025

**📍 Venue:** Eden Gardens, Kolkata

**Insights from Historical Data**

Teams have won 53.7% of tosses at Eden Gardens since 2010.

Kolkata Knight Riders have won 5 of the last 10 matches against Lucknow Super Giants.

Teams batting second have a 61.2% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:** Kolkata Knight Riders 🏏

**🏆 Match Winner:** Kolkata Knight Riders 🎉
## snippet 
```python

team1 = 'The Kolkata Knight Riders '
team2 = 'Lucknow Super Giants'
venue = 'Eden Gardens'
```

## MATCH 22
## PBKS VS CSK
**PBKS vs CSK: A Fierce Battle**

The Punjab Kings (PBKS) vs Chennai Super Kings (CSK) rivalry has seen 7 matches, with PBKS leading the head-to-head with 5 wins, while CSK has won 2 matches.

**📅 Date:** April 8, 2025

**📍 Venue:** Maharaja Yadavindra Singh International Cricket Stadium

**Insights from Historical Data**

Teams have won 50.2% of tosses at Maharaja Yadavindra Singh Stadium since 2010.

Punjab Kings have won 4 of the last 10 matches against Chennai Super Kings.

Teams batting second have a 58.9% win rate at this venue.

Model Predictions (Based on Historical Data & Machine Learning Models)

**✔ Toss Winner:** Chennai Super Kings 🏏
**🏆 Match Winner:** Chennai Super Kings 🎉

## snippet 
```python

team1 = 'The Punjab Kings'
team2 = 'Chennai Super Kings'
venue = 'Maharaja Yadavindra Singh International Cricket Stadium'
```

## MATCH 23
## GT VS RR
**GT vs RR: The New Age Battle**

The Gujarat Titans (GT) vs Rajasthan Royals (RR) rivalry has seen 6 matches, with GT dominating the head-to-head with 5 wins, while RR has won just 1 match.

**📅 Date:** April 9, 2025

**📍 Venue:** Narendra Modi Stadium, Ahmedabad

**Insights from Historical Data**

Teams have won 55.6% of tosses at Narendra Modi Stadium since 2010.

Gujarat Titans have won 6 of the last 10 matches against Rajasthan Royals.

Teams batting second have a 62.1% win rate at this venue.

**Model Predictions (Based on Historical Data & Machine Learning Models)**

**✔ Toss Winner:**Gujarat Titans 🏏
**🏆 Match Winner:**Gujarat Titans 🎉

## snippet 
```python

team1 = 'The Gujarat Titans '
team2 = 'Rajasthan Royals'
venue = 'Narendra Modi Stadium, Ahmedabad
```

## MATCH 24
## RCB VS DC
RCB vs DC: A Competitive Rivalry (2008-2024)

The Royal Challengers Bangalore (RCB) vs Delhi Capitals (DC) rivalry has seen 9 matches, with DC leading the head-to-head with 5 wins, while RCB has won 4 matches.

**📅 Date:** April 10, 2025  

**📍 Venue:** M. Chinnaswamy Stadium, Bengaluru  

**Insights from Historical Data**  

- Teams have won 50.2% of tosses at M. Chinnaswamy Stadium since 2010.  
- Royal Challengers Bangalore have won 6 of the last 10 matches against Delhi Capitals.  
- Teams batting second have a 58.4% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Delhi Capitals 🏏  
**🏆 Match Winner:** Royal Challengers Bangalore 🎉 

## snippet 
```python

team1 = 'The Royal Challengers Bangalore'
team2 = 'Delhi Capitals'
venue = 'M. Chinnaswamy Stadium, Bengaluru  '
```

## CSK VS KKR 
**CSK vs KKR: A Historic Rivalry**

The Chennai Super Kings (CSK) vs Kolkata Knight Riders (KKR) rivalry has seen 14 matches, with CSK leading the head-to-head with 9 wins, while KKR has won 5 matches.

**📅 Date:** April 11, 2025  

**📍 Venue:** MA Chidambaram Stadium, Chennai  

**Insights from Historical Data**  

- Teams have won 48.3% of tosses at MA Chidambaram Stadium since 2010.  
- Chennai Super Kings have won 7 of the last 10 matches against Kolkata Knight Riders.  
- Teams batting first have a 64.7% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Kolkata Knight Riders 🏏  
**🏆 Match Winner:** Chennai Super Kings 🎉

## snippet 
```python

team1 = 'Kolkata Knight Riders'
team2 = 'Chennai Super Kings'
venue = 'MA Chidambaram Stadium, Chennai 
```

## MATCH 26
## LSG VS GT 
**LSG vs GT: A New-Age Rivalry**

The Lucknow Super Giants (LSG) vs Gujarat Titans (GT) rivalry has seen 5 matches, with GT dominating the head-to-head with 4 wins, while LSG has won 1 match.

**📅 Date:** April 12, 2025  

**📍 Venue:** Ekana Cricket Stadium, Lucknow  

**Insights from Historical Data**  

- Teams have won 51.2% of tosses at Ekana Cricket Stadium since 2010.  
- Lucknow Super Giants have won 4 of the last 10 matches against Gujarat Titans.  
- Teams batting second have a 59.8% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Gujarat Titans 🏏  
**🏆 Match Winner:** Lucknow Super Giants 🎉  

## snippet 
```python

team1 = 'Gujarat Titans
team2 = 'Lucknow Super Giants'
venue = 'Ekana Cricket Stadium, Lucknow  '
```

## MATCH 27
## SRH VS PBKS

**SRH vs PBKS**

The rivalry between Sunrisers Hyderabad (SRH) and Punjab Kings (PBKS) may not be as intense as some other IPL matchups, but it has produced thrilling encounters over the years.

**📅 Date:** April 12, 2025  

**📍 Venue:** Rajiv Gandhi International Stadium, Hyderabad  

**Insights from Historical Data**  

- Teams have won 50.5% of tosses at Rajiv Gandhi International Stadium since 2010.  
- Sunrisers Hyderabad have won 6 of the last 10 matches against Punjab Kings.  
- Teams batting second have a 61.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Sunrisers Hyderabad 🏏  
**🏆 Match Winner:** Sunrisers Hyderabad 🎉  

## snippet 
```python

team1 = 'Sunrisers Hyderabad'
team2 = 'Punjab Kings'
venue = ' Rajiv Gandhi International Stadium, Hyderabad  
```


## MATCH 28
## RR VS RCB 

**Rajasthan Royals vs Royal Challengers Bangalore**

The rivalry between Rajasthan Royals (RR) and Royal Challengers Bangalore (RCB) has been one of the most exciting in IPL history. Both teams have had their moments of dominance, with several high-scoring encounters and thrilling finishes.

**📅 Date:** April 13, 2025  

**📍 Venue:** Sawai Mansingh Stadium, Jaipur  

**Insights from Historical Data**  

- Teams have won 52.7% of tosses at Sawai Mansingh Stadium since 2010.  
- Rajasthan Royals have won 5 of the last 10 matches against Royal Challengers Bangalore.  
- Teams batting second have a 58.9% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Rajasthan Royals 🏏  
**🏆 Match Winner:** Rajasthan Royals 🎉  

## snippet 
```python

team1 = 'Rajasthan Royals
team2 = 'Royal Challengers Bangalore'
venue = ' Sawai Mansingh Stadium, Jaipur '
```


## MATCH 29
## DC VS MI
**Delhi Capitals vs Mumbai Indians**

The battle between Delhi Capitals (DC) and Mumbai Indians (MI) has been a tale of two teams with different legacies—MI being the most successful IPL franchise and DC emerging as a strong competitor in recent years.
**📅 Date:** April 13, 2025  

**📍 Venue:** Arun Jaitley Stadium, Delhi  

**Insights from Historical Data**  

- Teams have won 49.2% of tosses at Arun Jaitley Stadium since 2010.  
- Delhi Capitals have won 4 of the last 10 matches against Mumbai Indians.  
- Teams batting second have a 60.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Mumbai Indians 🏏  
**🏆 Match Winner:** Mumbai Indians 🎉  

## snippet 
```python

team1 = 'Mumbai Indians'
team2 = 'Delhi Capitals 
venue = ' Arun Jaitley Stadium, Delhi'
```

 
## MATCH 30
## LSG VS CSK
**Lucknow Super Giants vs Chennai Super Kings (2022-2024)**

Despite being one of the newer IPL teams, Lucknow Super Giants (LSG) has built an exciting rivalry with the Chennai Super Kings (CSK) since their debut in 2022.

**📅 Date:** April 15, 2025  

**📍 Venue:** Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur  

**Insights from Historical Data**  

- Teams have won 50.2% of tosses at this venue since 2010.  
- Punjab Kings have won 4 of the last 10 matches against Rajasthan Royals.  
- Teams batting second have a 58.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Rajasthan Royals 🏏  
**🏆 Match Winner:** Rajasthan Royals 🎉  
## snippet 
```python

team1 = 'Lucknow Super Giants'
team2 = ' chennai Super Kings'
venue = ' Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur '
```

## MATCH 31 TO 40

![alt text](image-2.png)
## MATCH 31
## PBKS VS KKR 

**📅 Date:** April 15, 2025  

**📍 Venue:** Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur  

**Insights from Historical Data**  

- Teams have won 50.2% of tosses at this venue since 2010.  
- Punjab Kings have won 3 of the last 10 matches against Kolkata Knight Riders.  
- Teams batting second have a 58.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Kolkata Knight Riders 🏏  
**🏆 Match Winner:** Kolkata Knight Riders 🎉  
## snippet 
```python

team1 = 'Kolkata Knight Riders'
team2 = ' Punjab Kings'
venue = ' Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur '
```


## MATCH 32
## DC VS RR
Delhi Capitals vs Rajasthan Royals Rivalry (2019-2024)
The battle between Delhi Capitals (DC) and Rajasthan Royals (RR) has been competitive, with DC leading the head-to-head encounters. However, RR has managed to pull off some big-margin victories.

**📅 Date:** April 16, 2025  

**📍 Venue:** Arun Jaitley Stadium, Delhi  

**Insights from Historical Data**  

- Teams have won 47.8% of tosses at this venue since 2010.  
- Delhi Capitals have won 6 of the last 10 matches against Rajasthan Royals.  
- Teams batting second have a 60.5% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Delhi Capitals 🏏  
**🏆 Match Winner:** Delhi Capitals 🎉
## snippet 
```python

team1 = 'Delhi Capitals'
team2 = ' Rajasthan Royals'
venue = ' Arun Jaitley Stadium, Delhi
```

## MATCH 33 
## MI VS SRH
**Mumbai Indians vs Sunrisers Hyderabad Rivalry (2013-2024)**

The Mumbai Indians (MI) vs Sunrisers Hyderabad (SRH) rivalry has been intense, with MI holding a slight edge over SRH in their head-to-head clashes.

**📅 Date:** April 17, 2025  

**📍 Venue:** Wankhede Stadium, Mumbai  

**Insights from Historical Data**  

- Teams have won 50.2% of tosses at Wankhede Stadium since 2010.  
- Mumbai Indians have won 7 of the last 10 matches against Sunrisers Hyderabad.  
- Teams batting second have a 65.8% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Mumbai Indians 🏏  
**🏆 Match Winner:** Mumbai Indians 🎉  
## snippet 
```python

team1 = ' Mumbai Indians'
team2 = ' Sunrisers Hyderabad'
venue = ' Wankhede Stadium, Mumbai  '
```


## MATCH 34
## RCB VS PBKS
**RCB vs PBKS Rivalry**
The Royal Challengers Bangalore (RCB) vs Punjab Kings (PBKS) rivalry has been closely contested, with PBKS having a slight edge in recent years.

**📅 Date:** April 18, 2025  

**📍 Venue:** M. Chinnaswamy Stadium, Bengaluru  

**Insights from Historical Data**  

- Teams have won 47.6% of tosses at M. Chinnaswamy Stadium since 2010.  
- Royal Challengers Bangalore have won 6 of the last 10 matches against Punjab Kings.  
- Teams batting second have a 61.4% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Punjab Kings 🏏  
**🏆 Match Winner:** Royal Challengers Bangalore 🎉  
## snippet 
```python

team1 = 'Punjab Kings '
team2 = ' Royal Challengers Bangalore'
venue = '  M. Chinnaswamy Stadium, Bengaluru'
```

## MATCH 35
## GT VS DC
**GT vs DC Rivalry (2022-2024)**

The Gujarat Titans (GT) vs Delhi Capitals (DC) rivalry is relatively new but has already produced some exciting encounters.

**Head-to-Head Record
Total Matches:** 5

**📅 Date:** April 19, 2025  

**📍 Venue:** Arun Jaitley Stadium, Delhi  

**Insights from Historical Data**  

- Teams have won 49.2% of tosses at Arun Jaitley Stadium since 2010.  
- Delhi Capitals have won 4 of the last 10 matches against Gujarat Titans.  
- Teams batting second have a 58.7% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Gujarat Titans 🏏  
**🏆 Match Winner:** Gujarat Titans 🎉  
## snippet 
```python

team1 = ' Gujarat Titans
team2 = 'Delhi Capitals
venue = ' Arun Jaitley Stadium, Delhi  '
```

## MATCH 36
## RR VS LSG
**RR vs LSG Rivalry (2022-2024)**

The Rajasthan Royals (RR) vs Lucknow Super Giants (LSG) rivalry is fairly new but has been one-sided in favor of RR so far.

**📅 Date:** April 19, 2025  

**📍 Venue:** Sawai Mansingh Stadium, Jaipur  

**Insights from Historical Data**  

- Teams have won 51.3% of tosses at Sawai Mansingh Stadium since 2010.  
- Rajasthan Royals have won 6 of the last 10 matches against Lucknow Super Giants.  
- Teams batting first have a 60.4% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Rajasthan Royals 🏏  
**🏆 Match Winner:** Rajasthan Royals 🎉
## snippet 
```python

team1 = 'Rajasthan Royals'
team2 = 'Lucknow Super Giants'
venue = 'Sawai Mansingh Stadium, Jaipur  '
```

## MATCH 37
## PBKS VS RCB
**PBKS vs RCB Rivalry (2021-2023)**

The Punjab Kings (PBKS) vs Royal Challengers Bangalore (RCB) rivalry has been closely fought in recent years, with PBKS having a slight advantage.

**📅 Date:** April 20, 2025  

**📍 Venue:** Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur  

**Insights from Historical Data**  

- Teams have won 49.8% of tosses at this venue since 2010.  
- Punjab Kings have won 4 of the last 10 matches against Royal Challengers Bangalore.  
- Teams batting second have a 58.2% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Royal Challengers Bangalore 🏏  
**🏆 Match Winner:** Royal Challengers Bangalore 🎉  


## snippet 
```python

team1 = 'Royal Challengers Bangalore
team2 = 'Punjab Kings'
venue = 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur '
```

## MATCH 38
## MI VS CSK

**MI vs CSK: The El Clásico of IPL**

The Mumbai Indians (MI) vs Chennai Super Kings (CSK) rivalry is the most celebrated in IPL history, with 37 epic clashes. MI leads the battle with 20 wins, while CSK has secured 17 victories, making every encounter a blockbuster.

**📅 Date:** April 20, 2025  

**📍 Venue:** MA Chidambaram Stadium, Chennai  

**Insights from Historical Data**  

- Teams have won 48.3% of tosses at MA Chidambaram Stadium since 2010.  
- Chennai Super Kings have won 6 of the last 10 matches against Mumbai Indians.  
- Teams batting first have a 64.7% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Chennai Super Kings 🏏  
**🏆 Match Winner:** Chennai Super Kings 🎉

## snippet 
```python

team1 = 'Chennai Super Kings
team2 = 'Mumbai Indians'
venue = 'MA Chidambaram Stadium, Chennai'
```

## MATCH 39
## KKR VS GT

**KKR vs GT: A Battle of Powerhouses**  

Kolkata Knight Riders (KKR) and Gujarat Titans (GT) have faced off multiple times, with both teams showcasing their strengths. While KKR has experience and two IPL titles, GT has quickly risen as a formidable team since its debut.  

**📅 Date:** April 21, 2025  

**📍 Venue:** Eden Gardens, Kolkata  

**Insights from Historical Data**  

- Teams have won 50.2% of tosses at Eden Gardens since 2010.  
- Kolkata Knight Riders have won 4 of the last 10 matches against Gujarat Titans.  
- Teams chasing have a 61.5% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Gujarat Titans 🏏  
**🏆 Match Winner:** Gujarat Titans 🎉  

## snippet 
```python

team1 = 'Gujarat Titans 
team2 = 'Kolkata Knight Riders
venue = 'Eden Gardens, Kolkata '
```
## MATCH 40 
## LSG VS DC
**LSG vs DC: A Clash of Rising Contenders**  

Lucknow Super Giants (LSG) and Delhi Capitals (DC) have developed an exciting rivalry, with both teams bringing young talent and strategic gameplay to the field.  

**📅 Date:** April 22, 2025  

**📍 Venue:** Ekana Cricket Stadium, Lucknow  

**Insights from Historical Data**  

- Teams have won 47.8% of tosses at Ekana Cricket Stadium since 2010.  
- Lucknow Super Giants have won 3 of the last 5 matches against Delhi Capitals.  
- Teams batting second have a 59.2% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Lucknow Super Giants 🏏  
**🏆 Match Winner:** Lucknow Super Giants 🎉  

## snippet 
```python

team1 = 'Lucknow Super Giants'
team2 = 'Delhi Capitals'
venue = 'Ekana Cricket Stadium, Lucknow'
```

## MATCH 41 TO 50
![alt text](image-3.png)
## MATCH 41
## SRH VS MI

**SRH vs MI: Battle of the Powerhouses**  

Sunrisers Hyderabad (SRH) and Mumbai Indians (MI) have had some thrilling encounters, with MI historically having the upper hand in their clashes.  

**📅 Date:** April 23, 2025  

**📍 Venue:** Rajiv Gandhi International Stadium, Hyderabad  

**Insights from Historical Data**  

- Teams have won 51.2% of tosses at Rajiv Gandhi International Stadium since 2010.  
- Mumbai Indians have won 6 of the last 10 matches against Sunrisers Hyderabad.  
- Teams batting second have a 57.5% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Mumbai Indians 🏏  
**🏆 Match Winner:** Mumbai Indians 🎉  
## snippet 
```python

team1 = 'Mumbai Indians '
team2 = 'Sunrisers Hyderabad'
venue = 'Rajiv Gandhi International Stadium, Hyderabad '
```

## MATCH 42
## RCB VS RR
**RCB vs RR: A Royal Clash**  

Royal Challengers Bangalore (RCB) and Rajasthan Royals (RR) have had an intense rivalry, with both teams producing thrilling encounters over the years.  

**📅 Date:** April 24, 2025  

**📍 Venue:** M. Chinnaswamy Stadium, Bengaluru  

**Insights from Historical Data**  

- Teams have won 50.8% of tosses at M. Chinnaswamy Stadium since 2010.  
- Royal Challengers Bangalore have won 5 of the last 10 matches against Rajasthan Royals.  
- Teams chasing have a 59.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Royal Challengers Bangalore 🏏  
**🏆 Match Winner:** Royal Challengers Bangalore 🎉 
## snippet 
```python

team1 = ' Royal Challengers Bangalore'
team2 = 'Rajasthan Royals'
venue = ' M. Chinnaswamy Stadium, Bengaluru '
```

## MATCH 43
## CSK VS SRH

**CSK vs SRH: The Battle of the Yellows**  

Chennai Super Kings (CSK) and Sunrisers Hyderabad (SRH) have had some intense battles, with CSK historically dominating the rivalry.  

**📅 Date:** April 25, 2025  

**📍 Venue:** MA Chidambaram Stadium, Chennai  

**Insights from Historical Data**  

- Teams have won 48.3% of tosses at MA Chidambaram Stadium since 2010.  
- Chennai Super Kings have won 7 of the last 10 matches against Sunrisers Hyderabad.  
- Teams batting first have a 64.7% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Chennai Super Kings 🏏  
**🏆 Match Winner:** Chennai Super Kings 🎉 
## snippet 
```python

team1 = 'Chennai Super Kings'
team2 = 'Sunrisers Hyderabad'
venue = ' MA Chidambaram Stadium, Chennai'
```

## MATCH 44  
## KKR VS PBKS  

**KKR vs PBKS: A Fierce Rivalry in IPL**  

Kolkata Knight Riders (KKR) and Punjab Kings (PBKS) have shared an intense rivalry over the years, producing some thrilling encounters.  

**📅 Date:** April 26, 2025  

**📍 Venue:** Eden Gardens, Kolkata  

**Insights from Historical Data**  

- Teams have won 51.2% of tosses at Eden Gardens since 2010.  
- Kolkata Knight Riders have won 6 of the last 10 matches against Punjab Kings.  
- Teams batting second have a 60.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Kolkata Knight Riders 🏏  
**🏆 Match Winner:** Kolkata Knight Riders 🎉  

## snippet 
```python

team1 = 'Kolkata Knight Riders'
team2 = 'Punjab Kings'
venue = ' Eden Gardens, Kolkata'
```

## MATCH 45  
## MI VS LSG  

**MI vs LSG: A Clash of Powerhouses**  

Mumbai Indians (MI) and Lucknow Super Giants (LSG) have built a competitive rivalry in recent IPL seasons, with both teams boasting strong squads.  

**📅 Date:** April 27, 2025  

**📍 Venue:** Wankhede Stadium, Mumbai  

**Insights from Historical Data**  

- Teams have won 55.6% of tosses at Wankhede Stadium since 2010.  
- Mumbai Indians have won 4 of the last 10 matches against Lucknow Super Giants.  
- Teams batting second have a 62.1% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Mumbai Indians 🏏  
**🏆 Match Winner:** Mumbai Indians 🎉  

## snippet 
```python

team1 = 'Mumbai Indians'
team2 = 'Lucknow Super Giants'
venue = 'Wankhede Stadium, Mumbai'
```

## MATCH 46  
## DC VS RCB  

**DC vs RCB: A Battle for Supremacy**  

Delhi Capitals (DC) and Royal Challengers Bangalore (RCB) have had a competitive rivalry, with both teams showcasing top talent over the years.  

**📅 Date:** April 27, 2025  

**📍 Venue:** Arun Jaitley Stadium, Delhi  

**Insights from Historical Data**  

- Teams have won 50.2% of tosses at Arun Jaitley Stadium since 2010.  
- Royal Challengers Bangalore have won 6 of the last 10 matches against Delhi Capitals.  
- Teams batting second have a 58.4% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Royal Challengers Bangalore 🏏  
**🏆 Match Winner:** Royal Challengers Bangalore 🎉 
## snippet 
```python

team1 = 'Royal Challengers Bangalore '
team2 = 'Delhi Capitals'
venue = 'Arun Jaitley Stadium, Delhi '
```
## MATCH 47  
## RR VS GT  

**RR vs GT: Clash of the Titans**  

Rajasthan Royals (RR) and Gujarat Titans (GT) have developed a fierce rivalry in recent seasons, with both teams boasting strong squads.  

**📅 Date:** April 28, 2025  

**📍 Venue:** Sawai Mansingh Stadium, Jaipur  

**Insights from Historical Data**  

- Teams have won 52.1% of tosses at Sawai Mansingh Stadium since 2010.  
- Gujarat Titans have won 5 of the last 8 matches against Rajasthan Royals.  
- Teams batting first have a 59.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Gujarat Titans 🏏  
**🏆 Match Winner:** Gujarat Titans 🎉  
## snippet 
```python

team1 = ' Gujarat Titans'
team2 = 'Rajasthan Royals '
venue = 'Sawai Mansingh Stadium, Jaipur
```

## MATCH 48  
## DC VS KKR  

**DC vs KKR: A Rivalry Renewed**  

Delhi Capitals (DC) and Kolkata Knight Riders (KKR) have had thrilling encounters over the years, with both teams fighting for dominance.  

**📅 Date:** April 29, 2025  

**📍 Venue:** Arun Jaitley Stadium, Delhi  

**Insights from Historical Data**  

- Teams have won 50.2% of tosses at Arun Jaitley Stadium since 2010.  
- Kolkata Knight Riders have won 6 of the last 10 matches against Delhi Capitals.  
- Teams batting second have a 55.8% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Delhi Capitals 🏏  
**🏆 Match Winner:** Kolkata Knight Riders 🎉
**🏆 Match Winner:** Gujarat Titans 🎉  
## snippet 
```python

team1 = 'Delhi Capitals'
team2 = 'Kolkata Knight Riders'
venue = 'Arun Jaitley Stadium, Delhi  '
```


## MATCH 49  
## CSK VS PBKS  

**CSK vs PBKS: Clash of the Titans**  

Chennai Super Kings (CSK) and Punjab Kings (PBKS) have had some memorable encounters, with CSK historically holding the edge.  

**📅 Date:** April 30, 2025  

**📍 Venue:** MA Chidambaram Stadium, Chennai  

**Insights from Historical Data**  

- Teams have won 48.3% of tosses at MA Chidambaram Stadium since 2010.  
- Chennai Super Kings have won 7 of the last 10 matches against Punjab Kings.  
- Teams batting first have a 64.7% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Chennai Super Kings 🏏  
**🏆 Match Winner:** Chennai Super Kings 🎉  
## snippet 
```python

team1 = ' Chennai Super Kings
team2 = 'Punjab Kings
venue = 'MA Chidambaram Stadium, Chennai
```

## MATCH 50  
## RR VS MI  

**RR vs MI: A High-Stakes Encounter**  

Rajasthan Royals (RR) and Mumbai Indians (MI) have had a competitive rivalry, with both teams showcasing top performances over the years.  

**📅 Date:** May 1, 2025  

**📍 Venue:** Sawai Mansingh Stadium, Jaipur  

**Insights from Historical Data**  

- Teams have won 52.4% of tosses at Sawai Mansingh Stadium since 2010.  
- Mumbai Indians have won 6 of the last 10 matches against Rajasthan Royals.  
- Teams chasing have a 59.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Mumbai Indians 🏏  
**🏆 Match Winner:** Mumbai Indians 🎉  

## snippet 
```python

team1 = 'Mumbai Indians'
team2 = 'Rajasthan Royals'
venue = ' Sawai Mansingh Stadium, Jaipur ' 
```
## MATCH 51 TO 60
![alt text](image-4.png)

## MATCH 51  
## GT VS SRH  

**GT vs SRH: The Clash of Titans**  

Gujarat Titans (GT) and Sunrisers Hyderabad (SRH) have developed an exciting rivalry, with both teams displaying strong performances in recent seasons.  

**📅 Date:** May 2, 2025  

**📍 Venue:** Narendra Modi Stadium, Ahmedabad  

**Insights from Historical Data**  

- Teams have won 55.6% of tosses at Narendra Modi Stadium since 2010.  
- Gujarat Titans have won 5 of the last 10 matches against Sunrisers Hyderabad.  
- Teams batting second have a 62.1% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Gujarat Titans 🏏  
**🏆 Match Winner:** Gujarat Titans 🎉  

## snippet 
```python

team1 = 'Gujarat Titans'
team2 = 'Sunrisers Hyderabad'
venue = ' Narendra Modi Stadium, Ahmedabad  '
```
## MATCH 52  
## RCB VS CSK  

**RCB vs CSK: The Southern Derby**  

Royal Challengers Bangalore (RCB) and Chennai Super Kings (CSK) share one of the most intense rivalries in IPL history. With passionate fanbases and star-studded lineups, every encounter between these two giants is a spectacle.  

**📅 Date:** April 3, 2025  

**📍 Venue:** M. Chinnaswamy Stadium, Bengaluru  

**Insights from Historical Data**  

- Teams have won 49.2% of tosses at M. Chinnaswamy Stadium since 2010.  
- Chennai Super Kings have won 7 of the last 10 matches against Royal Challengers Bangalore.  
- Teams batting second have a 58.4% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Chennai Super Kings 🏏  
**🏆 Match Winner:** Chennai Super Kings 🎉  

## snippet 
```python

team1 = 'Chennai Super Kings'
team2 = 'Royal Challengers Bangalore '
venue = '  M. Chinnaswamy Stadium, Bengaluru  
```

## MATCH 53  
## KKR VS RR  

**KKR vs RR: The Clash of Titans**  

Kolkata Knight Riders (KKR) and Rajasthan Royals (RR) have had several close encounters in IPL history. Both teams bring a mix of experience and young talent, making this match an exciting showdown.  

**📅 Date:** April 4, 2025  

**📍 Venue:** Eden Gardens, Kolkata  

**Insights from Historical Data**  

- Teams have won 51.2% of tosses at Eden Gardens since 2010.  
- Kolkata Knight Riders have won 6 of the last 10 matches against Rajasthan Royals.  
- Teams batting second have a 61.5% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Kolkata Knight Riders 🏏  
**🏆 Match Winner:** Kolkata Knight Riders 🎉 

## snippet 
```python

team1 = 'Kolkata Knight Riders '
team2 = 'Rajasthan Royals'
venue = ' Eden Gardens, Kolkata '
```

## MATCH 54  
## PBKS VS LSG  

**PBKS vs LSG: A Battle for Dominance**  

Punjab Kings (PBKS) and Lucknow Super Giants (LSG) have shown fierce competition in recent IPL seasons. With both teams looking to climb the points table, this match promises high-intensity cricket.  

**📅 Date:** April 4, 2025  

**📍 Venue:** Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur  

**Insights from Historical Data**  

- Teams have won 49.8% of tosses at this venue since 2010.  
- Punjab Kings have won 5 of the last 10 matches against Lucknow Super Giants.  
- Teams batting second have a 58.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Lucknow Super Giants 🏏  
**🏆 Match Winner:** Lucknow Super Giants 🎉  
## snippet 
```python

team1 = 'Lucknow Super Giants'
team2 = 'Punjab Kings'
venue = ' Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur'
```

## MATCH 55  
## SRH VS DC  

**SRH vs DC: The Clash of Titans**  

Sunrisers Hyderabad (SRH) and Delhi Capitals (DC) have shared an intense rivalry over the years. Both teams are known for their explosive batting and lethal bowling attacks, making this encounter one to watch.  

**📅 Date:** May 5, 2025  

**📍 Venue:** Rajiv Gandhi International Cricket Stadium, Hyderabad  

**Insights from Historical Data**  

- Teams have won 51.2% of tosses at this venue since 2010.  
- Sunrisers Hyderabad have won 6 of the last 10 matches against Delhi Capitals.  
- Teams batting second have a 60.4% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Sunrisers Hyderabad 🏏  
**🏆 Match Winner:** Sunrisers Hyderabad 🎉  ## snippet 
```python

team1 = ' Sunrisers Hyderabad'
team2 = 'Delhi Capitals'
venue = ' Rajiv Gandhi International Cricket Stadium, Hyderabad  '
```
## MATCH 56  
## MI VS GT  

**MI vs GT: Powerhouses Collide**  

Mumbai Indians (MI) and Gujarat Titans (GT) are two of the most dynamic teams in the IPL. While MI boasts a rich history of championships, GT has made a strong impact since its debut.  

**📅 Date:** May 6, 2025  

**📍 Venue:** Wankhede Stadium, Mumbai  

**Insights from Historical Data**  

- Teams have won 54.2% of tosses at Wankhede Stadium since 2010.  
- Mumbai Indians have won 5 of the last 10 matches against Gujarat Titans.  
- Teams chasing have a 59.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Mumbai Indians 🏏  
**🏆 Match Winner:** Mumbai Indians 🎉  

```python

team1 = ' Mumbai Indians
team2 ='Gujarat Titans'
venue = ' Wankhede Stadium, Mumbai 
```

## MATCH 57  
## KKR VS CSK  

**KKR vs CSK: A Clash of Titans**  

Kolkata Knight Riders (KKR) and Chennai Super Kings (CSK) have built an exciting rivalry over the years. With CSK’s consistency and KKR’s aggressive approach, this matchup is always thrilling.  

**📅 Date:** May 7, 2025  

**📍 Venue:** Eden Gardens, Kolkata  

**Insights from Historical Data**  

- Teams have won 51.8% of tosses at Eden Gardens since 2010.  
- Chennai Super Kings have won 6 of the last 10 matches against Kolkata Knight Riders.  
- Teams chasing have a 61.4% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Chennai Super Kings 🏏  
**🏆 Match Winner:** Chennai Super Kings 🎉  

```python

team1 = 'Chennai Super Kings '
team2 ='Chennai Super Kings'
venue = 'Eden Gardens, Kolkata
```

## MATCH 58  
## PBKS VS DC  

**PBKS vs DC: The Northern Derby**  

Punjab Kings (PBKS) and Delhi Capitals (DC) have had a competitive history in the IPL, with both teams striving for dominance in the northern clash.  

**📅 Date:** May 8, 2025  

**📍 Venue:** Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur  

**Insights from Historical Data**  

- Teams have won 50.2% of tosses at this venue since 2010.  
- Delhi Capitals have won 6 of the last 10 matches against Punjab Kings.  
- Teams batting second have a 58.7% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Delhi Capitals 🏏  
**🏆 Match Winner:** Delhi Capitals 🎉  


```python

team1 = ' Delhi Capitals '
team2 ='Punjab Kings '
venue = ' Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur '
```

## MATCH 59  
## LSG VS RCB  

**LSG vs RCB: The Battle of Aggression**  

Lucknow Super Giants (LSG) and Royal Challengers Bangalore (RCB) have developed a fierce rivalry in recent IPL seasons, with both teams known for their explosive batting line-ups.  

**📅 Date:** May 9, 2025  

**📍 Venue:** Ekana Cricket Stadium, Lucknow  

**Insights from Historical Data**  

- Teams have won 49.8% of tosses at Ekana Cricket Stadium since 2010.  
- Royal Challengers Bangalore have won 5 of the last 10 matches against Lucknow Super Giants.  
- Teams batting second have a 60.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Lucknow Super Giants 🏏  
**🏆 Match Winner:** Lucknow Super Giants 🎉  

```python

team1 = ' Lucknow Super Giants
team2 ='Royal Challengers Bangalore
venue = ' Ekana Cricket Stadium, Lucknow 
```
## MATCH 60  
## SRH VS KKR  

**SRH vs KKR: The Clash of Titans**  

Sunrisers Hyderabad (SRH) and Kolkata Knight Riders (KKR) have had a competitive rivalry over the years, with both teams boasting strong squads.  

**📅 Date:** May 10, 2025  

**📍 Venue:** Rajiv Gandhi International Stadium, Hyderabad  

**Insights from Historical Data**  

- Teams have won 47.2% of tosses at Rajiv Gandhi International Stadium since 2010.  
- Kolkata Knight Riders have won 6 of the last 10 matches against Sunrisers Hyderabad.  
- Teams batting second have a 58.5% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Kolkata Knight Riders 🏏  
**🏆 Match Winner:** Kolkata Knight Riders 🎉  

```python

team1 = ' Kolkata Knight Riders'
team2 ='Royal Challengers Bangalore'
venue =' Rajiv Gandhi International Stadium, Hyderabad'
```
## MATCH 61 TO 70
![alt text](image-5.png)
## MATCH 61  
## PBKS VS MI  

**PBKS vs MI: A High-Voltage Showdown**  

Punjab Kings (PBKS) and Mumbai Indians (MI) have delivered some nail-biting encounters over the years, with MI holding a slight edge in head-to-head battles.  

**📅 Date:** May 11, 2025  

**📍 Venue:** Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur  

**Insights from Historical Data**  

- Teams have won 49.8% of tosses at Maharaja Yadavindra Singh Stadium since 2010.  
- Mumbai Indians have won 7 of the last 10 matches against Punjab Kings.  
- Teams batting second have a 60.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Mumbai Indians 🏏  
**🏆 Match Winner:** Mumbai Indians 🎉 

## code snippet

```python

team1 = ' Mumbai Indians'
team2 ='Punjab Kings'
venue = '  Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur'
```

## MATCH 62  
## DC VS GT  

**DC vs GT: A Clash of Powerhouses**  

Delhi Capitals (DC) and Gujarat Titans (GT) have developed a competitive rivalry, with both teams having their share of victories in past seasons.  

**📅 Date:** May 11, 2025  

**📍 Venue:** Arun Jaitley Stadium, Delhi  

**Insights from Historical Data**  

- Teams have won 50.2% of tosses at Arun Jaitley Stadium since 2010.  
- Gujarat Titans have won 6 of the last 10 matches against Delhi Capitals.  
- Teams batting second have a 58.6% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Gujarat Titans 🏏  
**🏆 Match Winner:** Gujarat Titans 🎉  


## code snippet

```python

team1 = ' Gujarat Titans'
team2 ='Delhi Capitals'
venue = ' Arun Jaitley Stadium, Delhi'
```
## MATCH 63  
## CSK VS RR  

**CSK vs RR: A Battle of Champions**  

Chennai Super Kings (CSK) and Rajasthan Royals (RR) have a historic rivalry, with CSK often having the upper hand, but RR pulling off key upsets in recent seasons.  

**📅 Date:** May 12, 2025  

**📍 Venue:** MA Chidambaram Stadium, Chennai  

**Insights from Historical Data**  

- Teams have won 48.3% of tosses at MA Chidambaram Stadium since 2010.  
- Chennai Super Kings have won 6 of the last 10 matches against Rajasthan Royals.  
- Teams batting first have a 64.7% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Chennai Super Kings 🏏  
**🏆 Match Winner:** Chennai Super Kings 🎉  
## code snippet

```python

team1 = ' Chennai Super Kings
team2 ='Rajasthan Royals'
venue = 'MA Chidambaram Stadium, Chennai  '
```

## MATCH 64  
## RCB VS SRH  

**RCB vs SRH: A Fierce Encounter**  

Royal Challengers Bangalore (RCB) and Sunrisers Hyderabad (SRH) have had thrilling encounters over the years, with both teams showcasing their strengths in different seasons.  

**📅 Date:** May 13, 2025  

**📍 Venue:** M. Chinnaswamy Stadium, Bengaluru  

**Insights from Historical Data**  

- Teams have won 51.2% of tosses at M. Chinnaswamy Stadium since 2010.  
- Sunrisers Hyderabad have won 6 of the last 10 matches against Royal Challengers Bangalore.  
- Teams chasing have a 58.4% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Sunrisers Hyderabad 🏏  
**🏆 Match Winner:** Sunrisers Hyderabad 🎉  
## code snippet

```python

team1 = ' Sunrisers Hyderabad'
team2 ='Royal Challengers Bangalore'
venue = 'M. Chinnaswamy Stadium, Bengaluru '
```

## MATCH 65  
## GT VS LSG  

**GT vs LSG: The Battle of the Newcomers**  

Gujarat Titans (GT) and Lucknow Super Giants (LSG) have had competitive encounters since their debut, making this matchup an exciting clash.  

**📅 Date:** May 14, 2025  

**📍 Venue:** Narendra Modi Stadium, Ahmedabad  

**Insights from Historical Data**  

- Teams have won 55.6% of tosses at Narendra Modi Stadium since 2010.  
- Gujarat Titans have won 5 of the last 7 matches against Lucknow Super Giants.  
- Teams chasing have a 62.1% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Gujarat Titans 🏏  
**🏆 Match Winner:** Gujarat Titans 🎉  
## code snippet

```python

team1 = ' Gujarat Titans'
team2 ='Lucknow Super Giants'
venue = 'Narendra Modi Stadium, Ahmedabad 
```
## MATCH 66  
## MI VS DC  

**MI vs DC: The Battle of Capitals**  

Mumbai Indians (MI) and Delhi Capitals (DC) have had a closely contested rivalry, with MI holding a historical edge in head-to-head encounters.  

**📅 Date:** May 15, 2025  

**📍 Venue:** Wankhede Stadium, Mumbai  

**Insights from Historical Data**  

- Teams have won 51.2% of tosses at Wankhede Stadium since 2010.  
- Mumbai Indians have won 6 of the last 10 matches against Delhi Capitals.  
- Teams chasing have a 59.8% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Mumbai Indians 🏏  
**🏆 Match Winner:** Mumbai Indians 🎉  
## code snippet

```python

team1 = '  Mumbai Indians'
team2 ='Delhi Capitals
venue = 'Wankhede Stadium, Mumbai '
```
## MATCH 67  
## RR VS PBKS  

**RR vs PBKS: A High-Octane Clash**  

Rajasthan Royals (RR) and Punjab Kings (PBKS) have always produced thrilling encounters, with both teams having their fair share of victories over the years.  

**📅 Date:** May 16, 2025  

**📍 Venue:** Sawai Mansingh Stadium, Jaipur  

**Insights from Historical Data**  

- Teams have won 50.6% of tosses at Sawai Mansingh Stadium since 2010.  
- Rajasthan Royals have won 7 of the last 10 matches against Punjab Kings.  
- Teams batting first have a 61.3% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Rajasthan Royals 🏏  
**🏆 Match Winner:** Rajasthan Royals 🎉  
## code snippet

```python

team1 = ' Rajasthan Royals'
team2 ='Punjab Kings'
venue = 'Sawai Mansingh Stadium, Jaipur  '

```

## MATCH 68  
## RCB VS KKR  

**RCB vs KKR: A Fierce Rivalry**  

Royal Challengers Bangalore (RCB) and Kolkata Knight Riders (KKR) have had a long-standing rivalry, with KKR holding an edge in their past encounters.  

**📅 Date:** May 17, 2025  

**📍 Venue:** M. Chinnaswamy Stadium, Bengaluru  

**Insights from Historical Data**  

- Teams have won 52.4% of tosses at M. Chinnaswamy Stadium since 2010.  
- Kolkata Knight Riders have won 6 of the last 10 matches against Royal Challengers Bangalore.  
- Teams chasing have a 58.9% win rate at this venue due to its high-scoring nature.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Kolkata Knight Riders 🏏  
**🏆 Match Winner:** Kolkata Knight Riders 🎉  
## code snippet

```python

team1 = ' Kolkata Knight Riders'
team2 ='Royal Challengers Bangalore'
venue = 'M. Chinnaswamy Stadium, Bengaluru  '

```

## MATCH 69  
## GT VS CSK  

**GT vs CSK: Clash of Champions**  

Gujarat Titans (GT) and Chennai Super Kings (CSK) have developed an exciting rivalry in recent seasons, with both teams showcasing top performances.  

**📅 Date:** May 18, 2025  

**📍 Venue:** Narendra Modi Stadium, Ahmedabad  

**Insights from Historical Data**  

- Teams have won 55.6% of tosses at Narendra Modi Stadium since 2010.  
- Gujarat Titans have won 4 of the last 7 matches against Chennai Super Kings.  
- Teams batting second have a 62.1% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Gujarat Titans 🏏  
**🏆 Match Winner:** Gujarat Titans 🎉  
## code snippet

```python

team1 = 'Gujarat Titans'
team2 ='Chennai Super Kings'
venue = 'Narendra Modi Stadium, Ahmedabad'

```
## MATCH 70  
## LSG VS SRH  

**LSG vs SRH: A Battle for Supremacy**  

Lucknow Super Giants (LSG) and Sunrisers Hyderabad (SRH) have had competitive encounters, with both teams eager to dominate.  

**📅 Date:** May 18, 2025  

**📍 Venue:** Ekana Cricket Stadium, Lucknow  

**Insights from Historical Data**  

- Teams have won 51.3% of tosses at Ekana Cricket Stadium since 2010.  
- Lucknow Super Giants have won 3 of the last 5 matches against Sunrisers Hyderabad.  
- Teams batting second have a 58.4% win rate at this venue.  

**Model Predictions (Based on Historical Data & Machine Learning Models)**  

**✔ Toss Winner:** Lucknow Super Giants 🏏  
**🏆 Match Winner:** Lucknow Super Giants 🎉   

```python

team1 = 'Lucknow Super Giants'
team2 ='Sunrisers Hyderabad
venue = ' Ekana Cricket Stadium, Lucknow'
```

