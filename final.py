import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from textblob import TextBlob
from tweepy import Stream
ats= "66RBqAbtU3oFTQv1lJpKply0gbsWH8oe5Ghal2NgA344C"
at= "876055793505779712-pzjjcZFsD2CDMat2tZUg5dZrEin1ILl"
ck= "91HpG0wCCwNa8BXMHaKHTk2aB"
cs= "K6S9qYtLnj5UgSfmwCMihDgLLe5xQ6q9gpqBY90I4ZR1KSwTfn"
auth=OAuthHandler(ck,cs)
auth.set_access_token(at,ats)
api=tweepy.API(auth)
t=input("enter search\n")
'''r=input("enter no. of tweets \n")'''
tweet1=tweepy.Cursor(api.search,q=t,lang ='en').items(100)
x=0 #neutral
y=0 #positive
z=0 #negative
for tweet in tweet1:
	print(tweet.text)
	analysis=TextBlob(tweet.text)
	polarity = analysis.sentiment.polarity 
	if(analysis.sentiment.polarity == 0):
		x=x+1
	elif(analysis.sentiment.polarity > 0.00):
		y=y+1
	elif(analysis.sentiment.polarity < 0.00):
		z=z+1
if (x>y and x>z):
	print("overall tweet is neutral",(x/(x+y+z)*100))
elif(y>x and y>z):
	print("overall tweet is positive",(y/(x+y+z)*100))
else:
	print("overall tweet is negative",(z/(x+y+z)*100))
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
objects = ('Positive','negative','neutral')
a1=y/(x+y+z)*100
a2=z/(x+y+z)*100
a3=x/(x+y+z)*100
y_pos = np.arange(len(objects))
performance = [a1,a2,a3]
 
plt.bar(y_pos, performance, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Programming language usage')
 
plt.show()
