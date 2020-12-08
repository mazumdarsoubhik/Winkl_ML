'''
Influencer Categorization
Winkl.co API
Made by:
Soubhik Mazumdar
mazumdarsoubhik@gmail.com
'''

# Import Packages #
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re
from itertools import chain 
import math
import time
import logging 
import requests
import json
print("Imported all packages.")

tic = time.time()
print("Loading GoogleNews...")
from gensim import models
w = models.KeyedVectors.load_word2vec_format(r"../GoogleNews-vectors-negative300.bin.gz", binary=True, limit=10000)
print("Loaded GoogleNews!")



def process(array,avoidwords):
    text = re.sub(r'\[[0-9]*\]',' ',str(array))  #Remove Numbers
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text) # Remove nums
    text = re.sub(r'\s+',' ',text)  #Remove extra space
    text = re.sub(r"[^a-zA-Z0-9]+",' ',text)  #Remove special characters
    text = text.lower()  #Lower case all
    text = nltk.sent_tokenize(text)  #Tokenize to sentences 
    keywords = [nltk.word_tokenize(sentence) for sentence in text]
    stop_words = stopwords.words('english')
    stop_words.extend(avoidwords)
    for i in range(len(keywords)):
        keywords[i] = [word for word in keywords[i] if word not in stop_words]
    return keywords

#Frame pre-processing function
def prepro(frame,cap_or_bio,avoidwords):
    frame2 = frame
    for c in range(len(frame2)):
        keywords = process(frame2[cap_or_bio][c],avoidwords)
        frame2[cap_or_bio][c] = keywords
    return frame2

def flat(array):     # 2D array reduction
    flatten_list = [j for sub in array for j in sub] 
    flatten_list = list(chain.from_iterable(flatten_list))
    return flatten_list

def soft_flat(array):     # 2D array reduction
    flatten_list = [j for sub in array for j in sub] 
    return flatten_list


# normalize() -> given an array, converts to 1/0, top int(pos) will be 1
def normalize(keys, pos =3):  
    ax = [i for i in keys]
    temp = [i for i in keys]
    temp.sort()
    temp = temp[-pos:]
    for x in temp:
        ax[keys.index(x)] = 1
    for x in range(len(ax)):
        if ax[x] != 1:
            ax[x] = 0
    return ax

def normalizeSD(keys, thre =3):    # Given score array return shortlisted cats in given threshold
    ax = deviation(keys)
    ax = dev_shortlist(ax)
    return ax

# compute() => category[] to be called outside
def compute(caption,category,top =3):
    ar = []
    score = []

    # Code to get frequency distribution and unique keywords array
    keywords = []
    caption_freq = []
    counts = Counter(caption)
    if len(counts) > 0:
        labels, values = zip(*counts.items())
        ## sort your values in descending order
        indSort = np.argsort(values)[::-1]
        ## rearrange your data
        keywords = np.array(labels)[indSort]  # Label
        caption_freq = np.array(values)[indSort]  # Values
    
    # Detect words not in Google Dict | Put freq = 0
    for x in keywords:
        try:
            restConst = w.similarity(x,'something')
        except KeyError:
            caption_freq[np.where(keywords == x)] = 0
        
    #Google similaity function
    for x in category:
        empty = []
        for y in keywords:
            try:
                empty.append(w.similarity(x,y))
            except:
                empty.append(0)
        ar.append(empty)
    
    # Store the similarity values in dataframe
    frame = pd.DataFrame()
    frame = pd.DataFrame(ar, columns = keywords)
    
    
    ### CHANGES  MADE
    #Normalize | top select
    for key in frame.columns:
        frame[key] = normalizeSD(frame[key].tolist(),top)
    
    # Multiply with frequency
    for row in range(len(frame)):
        frame.values[row] = [i*j for i,j in zip(frame.values[row],caption_freq)]
    # Sum the values => Score
    for row in range(len(frame)):
        score.append(sum(frame.values[row]))
    
    frame['category'] = category
    frame['Scores'] = score
    return frame,keywords[:20]

def deviation(array):
    mu = max(array)
    l = len(array)
    ar = []
    for x in range(l):
        ar.append(math.sqrt((array[x]-mu)**2)/l)
    total = sum(ar)
    for x in range(l):
        if total != 0:
            ar[x] = (ar[x]/total)*100
    return ar

def mean_deviation(array):
    l = len(array)
    mu = sum(array)/l
    ar = []
    for x in range(l):
        ar.append(math.sqrt((array[x]-mu)**2)/l)
    total = sum(ar)
    for x in range(l):
        if total != 0:
            ar[x] = (ar[x]/total)*100
    return ar

def dev_shortlist(dev_array,thre = 2):  # Shortlist using threshold from deviation array | return array in 1/0
    final_cat = [0]*len(dev_array)
    for i in range(len(dev_array)):
        if dev_array[i] <=thre:
            final_cat[i] = 1
    return final_cat

def sort_cat(dev,cat, thre = 2):   # Shortlist using thre | Return category array
    final_cat = []
    for i in range(len(dev)):
        if dev[i] <=thre:
            final_cat.append(cat[i])
    return final_cat

def get_row_pscore(col_name,f1,i,f2, scoreType):  # f1-mainframe | f2-frame
    ud = f1.loc[i,'id']
    ul = f1.loc[i,'url']
    row_in_array = [ud,ul]
    dev_array = f2[scoreType].tolist()
    row_in_array.extend(dev_array)
    tk = f2.loc[0,'Top keywords']
    row_in_array.extend([tk])
    zip_it = zip(col_name,row_in_array)
    convert_to_dict = dict(zip_it)
    return convert_to_dict

def top_category(f2,categories,thre=2):
    final_cat = [0]*len(categories)
    dev_scores = f2['Deviation'].tolist()
    rank_of_cat = 1
    for i in range(len(dev_scores)):
        if dev_scores[i] <=thre:
            final_cat[i] = rank_of_cat 
            rank_of_cat = rank_of_cat +1
    return final_cat

def top_category_get_percent(f2,categories,thre=2):
    final_cat = [0]*len(categories)
    dev_scores = f2['Deviation'].tolist()
    percent_array = f2['Percentage'].tolist()
    for i in range(len(dev_scores)):
        if dev_scores[i] <=thre:
            final_cat[i] = percent_array[i]
    return final_cat
    
    
def get_row_result(col_name,f1,f2,rank_array):  # f1-mainframe
    ud = f1.at[0,'user_id']
    hd = f1.at[0,'handle']
    fl = f1.at[0,'followers']
    ul = f1.at[0,'url']
    row_in_array = [ud,hd,fl,ul]
    row_in_array.extend(rank_array)
    tk = f2.at[0,'Top keywords']
    row_in_array.extend([tk])
    zip_it = zip(col_name,row_in_array)
    convert_to_dict = dict(zip_it)
    return convert_to_dict

# To make data in 
# DB format | API post 
def to_dict_api(percentages,categories,top_keywords,frame,i): #frame and i to get id
    mydict = {}
    cat_array =[]
    empty_percent = [0]*(len(categories)-len(percentages))
    percent_array = [y for y in percentages]
    percent_array.extend(empty_percent)
    mydict['user_id'] = frame.loc[i,'id']
    mydict['keywords'] = json.dumps(top_keywords.tolist())
    for j in range(len(categories)):
        cat_array.append({'tag':categories[j],'percentage':percent_array[j]})
    mydict['categories'] = json.dumps(cat_array)
    return mydict


# Categories
categories = ['food', 'fashion', 'makeup', 'beauty', 'lifestyle','luxury', 'traveler','photography','fitness','sports','gaming', 'entertainment', 'technology','investment','education', 'animal', 'health']
API_categories = ['Food','Fashion', 'Makeup', 'Beauty', 'Lifestyle','Luxury', 'Travel','Photography','Fitness','Sports','Gaming', 'Entertainment', 'Gadgets & Tech','Finance','Education', 'Animal/Pet', 'Health','Art', 'Self Improvement', 'Parenting', 'Books']

# This required to run some function
col_name = ['user_id','url','food', 'fashion', 'makeup', 'beauty', 'lifestyle','luxury', 'travel', 'photography','fitness','sports','gaming', 'entertainment', 'technology','investment','education', 'animal', 'health', 'parenting','top keywords']


# void main() #

x = requests.get('http://44.229.68.155/insta_users/get_uncategorized_accounts?limit=10&current_id=0', headers={'Authorization': 'Token ruor7REQi9KJz6wIQKDXvwtt'})
status = x.status_code
data = x.json()
df = pd.DataFrame(data['users'])
pages = 0
idsdone = 0
txt = "Done {} pages, the last_id is {} and time taken {} seconds"

while(len(data['users']) !=0 and pages<4):
    try:
        new_tic = time.time()
        if(status != 200):
            raise Exception("GET request error: {}".format(status))
        dfnew = pd.DataFrame(columns=['id','handle','name','url','gender','country','captions'], data = df[['id','handle','name','url','gender','country','captions']].values)
        last_id = dfnew['id'].iloc[-1]
        # Fresh dataframe
        profile_percentages =  pd.DataFrame(columns = ['user_id','url','food', 'fashion', 'makeup', 'beauty', 'lifestyle','luxury', 'travel', 'photography','fitness','sports','gaming', 'entertainment', 'technology','investment','education', 'animal','health', 'parenting','top keywords'])


        # Main Categorization # 
        for i in range(len(dfnew)):

            try:
                #Store userid | caption | total posts
                userid = dfnew['id'].iloc[i]
                captions = dfnew['captions'].iloc[i]
                total_posts = len(captions)

                # Words which mostly occurs in insta post and we want to avoid considering them for the sake of accuracy of results
                avoidwords = ['verified','none']

                #Converting to keywords
                captions = process(captions,avoidwords)
                caption_array = captions[0]
                
                #Temporary array i-> interim
                icaption_array = [z for z in caption_array]
                # Removing words not in dictionary also single characters
                for x in caption_array:
                    try:
                        checkword = w.similarity(x,'something') #Check word if exist in googlenews
                        if len(x) <2: #Removing single character
                            icaption_array.pop(icaption_array.index(x))
                    except KeyError:
                        icaption_array.pop(icaption_array.index(x))
                #Restore Array
                caption_array = [z for z in icaption_array]
                
                if len(caption_array) ==0:
                    raise Exception("No Words in profile for categorization or Different language")

                # Punishing accounts which has less than 1.5 words in caption
                if len(caption_array) < 2*(total_posts):
                    raise Exception("Too less words for categorization")
                
                # Word2vec computation
                frame = pd.DataFrame()
                frame, top_keywords = compute(caption_array,categories,3)
                
                #Convert to Percentage
                per = frame['Scores'].tolist()
                per_sum = sum(per)
                for x in range(len(per)):
                    temp_number = (float)(per[x])
                    per[x] = round((temp_number/per_sum)*100)
                frame['Percentage'] = per
                
                #Store profile percentage
                row_df_5 = get_row_pscore(col_name,dfnew,i,frame,'Percentage')
                profile_percentages = profile_percentages.append(row_df_5,ignore_index=True)
                # POST API Request
                file = to_dict_api(frame['Percentage'].tolist(),API_categories,top_keywords,dfnew,i)
    #             url = 'http://44.229.68.155/insta_user/add_category_to_insta_user'
    #             y = requests.post(url, data = file,headers={'Authorization': 'Token ruor7REQi9KJz6wIQKDXvwtt'})

    #             if y.status_code !=200:
    #                 raise Exception("Post request error {}".format(y.status_code))

                print(type(file))
                print(file)
                
                idsdone = idsdone +1

            except Exception as Argument:
                # creating/opening a file 
                f = open(r"e1.txt", "a") 
                # writing in the file 
                f.write("Userid\t"+str(userid)+"\t: "+str(Argument)+str("\n")) 
                # closing the file 
                f.close()  

        # END of Main Categorization #
        profile_percentages.to_csv(r'test1.csv')
        pages = pages +1
        toc = time.time()
        print(txt.format(pages,last_id,toc-new_tic))
        # Request new page
        x = requests.get('http://44.229.68.155/insta_users/get_uncategorized_accounts?limit=10&current_id='+str(last_id), headers={'Authorization': 'Token ruor7REQi9KJz6wIQKDXvwtt'})
        data = x.json()
        df = pd.DataFrame(data['users'])
        status = x.status_code
    
    except Exception as Argument:
        # creating/opening a file 
        f = open(r"e1.txt", "a") 
        # writing in the file 
        f.write("Currently in "+str(pages)+"\t"+str(Argument)+str("\n")) 
        # closing the file 
        f.close()  
    
    

toc = time.time()
f = open(r"e1.txt", "a") 
# writing in the file 
f.write("The model ran in "+str(toc - tic)+" seconds"+str("\n")) 
f.write("Total ids done: "+str(idsdone)+"\n")
f.write("Last user_id {}".format(last_id)+"\n")
# closing the file 
f.close() 

