def to_dict_api(percentages,categories,top_keywords,frame,i): #frame and i to get id
    mydict = {}
    cat_array =[]
    empty_percent = [0]*4
    percent_array = [y for y in percentages]
    percent_array.extend(empty_percent)
    mydict['user_id'] = frame['id'].iloc[i]
    mydict['keywords'] = json.dumps(top_keywords.tolist())
    for i in range(len(categories)):
        cat_array.append({'tag':categories[i],'percentage':percent_array[i]})
    mydict['categories'] = json.dumps(cat_array)
    return mydict


frame = pd.DataFrame(columns = ['id','url'], data = [[123,'f.com'],[345,'q.com']])
percent = [5,8,7,9]
cat = ['alpha','beta','coma','delta']
tops = ['hello', 'ali', 'bhatt', 'wikis', 'enabled', 'wiki', 'access', 'control', 'rules', 'may', 'imposed', 'organize', 'content']
file = to_dict_api(percent,cat,tops,frame,1)
print(file)