per = [125,125,2,8,5645,654,545,15,54,54]
per_sum = sum(per)
for x in range(len(per)):
    temp = (float)per[x]
    per[x] = round((per[x]/per_sum)*100)
print(per)