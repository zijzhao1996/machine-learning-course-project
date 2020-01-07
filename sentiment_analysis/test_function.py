stopwords = []
file = open("stopwords.txt", "r") 
for line in file:
	line = line.strip('\n')
	stopwords.append(line)
file.close()

stopwords = ' '.join(stopwords)
print(stopwords)