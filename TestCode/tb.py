from textblob import TextBlob

blob = TextBlob("This is a red ball")



for item in blob.noun_phrases:
    print item