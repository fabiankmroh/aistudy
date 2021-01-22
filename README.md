**20200604**:
As mentioned in the blog, I have learned how to train natural language into a neural net. 
Letters may display a tendency to ignore the word's meaning. For example, the word LISTEN and SILENT both have the same letters, but each of them convey an entire different meaning.
Therefore, words are assigned a numerical value in a sentence. Tokenizer is used to *tokenize words in the sentence*. Also, ```oov_token = "<OOV>"``` shows that any word found in the sentence and not registered to the ```word_index``` will be automatically categorized as 1.<br>
**20200605**:
Today, I have learned about the tokenizing logic in the sarcasm detector. Despite sensing that the fundamentals are similar, I did not have a great achievement of learning new stuff.<br>
**20200607**:
Today, I have learned how to train words, which I did not understand a bit. To my knowledge, I learned that the sentence comes from the IMDB database. After loading the IMDB reviews, half of the 50,000 reviews, which I believe is converted into numerical values since NumPy was used, is appended to the training sets while the other half is appended to the testing set. After tokenizing, these values are padded to unify the sizes and then did the ```model.fit```. I cannot understand the rest of the part after the ```reverse_word_index``` and the part after ```model.fit```.
