###### BASIC TEXT PROCESSING #######

#read raw data into data frame
bbc_data <- read.csv("bbc_news_stories.csv", stringsAsFactors = FALSE)

#reference tm package
library(tm)

#load the data frame into a corpus object type, so the tm package can process it
bbc_corpus <- VCorpus(DataframeSource(bbc_data))

#do various cleaning steps on the corpus, each overwrites the existing/previous corpus, so cleaning is sequential
bbc_corpus <- tm_map(bbc_corpus, content_transformer(tolower)) #converts all text to lower-case
bbc_corpus <- tm_map(bbc_corpus, removeWords, stopwords("SMART")) #removes stopwords such as 'the', 'and', etc
bbc_corpus <- tm_map(bbc_corpus, removePunctuation)
bbc_corpus <- tm_map(bbc_corpus, removeNumbers)
#bbc_corpus <- tm_map(bbc_corpus, stemDocument) #applies stemming to convert things like plural versions of words to singular/stemmed form - can be over-zelous

#create document-term and term-document matrices - one is transpse of the other
bbc_dtm <- DocumentTermMatrix(bbc_corpus)
bbc_tdm <- TermDocumentMatrix(bbc_corpus)

#view preview of matrices
inspect(bbc_dtm)
inspect(bbc_tdm)

#report 5 most frequent terms for each doc in matrix (will work with either document-term, or term-document)
findMostFreqTerms(bbc_dtm, 5)

#find terms that are correlated to at least 0.5 correlation coefficient with word 'local' - this is correlated in terms of the frequency of their use across different docs
findAssocs(bbc_dtm, "blair", 0.5)

#use tf-idf (term frequency, inverse document frequency) weighting when creating a document-term matrix - this will help find words that are not just frequently used in a doc, but also those that help differentiate a doc from others in a collection
bbc_dtm_weighted <- DocumentTermMatrix(bbc_corpus, control=list(weighting=weightTfIdf))

#inspect results of the weighted matrix and report most frequent terms for each doc - frequent as this point means the output of the tf-idf weighting
inspect(bbc_dtm_weighted)
findMostFreqTerms(bbc_dtm_weighted, 5)


###### WORD CLOUDS #######

#reference some relevant packages
library(tidytext)
library(dplyr)
library(wordcloud)
library(wordcloud2)

#create long form version of the document term matrix output - uses the tidytext package
dtm_long <- tidy(bbc_dtm)

#create summarised table showing each unique term and number of times used across all docs - uses dplyr package
words_by_freq <- summarise(group_by(dtm_long,term),n=sum(count))

#present top 50 words in basic wordcloud - uses wordcloud package
wordcloud(words_by_freq$term, words_by_freq$n, max.words = 50, random.order = FALSE)

#sort the summarised table in descending order and use wordcloud2 package to create alternative wordcloud output
words_by_freq <- arrange(words_by_freq, -n)
wordcloud2(head(words_by_freq,50), shape="diamond")


###### N-GRAMS ######

#reference the RWeka package
library(RWeka)

#creates function to create n-grams (min/max = specifies length of n-grams required) - example below is for n-grams of length 2, eg. 2-word phrases
ngram_function <- function(x) NGramTokenizer(x, Weka_control (min=2, max=2))

#creates document-term matrix using n-gram function referenced above, so phrases in the output rather than individual words. NB - matrix will be bigger than single word based matrix
bbc_dtm_ngram <- DocumentTermMatrix(bbc_corpus, control = list(tokenize = ngram_function))

#inspect results of ngram based matrix and report most frequent phrases for each doc
#Note - might not want to run this without stopwords or with stemming applied as n-grams will not be true phrases
#Note also that similar tf-idf weighting can be applied if interested in phrases that highlight differences between documents
inspect(bbc_dtm_ngram)
findMostFreqTerms(bbc_dtm_ngram, 5)


##### SENTIMENT ANALYSIS ######

#read in the list of words with sentiment scores
words_with_sentiment <- get_sentiments("bing")
words_with_sentiment$score <- ifelse(words_with_sentiment$sentiment == "positive", 1, -1)

#join the sentiment score list to the long version (as created through the tidy fuction) of the document term matrix 
dtm_long_with_sentiment <- inner_join(dtm_long, words_with_sentiment, by=c("term"="word"))

#aggregate the result to provide an overall net sentiment score for each document (numnber of time positive words mentioned minus number of times negtaive words mentioned)
docs_with_sentiment <- summarise(group_by(dtm_long_with_sentiment, document), net_score = sum(count * score))

#plot of sentiment distribution
hist(docs_with_sentiment$net_score, breaks=30)

#create filtered version of the long version of the document term matrix for just one document
selected_doc <- filter(dtm_long_with_sentiment, document == "Terror powers expose 'tyranny'")

#pivot the filtered list to provide 3 column output
library(reshape2)
selected_doc_pivot <- acast(selected_doc, term ~ sentiment, value.var="count", fill=0)

#create a comparison word cloud to show positive words used in the doc vs negative words
comparison.cloud(selected_doc_pivot, colors = c("red","darkgreen"))


#### CLUSTERING ####

#create a distance matrix from the document term matrix
bbc_dtm_reduced <- bbc_dtm[0:100,]
inspect(bbc_dtm_reduced)
dist_bbc_dtm <- dist(bbc_dtm_reduced) #euclidean distances
dist_bbc_dtm_dfversion <- as.data.frame(as.matrix(dist_bbc_dtm)) #version of distance matrix that can be easily previewed in R

#basic hierarchical clustering
bbc_clusters <- hclust(dist_bbc_dtm)
plot(bbc_clusters)
plot(bbc_clusters, hang=-0.1)
plot(bbc_clusters, hang=-0.1, labels=substr(bbc_clusters$labels,1,20), cex=0.6)

rect.hclust(bbc_clusters, k = 20)
bbc_clusters_groups <- as.data.frame(cutree(bbc_clusters, k=20))

#word embeddings based clustering
library(data.table)
library(tidyr)
glov6b <- fread("glove.6B.50d.txt", sep = " ")
names(glov6b)[1]<-"token"

#wrangle the doc term matrix and join to glove embeddings to calculate averaged embeddings for each bbc post
bbc_dtm_reduced_long <- tidy(bbc_dtm_reduced)
bbc_dtm_reduced_long_with_glove <- inner_join(bbc_dtm_reduced_long, glov6b, by=c("term"="token"))
bbc_dtm_reduced_long_with_glove_long <- pivot_longer(bbc_dtm_reduced_long_with_glove, starts_with("V"), names_to = "dim", values_to = "score") #starts_with d

bbc_dtm_reduced_long_with_glove_agg <- summarise(group_by(bbc_dtm_reduced_long_with_glove_long, document, dim), 
                                                 score = sum(score * count)/sum(count))

#pivot the result and calculate distance matrix
bbc_dtm_reduced_long_with_glove_agg_pvt <- pivot_wider(bbc_dtm_reduced_long_with_glove_agg, names_from=dim, values_from = score)

docnames <- bbc_dtm_reduced_long_with_glove_agg_pvt$document

bbc_dtm_reduced_long_with_glove_agg_pvt <- bbc_dtm_reduced_long_with_glove_agg_pvt[,-1]

dist_bbc_dtm_glove <- dist(bbc_dtm_reduced_long_with_glove_agg_pvt)
dist_bbc_dtm_glove_dfversion <- as.data.frame(as.matrix(dist_bbc_dtm_glove))
colnames(dist_bbc_dtm_glove_dfversion) <- docnames
row.names(dist_bbc_dtm_glove_dfversion) <- docnames

#do clustering
bbc_clusters_glove <- hclust(dist_bbc_dtm_glove)
bbc_clusters_glove$labels <- docnames
plot(bbc_clusters_glove, hang=-0.1, labels=substr(bbc_clusters_glove$labels,1,20), cex=0.6)
rect.hclust(bbc_clusters_glove, k = 20)








