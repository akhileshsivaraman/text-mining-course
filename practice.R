library(tm)

# read data
bbc_data <- read.csv("bbc_news_stories.csv")


# create corpus
bbc_corpus <- VCorpus(DataframeSource(bbc_data))
# data contains doc_id and text columns so no need to make transformations


# clean data
clean_bbc_corpus <- bbc_corpus |>
  tm_map(content_transformer(tolower)) |>
  tm_map(removeWords, stopwords("SMART")) |>
  tm_map(removePunctuation)


# create DocumentTermMatrix
bbc_dtm <- DocumentTermMatrix(clean_bbc_corpus)


# create TermDocumentMatrix
bbc_tdm <- TermDocumentMatrix(clean_bbc_corpus)


# tidy data for visualisation
library(tidytext)
library(dplyr)
library(wordcloud2)

dtm_long <- tidy(bbc_dtm)
tdm_long <- tidy(bbc_tdm) # these are basically the same, just the order of columns is different

words_by_freq <- dtm_long |>
  group_by(term) |>
  summarise(n = sum(count)) |>
  arrange(desc(n))

wordcloud2(head(words_by_freq, 100))


#---- sentiment analysis ----
# get a dictionary with sentiments
words_with_sentiment <- get_sentiments("bing")
words_with_sentiment$score <- ifelse(words_with_sentiment$sentiment == "positive", 1, -1)

# join the sentiments to dtm_long
dtm_with_sentiments <- dtm_long |>
  inner_join(words_with_sentiment, by = c("term" = "word"))

# aggregate sentiment
docs_with_sentiments <- summarise(group_by(dtm_with_sentiments, document), net_score = count * score)

# histogram
hist(docs_with_sentiments$net_score, breaks = 4)

# word cloud
library(reshape2)
selected_doc <- dtm_with_sentiments |>
  filter(document == "Hodgson relishes European clashes")

selected_doc_pivot <- acast(selected_doc, term ~ sentiment, value.var = "count", fill = 0)
wordcloud::comparison.cloud(selected_doc_pivot, color = c("red", "blue"))


#---- clustering ----


#---- embeddings ----