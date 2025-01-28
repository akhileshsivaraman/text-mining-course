#read raw data into data frame
bbc_data <- read.csv("bbc_news_stories.csv", stringsAsFactors = FALSE)

#reference tm package
library(tm)

#load the data frame into a corpus object type, so the tm package can process it
