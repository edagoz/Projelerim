#Aşağıdaki kod bloğu çalışmam için gerekli kütüphaneleri ve dosyaları, çalışmama dahil edebilmek için yazılmıştır.
import pandas as pd
import re
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
from alive_progress import alive_bar

#Aşağıdaki fonksiyon twitter üzerinde yapılan bir arama ile alınan tweetleri döndürmektedir.


def get_tweets(query, limit=1_000_000_000, also_csv=False, csv_name='tweets.csv'):
    """
    Tasks
    -----
        Gets tweets from Twitter.

    Parameters
    ----------
    query: str
        The query to be searched on Twitter.
    limit: int (default=1000000000)
        The limit of tweets to be searched.
    also_csv: bool (default=False)
        If True, saves the tweets as a csv file.
    csv_name: str (default='tweets.csv')
        The name of the csv file to be saved.
    Returns
    -------
    dataframe: pandas.DataFrame
        The dataframe containing the tweets.
    """

    import snscrape.modules.twitter as sntwitter

    tweets = []

    with alive_bar(limit, force_tty=True) as bar:

        if limit == 1000000000: print("Limit not defined. Progress bar may work untidy.")
        for i, t in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= limit:
                break
            else:
                tweets.append(
                    [t.id, t.url, t.media, t.date.strftime("%d/%m/%Y, %H:%M:%S"), t.retweetCount, t.likeCount,
                     t.quoteCount, t.hashtags, t.content, t.lang, t.user.location, t.cashtags, t.conversationId,
                     t.coordinates, t.inReplyToTweetId, t.inReplyToUser, t.mentionedUsers, t.outlinks, t.place,
                     t.quotedTweet, t.renderedContent, t.replyCount, t.retweetCount, t.retweetedTweet, t.source,
                     t.sourceLabel, t.sourceUrl, t.tcooutlinks, t.user, t.user.username,
                     t.user.created.strftime("%d-%m-%Y %H:%M:%S"), t.user.description, t.user.descriptionUrls,
                     t.user.displayname, t.user.favouritesCount, t.user.followersCount, t.user.friendsCount, t.user.id,
                     t.user.label, t.user.linkTcourl, t.user.linkUrl, t.user.listedCount, t.user.location,
                     t.user.mediaCount, t.user.profileBannerUrl, t.user.profileImageUrl, t.user.protected,
                     t.user.rawDescription, t.user.statusesCount, t.user.url, t.user.username, t.user.verified])
            bar()

    dataframe = pd.DataFrame(tweets, columns=['id', 'url', 'media', 'date', 'retweet_count', 'like_count', 'quoteCount',
                                              'hashtags', 'content', 'lang', 'user_location',
                                              'cashtags', 'conversation_id', 'coordinates', 'inReplyToTweetId',
                                              'inReplyToUser', 'mentionedUsers', 'out_links', 'place',
                                              'quotedTweet', 'renderedContent', 'replyCount', 'retweetCount',
                                              'retweetedTweet', 'source', 'sourceLabel', 'sourceUrl', 'tco_out_links',
                                              'user', 'user_name', 'user_created', 'user_description',
                                              'user_descriptionUrls', 'user_display_name', 'user_favouritesCount',
                                              'user_followersCount', 'user_friendsCount', 'user_id', 'user_label',
                                              'user_link_Tco_url', 'user_linkUrl', 'user_listedCount', 'user_location',
                                              'user_media_count', 'user_profile_banner_url', 'user_profile_image_url',
                                              'user_protected', 'user_raw_description', 'user_statuses_count',
                                              'user_url', 'user_username', 'user_verified'])

    if also_csv:
        dataframe.to_csv(csv_name, index=False)
        print("CSV file created")

    print(f"Dataframe has {dataframe.shape[0] - 1} tweets")
    return dataframe
