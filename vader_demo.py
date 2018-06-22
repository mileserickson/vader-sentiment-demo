"""Vader sentiment analysis demo."""

from typing import Iterable, Dict, Tuple, Any
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from matplotlib.axes._subplots import Axes
import pandas as pd
from nltk.sentiment import vader

SentimentScore = Dict[str, Any]
Plot = Tuple[Figure, Axes]


def get_sentiment_score(text: str) -> SentimentScore:
    """Return a dictionary of sentiment scores for text."""
    sia = vader.SentimentIntensityAnalyzer()
    score: SentimentScore = sia.polarity_scores(text)
    score['text'] = text
    score['color'] = get_sentiment_color(score)
    return score


def get_sentiment_color(score: SentimentScore) -> Tuple[float, float, float]:
    """Return (red, green, blue) indicating sentiment."""
    red = score['neg']
    green = score['pos']
    blue = score['neu']
    return (red, green, blue)


def create_scores_df(phrases: Iterable[str]) -> pd.DataFrame:
    """Return Pandas DataFrame tagging each phrase with sentiment and color."""
    return pd.DataFrame([get_sentiment_score(text) for text in phrases])


def plot_sentiment_scores(scores: pd.DataFrame) -> Plot:
    """Plot sentiment scores as a horizontal bar chart."""
    fig, ax = plt.subplots()
    ax.barh(y=scores['text'],
            width=scores['compound'],
            color=scores['color'])
    ax.set_title('Vader Sentiment Examples')
    legend_data = [['Positive', Patch(color='green')],
                   ['Neutral', Patch(color='blue')],
                   ['Negative', Patch(color='red')]]
    labels, handles = zip(*legend_data[::-1])
    ax.legend(labels=labels, handles=handles)
    return (fig, ax)
