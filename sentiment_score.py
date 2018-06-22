"""Vader sentiment analysis demo."""

from typing import Iterable, Dict, Tuple, Any
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from matplotlib.axes._subplots import Axes
import pandas as pd
from nltk.sentiment import vader

Plot = Tuple[Figure, Axes]
RGBColor = Tuple[float, float, float]


class SentimentScore:
    """Vader sentiment score."""
    analyzer = vader.SentimentIntensityAnalyzer()

    def __init__(self, text):
        """Create a new sentiment score."""
        self.text = text
        self.score = SentimentScore.analyzer.polarity_scores(text)
        self.color = self._get_sentiment_color()

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary including, sentiment scores and bar color."""
        score = self.score.copy()
        score['color'] = self.color
        score['text'] = self.text
        return score

    def _get_sentiment_color(self) -> RGBColor:
        """Return (red, green, blue) indicating sentiment."""
        red = self.score['neg']
        green = self.score['pos']
        blue = self.score['neu']
        return (red, green, blue)


def create_scores_df(phrases: Iterable[str]) -> pd.DataFrame:
    """Return Pandas DataFrame tagging each phrase with sentiment and color."""
    return pd.DataFrame([SentimentScore(text).to_dict() for text in phrases])


def plot_sentiment_scores(scores: pd.DataFrame) -> Plot:
    """Plot sentiment scores as a horizontal bar chart."""
    fig, ax = plt.subplots()
    ax.barh(y=scores['text'],
            width=scores['compound'],
            color=scores['color'])
    ax.set_title('Vader Sentiment Examples')
    labels = ['Negative', 'Neutral', 'Positive']
    handles = [Patch(color='red'), Patch(color='blue'), Patch(color='green')]
    ax.legend(labels=labels, handles=handles)
    return (fig, ax)
