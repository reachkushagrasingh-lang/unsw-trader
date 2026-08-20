"""Innovation evidence: does the finance lexicon improve VADER on finance headlines?

Scores every headline twice - plain VADER vs VADER + our finance lexicon - and
quantifies the change in the false-neutral rate and signal strength. Run:

    python scripts/innovation_lexicon.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd                                    # noqa: E402
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: E402

from src import etl, features, sentiment               # noqa: E402

TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
NEUTRAL_BAND = 0.05          # |compound| < 0.05 counts as "neutral"


def build_analyzers():
    plain = SentimentIntensityAnalyzer()
    boosted = SentimentIntensityAnalyzer()
    boosted.lexicon.update(sentiment.FINANCE_LEXICON)
    return plain, boosted


def main():
    eq = etl.load_clean_equities()
    trading_days = features.daily_returns(eq).index
    news = etl.load_clean_news()
    panel = features.assemble_headline_panel(news, trading_days)
    titles = panel["title"]

    plain, boosted = build_analyzers()
    c_plain = titles.map(lambda t: plain.polarity_scores(t)["compound"])
    c_boost = titles.map(lambda t: boosted.polarity_scores(t)["compound"])

    def summary(c):
        return {
            "n_headlines": len(c),
            "neutral_rate": float((c.abs() < NEUTRAL_BAND).mean()),
            "mean_abs_compound": float(c.abs().mean()),
            "pct_positive": float((c > NEUTRAL_BAND).mean()),
            "pct_negative": float((c < -NEUTRAL_BAND).mean()),
        }

    table = pd.DataFrame({"plain_vader": summary(c_plain),
                          "finance_lexicon": summary(c_boost)}).T
    table.to_csv(TAB / "lexicon_ablation.csv")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    # concrete evidence: headlines the lexicon rescued from a false neutral
    flipped = panel.loc[(c_plain.abs() < NEUTRAL_BAND) & (c_boost.abs() >= NEUTRAL_BAND),
                        ["ticker", "title"]].copy()
    flipped["plain"] = c_plain[flipped.index].round(3)
    flipped["boosted"] = c_boost[flipped.index].round(3)
    print(f"\n{len(flipped)} headlines moved from neutral -> signal. Examples:")
    print(flipped.head(10).to_string(index=False))
    flipped.head(50).to_csv(TAB / "lexicon_rescued_examples.csv", index=False)


if __name__ == "__main__":
    main()