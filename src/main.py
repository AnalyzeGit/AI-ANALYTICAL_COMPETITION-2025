import pandas as pd
from typing import List, Dict, Union
from nlp_tools.nlp_config  import RAW_DATA_DIR
from nlp_tools.text_scorer import SentimentScore, KeywordBasedScorer

if __name__ == '__main__':
    # 0) 입력 데이터 준비
    # df: 반드시 'text' 컬럼을 포함해야 함
    # Type Alias
    SentimentResult = Dict[str, Union[str, float]]

    df = pd.read_csv(RAW_DATA_DIR / "돌봄인형데이터.csv", encoding='cp949')
    assert "text" in df.columns, "'text' 컬럼이 필요합니다."
    texts: List[str] = df['text'].tolist()

    # 1) 감성 분석
    sentiment  = SentimentScore()    
    sentiment_results: List[SentimentResult] = sentiment.get_sentiment_scores(texts)
    sentiment_results_df: pd.DataFrame  = pd.DataFrame(sentiment_results)

    # 2) 키워드 기반 점수화
    keyword_scorer = KeywordBasedScorer()
    kewyword_scores: Dict[str, float] = df["text"].apply(
        lambda t: pd.Series(keyword_scorer.keyword_count_score(str(t)))
    )
    
    # 3) 데이터 병합
    # - 원본 df에서 text 컬럼은 제거 (sentiment_results_df에 'text' 포함)
    # - 인덱스 정렬로 안전하게 병합
    analysis_dataset: pd.DataFrame = pd.concat(
            [
                df.drop(columns='text').reset_index(drop=True), 
                sentiment_results_df.reset_index(drop=True), 
                kewyword_scores.reset_index(drop=True)
            ],
            axis=1
    )

    
    # (선택) 컬럼 순서 정리: id류 -> 감성 -> 키워드 점수
    cols_front: List[str] = [c for c in analysis_dataset.columns if c not in {"sentiment_pos", "sentiment_neg"} and not c.startswith("keyword_score_")]
    cols_ordered = (
        cols_front
        + ["sentiment_pos", "sentiment_neg"]
        + sorted([c for c in analysis_dataset.columns if c.startswith("keyword_score_")])
    )
    analysis_dataset = analysis_dataset[cols_ordered]


# if __name__ == "__main__":
#     data = pd.read_excel(RAW_DATA_DIR / "발화데이터(대전중구)_pub.xlsx")
#     utterance = UtterancePreprocessor(data)
#     utterance_results = utterance.preprocess_utterances()
