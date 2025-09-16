import re
from typing import List, Dict, Union

import pandas as pd

from transformers import (
     pipeline, 
     TextClassificationPipeline
    )

# Type Alias
SentimentResult = Dict[str, Union[str, float]]

class SentimentScore:
    """ 
    지정한 사전 학습 모델(HuggingFace Transformers)을 이용해 입력된 텍스트의 긍정/부정 점수를 계산하여 반환합니다.
    """
    def __init__(self, model_name:str = 'sangrimlee/bert-base-multilingual-cased-nsmc') -> None:
        self.pipe: TextClassificationPipeline = pipeline(
        'sentiment-analysis',
        model = model_name
            )   

    def get_sentiment_scores(self, texts: List[str]) -> List[SentimentResult]:
        """
        텍스트 리스트의 감성을 분석하여 긍정/부정 점수를 반환합니다.

        Args:
            texts (List[str]): 감성 분석할 텍스트 문자열 리스트.

        Returns:
            List[Dict[str, float]]: 각 텍스트별 분석 결과 리스트.
                각 결과는 다음 키를 포함합니다:
                - 'text' (str): 입력 텍스트
                - 'sentiment_pos' (float): 긍정 점수 (0~1 사이 확률)
                - 'sentiment_neg' (float): 부정 점수 (0~1 사이 확률)
        """
        results: List[SentimentResult]  = []
        outputs: List[SentimentResult] = self.pipe(texts) # 배치 처리
    
        for t, output in zip(texts, outputs):
            if output['label'].lower() == 'positive':
                pos, neg = output['score'], 1 - output['score']
            else:
                neg, pos = output['score'], 1 - output['score']
            results.append({'text':t, 'sentiment_pos': pos, "sentiment_neg": neg})
        return results


class KeywordBasedScorer:
    """  키워드 기반 점수화 클래스.
    사전에 정의된 감성/위험 키워드 사전(lexicon)을 기반으로
    텍스트 내 특정 키워드의 등장 횟수와 가중치를 계산하여
    카테고리별 점수를 반환합니다.
    """
    def __init__(self) -> None:
        self.lexicon: Dict[str, Dict[str, float]] = {
                        "positive": {
                            "고마워": 1.0,
                            "좋아": 0.8,
                            "행복": 1.0
                        },
                        "negative": {
                            "싫어": 1.0,
                            "우울": 1.2,
                            "짜증": 0.9
                        },
                        "danger": {
                            "죽고 싶": 2.0,
                            "끝내고 싶": 2.0
                        }
                    }

    def keyword_count_score(self, text: str) -> Dict[str, float]:
        """
        입력 텍스트에서 키워드를 검색하여 카테고리별 점수를 계산합니다.

        Args:
            text (str): 분석할 입력 텍스트.

        Returns:
            Dict[str, float]: 카테고리별 키워드 점수를 담은 딕셔너리.
                각 키는 'keyword_score_{category}' 형태로 반환됩니다.
                예:
                    {
                        "keyword_score_positive": 1.8,
                        "keyword_score_negative": 0.0,
                        "keyword_score_danger": 2.0
                    }
        """
        scores: Dict[str, float] = {}
        for category, keywords in self.lexicon.items():
            score = 0.0
            for word, weight in keywords.items():
                count = len(re.findall(word, text)) #  부분 일치 회수
                score += count * weight
            scores[f'keyword_score_{category}'] = score
        return scores