import re
import pandas as pd
import kss
from typing import List, Tuple, Final

class UtterancePreprocessor:
    _FILLERS:Final  = re.compile(r'\b(음|어|그|막|약간|그러니까|뭐지|진짜|아니|그러면)\b')
    def __init__(self, df: pd.DataFrame) -> None:
        """
        발화(utterance) 데이터 전처리를 위한 클래스.

        Args:
        ----------
        df : pd.DataFrame
            발화 데이터프레임. 최소한 ['doll_id', 'uttered_at', 'text'] 컬럼을 포함해야 함.
        """
        self.df = df

    def remove_missing(self) -> "UtterancePreprocessor":
        """
        결측치가 포함된 행을 제거.

        Returns
        -------
        UtterancePreprocessor
            자기 자신(self)을 반환하여 체이닝 가능.
        """
        self.df = self.df.dropna()
        return self

    def generate_window_id(self) -> "UtterancePreprocessor":
        """
        doll_id별로 시간순 정렬 후, 5분 간격 기준으로 window_id를 생성.

        Returns
        -------
        UtterancePreprocessor
            자기 자신(self)을 반환하여 체이닝 가능.
        """

        # doll_id별, 시간순 정렬
        self.df = self.df.sort_values(by=['doll_id', 'uttered_at']).reset_index(drop=True)

        # 윈도우 기준 생성
        self.df['window_id'] = (
            self.df.groupby('doll_id')['uttered_at']
            .transform(lambda x: x.diff().gt(pd.Timedelta(minutes=5)).cumsum().astype(int))
        )
        return self
    
    def join_texts_by_window(self) -> "UtterancePreprocessor":
        """
        doll_id와 window_id를 기준으로 text를 조인하여 utterances 데이터 생성.

        Returns
        -------
        pd.DataFrame
            doll_id, window_id, texts 컬럼을 가진 DataFrame.
        """
        
        # doll_id, window_id를 기준으로 해당 텍스트 조인
        utterances = (
            self.df.groupby(['doll_id', 'window_id'])['text']
            .apply(lambda x: ' '.join(x))
            .to_frame()
            .reset_index()
            .rename(columns={'text': 'texts'}) 
        )
        self.utterances = utterances
        return self
    
    @staticmethod
    def kss_split_safe(text: str) -> List:
        """KSS 있으면 사용, 없으면 간단 규칙으로 백업."""
        if not isinstance(text, str) or not text.strip():
            return []
        try:
            sents = [s.strip() for s in kss.split_sentences(text) if s.strip()]
        except Exception:
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        return sents
    
    @classmethod
    def step2_clean(cls, sentence: str) -> str:
        s = sentence
        # [noise], [inaudible], (웃음) 등 제거
        s = re.sub(r'\[(?:noise|inaudible).*?\]|\(웃음\)', '', s, flags=re.I)
        # ㅋㅋㅋㅋ/ㅎㅎㅎㅎ 축약
        s = re.sub(r'(ㅋㅋ+|ㅎㅎ+)', lambda m: m.group(1)[:2], s)
        # 과도한 문자 반복 축약 (3회↑ -> 1회)
        s = re.sub(r'([가-힣A-Za-z])\1{2,}', r'\1', s)
        # 간단 추임새 제거 (감정 표현 “헐/와” 등은 남김)
        s = cls._FILLERS.sub('', s)
        # 공백 정리
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def split_sentences_keep_schema(
        self,
        text_col: str = "texts",
        id_cols: Tuple[str, str]= ("doll_id","window_id")) -> pd.DataFrame:
        """
        id_cols는 그대로 유지하고, text_col을 문장 단위로 분리해 행을 확장합니다.
        반환 컬럼:
        * 기존 id_cols
        * sentence_index (0..n-1)
        * sentence_count
        * sentence_text
        """
        work = self.utterances[:30].loc[:, list(id_cols) + [text_col]].copy()
        work['sentences'] = work[text_col].map(UtterancePreprocessor.kss_split_safe)
        work['sentences_count'] = work['sentences'].map(len)
        out = (
            work
            .explode('sentences', ignore_index=True)
            .rename(columns={'sentences':"sentence_text"})
        )
        # 문장 인덱스 부여(ID 조합마다 0,1,2...)
        out['sentence_index'] = (out
            .groupby(list(id_cols)).cumcount()
        )
        # 문장 없는 케이스(drop)
        out = out.dropna(subset=["sentence_text"]).reset_index(drop=True)
        # 보기 좋게 정렬
        return out[list(id_cols) + ["sentence_index","sentences_count","sentence_text"]]

    def preprocess_utterances(self) -> pd.DataFrame:
        """
        전처리 전체 파이프라인 실행:
        1) 결측치 제거
        2) window_id 생성
        3) 텍스트 조인

        Returns
        -------
        pd.DataFrame
            최종적으로 doll_id, window_id, texts 컬럼을 가진 DataFrame.
        """
        
        utterances = (
            self.remove_missing()
            .generate_window_id()
            .join_texts_by_window()
            .split_sentences_keep_schema()
        )
        utterances['sentence_text'] = utterances['sentence_text'].map(UtterancePreprocessor.step2_clean)
        return utterances