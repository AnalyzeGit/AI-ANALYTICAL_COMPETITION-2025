import pandas as pd

class UtterancePreprocessor:
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
    
    def join_texts_by_window(self) -> pd.DataFrame:
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
        return utterances
    
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
        
        return (
            self.remove_missing()
            .generate_window_id()
            .join_texts_by_window()
        )