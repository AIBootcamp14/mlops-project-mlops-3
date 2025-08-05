"""
영화 평점 예측을 위한 데이터 스키마 정의
Pydantic 모델을 사용하여 입력/출력 데이터 구조와 검증 규칙을 정의합니다.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class MovieInput(BaseModel):
    """
    영화 평점 예측을 위한 입력 데이터 스키마
    TMDb API에서 가져올 수 있는 주요 영화 정보들을 포함합니다.
    """
    
    # 기본 정보
    title: str = Field(
        ...,  # 필수 필드
        description="영화 제목",
        example="아바타: 물의 길"
    )
    
    overview: Optional[str] = Field(
        None,  # 선택적 필드
        description="영화 줄거리/개요",
        example="판도라 행성에서 펼쳐지는 제이크 설리 가족의 모험..."
    )
    
    # 장르 정보
    genres: Optional[List[str]] = Field(
        None,
        description="영화 장르 목록",
        example=["액션", "어드벤처", "SF"]
    )
    
    # 제작 정보
    runtime: Optional[int] = Field(
        None,
        description="상영 시간 (분)",
        example=192,
        ge=1,  # 1분 이상
        le=500  # 500분 이하 (현실적 범위)
    )
    
    budget: Optional[float] = Field(
        None,
        description="제작비 (달러)",
        example=350000000.0,
        ge=0  # 0 이상
    )
    
    # 출시 정보
    release_date: Optional[str] = Field(
        None,
        description="개봉일 (YYYY-MM-DD 형식)",
        example="2022-12-14"
    )
    
    # 인기도/점수 관련
    popularity: Optional[float] = Field(
        None,
        description="TMDb 인기도 점수",
        example=2547.815,
        ge=0
    )
    
    # 제작진 정보
    director: Optional[str] = Field(
        None,
        description="감독명",
        example="제임스 카메론"
    )
    
    production_companies: Optional[List[str]] = Field(
        None,
        description="제작사 목록",
        example=["20th Century Studios", "Lightstorm Entertainment"]
    )
    
    # 언어/지역 정보
    original_language: Optional[str] = Field(
        None,
        description="원본 언어 코드",
        example="en"
    )
    
    production_countries: Optional[List[str]] = Field(
        None,
        description="제작 국가 목록",
        example=["US"]
    )


class PredictionResult(BaseModel):
    """
    영화 평점 예측 결과 스키마
    모델의 예측 결과와 관련 메타데이터를 포함합니다.
    """
    
    # 예측 결과
    predicted_rating: float = Field(
        ...,
        description="예측된 평점 (0-10 범위)",
        example=7.8,
        ge=0,
        le=10
    )
    
    # 신뢰도/확률 정보
    confidence: Optional[float] = Field(
        None,
        description="예측 신뢰도 (0-1 범위)",
        example=0.85,
        ge=0,
        le=1
    )
    
    # 메타데이터
    model_version: str = Field(
        ...,
        description="사용된 모델 버전",
        example="v1.0.0"
    )
    
    prediction_timestamp: datetime = Field(
        ...,
        description="예측 수행 시각",
        example="2024-01-15T10:30:00"
    )
    
    # 입력 데이터 요약
    input_summary: dict = Field(
        ...,
        description="예측에 사용된 입력 데이터 요약",
        example={
            "title": "아바타: 물의 길",
            "genres": ["액션", "어드벤처", "SF"],
            "runtime": 192
        }
    )


class PredictionResponse(BaseModel):
    """
    API 응답 전체 스키마
    성공/실패 상태와 함께 예측 결과나 오류 정보를 포함합니다.
    """
    
    success: bool = Field(
        ...,
        description="예측 성공 여부",
        example=True
    )
    
    message: str = Field(
        ...,
        description="응답 메시지",
        example="예측이 성공적으로 완료되었습니다."
    )
    
    data: Optional[PredictionResult] = Field(
        None,
        description="예측 결과 데이터 (성공시에만 포함)"
    )
    
    error_details: Optional[dict] = Field(
        None,
        description="오류 상세 정보 (실패시에만 포함)",
        example={
            "error_type": "ValidationError",
            "error_message": "필수 필드가 누락되었습니다."
        }
    )


class HealthResponse(BaseModel):
    """
    헬스체크 응답 스키마
    """
    
    status: str = Field(
        ...,
        description="서버 상태",
        example="healthy"
    )
    
    timestamp: datetime = Field(
        ...,
        description="응답 시각"
    )
    
    version: str = Field(
        ...,
        description="API 버전",
        example="0.1.0"
    )
    
    model_status: Optional[str] = Field(
        None,
        description="ML 모델 상태",
        example="loaded"
    )