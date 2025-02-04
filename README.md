# 중소기업 신용예측 솔루션 개발

## 1. 문제상황
- 전체 기업의 99.9%, 전체 기업 종사자의 81.3% 가 중소기업으로 이루어져 있어 중소기업은 국내 경제의 버팀목이라고 할 수 있음
- 한편, 중소기업은 재무정보의 신뢰성이 낮고 비대칭성이 높으며, 금융 충격(Financial shock)에 취약해 연쇄도산의 위험이 존재함

## 2. 필요성
- (정부 차원) 과학적이고 체계적인 신용예측 시스템을 도입하여 중소기업의 연쇄 도산을 막을 필요성이 있음
- (금융 기관) 중소기업은 재무정보의 신뢰성이 낮고 비대칭성이 높으므로 머신러닝을 통해 이를 보완 할 수 있는 방법론 도입이 필요함
- (거래 기업) 사업 리스크 관리를 위해 저신용기업과의 거래를 회피할 필요성이 존재함

## 3. 데이터셋
- 중소기업현황정보시스템 재무 데이터
- 국세청 사업자등록상태조회 활용

## 4. 결과 및 액션
- 중소기업 신용예측 AI 모델을 API 서비스로 구현, 배포하였음

## 5. 분석 내용
- 데이터 수집(셀레니움을 활용한 동적스크레이핑)
- 데이터 적재(MongoDB 활용)
- API 서비스 개발(FLASK)
- 웹 배포(AWS EC2)
