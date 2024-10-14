### from_s3/

Column|Description
---|---
code|거래된 암호화폐의 종목 코드 (예: BTC-USD), 암호화폐 시장에서의 거래 쌍
trade_date|거래가 발생한 날짜 (형식: YYYY-MM-DD)
trade_time|거래가 발생한 시간 (형식: HH:MM)
trade_timestamp|거래가 발생한 시간의 UNIX 타임스탬프 (밀리초 단위), 1970년 1월 1일 00:00:00 UTC 이후의 밀리초 단위 시간
high_price|거래된 시간 범위 내 가장 높은 가격을 기록한 시점의 가격
low_price|거래된 시간 범위 내 가장 낮은 가격을 기록한 시점의 가격
trade_price|해당 거래의 실제 거래 가격
change|가격 변동의 유형 (RISE, FALL, EVEN)
change_price|이전 거래와 비교한 가격 변동량 (양수면 상승, 음수면 하락)
change_rate|가격 변동률 (백분율), 가격이 몇 퍼센트 변동했는지
timestamp|거래 데이터가 기록된 시점의 UNIX 타임스탬프 (밀리초 단위)


### bitcoin_2017_to_2023.csv

+ 분 당 가격

Column|Description
---|---
timestamp|거래 시간
open|시작가 (USD)
high|고가 (USD)
low|저가 (USD)
close|종가 (USD)
volume|거래량
quote_asset_volume|지정된 시간 동안 발생한 거래에서 달러(USD)로 측정된 총 거래 금액
number_of_trades|거래 수 (거래된 코인 개수)
taker_buy_base_asset_volume|테이커 거래자들이 해당 기간동안 구매한 비트코인(BTC)의 총량
taker_buy_quote_asset_volume|테이커가 비트코인을 구매할 때 사용한 달러의 총


### from_pyupbit

Column|Description
---|---
timestamp|거래 시간
open|시작가 (USD)
high|고가 (USD)
low|저가 (USD)
close|종가 (USD)
volume|거래량 (거래된 코인 개수)
value|거래대금