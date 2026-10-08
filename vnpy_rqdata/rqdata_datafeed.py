import re
from datetime import date, datetime, time, timedelta
from typing import Any, cast
from collections.abc import Callable
from zoneinfo import ZoneInfo

from numpy import ndarray
from pandas import DataFrame, Timestamp
from rqdatac import get_trading_periods, init, instruments
from rqdatac.services.get_price import get_price
from rqdatac.services.future import get_dominant_price
from rqdatac.services.basic import all_instruments
from rqdatac.services.calendar import get_next_trading_date, get_trading_dates
from rqdatac.share.errors import RQDataError

from vnpy.trader.setting import SETTINGS
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData, HistoryRequest
from vnpy.trader.utility import round_to
from vnpy.trader.datafeed import BaseDatafeed


INTERVAL_VT2RQ: dict[Interval, str] = {
    Interval.MINUTE: "1m",
    Interval.HOUR: "60m",
    Interval.DAILY: "1d",
}

INTERVAL_ADJUSTMENT_MAP: dict[Interval, timedelta] = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta()         # no need to adjust for daily bar
}

FUTURES_EXCHANGES: set[Exchange] = {
    Exchange.CFFEX,
    Exchange.SHFE,
    Exchange.CZCE,
    Exchange.DCE,
    Exchange.INE,
    Exchange.GFEX
}

CHINA_TZ = ZoneInfo("Asia/Shanghai")
RQ_EXCHANGE_TO_VT: dict[str, str] = {
    "CCFX": "CFFEX",
    "CFFEX": "CFFEX",
    "XSGE": "SHFE",
    "SHFE": "SHFE",
    "XZCE": "CZCE",
    "CZCE": "CZCE",
    "XDCE": "DCE",
    "DCE": "DCE",
    "XINE": "INE",
    "INE": "INE",
    "GFEX": "GFEX",
}
STANDARD_FUTURES_PATTERN = re.compile(r"^[A-Za-z]+[0-9]{4}$")


def to_china_tz(dt: datetime) -> datetime:
    """将 datetime 对象转换为 CHINA_TZ 时区"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=CHINA_TZ)
    dt = dt.astimezone(CHINA_TZ)
    return dt


def _parse_trading_hours(text: str) -> list[tuple[time, time]]:
    """解析形如 "09:31-11:30,13:01-15:00" 的交易时间段字符串"""
    sessions: list[tuple[time, time]] = []
    for part in text.split(","):
        start_text, separator, end_text = part.strip().partition("-")
        if not separator:
            continue
        try:
            sessions.append((time.fromisoformat(start_text.strip()), time.fromisoformat(end_text.strip())))
        except ValueError:
            continue
    return sessions


def to_rq_symbol(symbol: str, exchange: Exchange, all_symbols: ndarray) -> str:
    """将交易所代码转换为米筐代码"""
    # 股票
    if exchange in {Exchange.SSE, Exchange.SZSE}:
        if exchange == Exchange.SSE:
            rq_symbol: str = f"{symbol}.XSHG"
        else:
            rq_symbol = f"{symbol}.XSHE"
    # 金交所现货
    elif exchange == Exchange.SGE:
        for char in ["(", ")", "+"]:
            symbol = symbol.replace(char, "")
        symbol = symbol.upper()
        rq_symbol = f"{symbol}.SGEX"
    # 期货和期权
    elif exchange in {
        Exchange.CFFEX,
        Exchange.SHFE,
        Exchange.DCE,
        Exchange.CZCE,
        Exchange.INE,
        Exchange.GFEX
    }:
        for count, word in enumerate(symbol):  # noqa: B007
            if word.isdigit():
                break

        product: str = symbol[:count]
        time_str: str = symbol[count:]

        # 期货
        if time_str.isdigit():
            # 只有郑商所需要特殊处理
            if exchange is not Exchange.CZCE:
                return symbol.upper()

            # 检查是否为连续合约或者指数合约
            if time_str in ["88", "888", "99", "889"]:
                return symbol

            # 提取年月
            year: str = symbol[count]
            month: str = symbol[count + 1:]

            guess_1: str = f"{product}1{year}{month}".upper()
            guess_2: str = f"{product}2{year}{month}".upper()

            # 优先尝试20年后的合约
            if guess_2 in all_symbols:
                rq_symbol = guess_2
            else:
                rq_symbol = guess_1
        # 期权以及期货次主力连续合约
        else:
            if time_str == "88A2":
                return symbol.upper()

            if exchange in {
                Exchange.CFFEX,
                Exchange.DCE,
                Exchange.SHFE,
                Exchange.INE,
                Exchange.GFEX
            }:
                rq_symbol = symbol.replace("-", "").upper()
            elif exchange == Exchange.CZCE:
                year = symbol[count]
                suffix: str = symbol[count + 1:]

                guess_1 = f"{product}1{year}{suffix}".upper()
                guess_2 = f"{product}2{year}{suffix}".upper()

                # 优先尝试20年后的合约
                if guess_2 in all_symbols:
                    rq_symbol = guess_2
                else:
                    rq_symbol = guess_1
    else:
        rq_symbol = f"{symbol}.{exchange.value}"

    return rq_symbol


class RqdataDatafeed(BaseDatafeed):
    """米筐RQData数据服务接口"""

    def __init__(self) -> None:
        """"""
        self.username: str = SETTINGS["datafeed.username"]
        self.password: str = SETTINGS["datafeed.password"]

        self.inited: bool = False
        self.instruments: DataFrame = DataFrame()
        self.instrument_exchanges: dict[str, str] = {}

    def init(self, output: Callable = print) -> bool:
        """初始化"""
        if self.inited:
            return True

        if not self.username:
            output("RQData数据服务初始化失败：用户名为空！")
            return False

        if not self.password:
            output("RQData数据服务初始化失败：密码为空！")
            return False

        try:
            init(
                self.username,
                self.password,
                ("rqdatad-pro.ricequant.com", 16011),
                use_pool=True,
                max_pool_size=1,
                auto_load_plugins=False
            )

            self.instruments = all_instruments()
            self.symbols: ndarray = self.instruments["order_book_id"].to_numpy()
        except RQDataError as ex:
            output(f"RQData数据服务初始化失败：{ex}")
            return False
        except RuntimeError as ex:
            output(f"发生运行时错误：{ex}")
            return False
        except Exception as ex:
            output(f"发生未知异常：{ex}")
            return False

        self.inited = True
        return True

    def get_instrument_exchanges(self, order_book_ids: list[str]) -> dict[str, str]:
        """Resolve RQData exchanges in one batch and cache the result."""
        missing = [item for item in order_book_ids if item not in self.instrument_exchanges]
        if missing:
            resolved = instruments(missing)
            if not isinstance(resolved, list):
                resolved = [resolved]
            self.instrument_exchanges.update(
                {
                    item.order_book_id: item.exchange
                    for item in resolved
                    if item is not None and getattr(item, "exchange", None)
                }
            )
        return {
            order_book_id: self.instrument_exchanges[order_book_id]
            for order_book_id in order_book_ids
            if order_book_id in self.instrument_exchanges
        }

    def is_trading_day(self, value: date) -> bool:
        """Return whether the date is in RQData's Chinese trading calendar."""
        if not self.inited and not self.init():
            raise RuntimeError("RQData 数据服务初始化失败")
        return len(get_trading_dates(value, value)) > 0

    def get_trading_days(self, start: date, end: date) -> tuple[date, ...]:
        """Return a date range using a single RQData calendar request."""
        if not self.inited and not self.init():
            raise RuntimeError("RQData 数据服务初始化失败")
        return tuple(
            cast(Any, value).date() if hasattr(value, "date") else value
            for value in get_trading_dates(start, end)
        )

    def get_trading_session_snapshot(
        self, vt_symbols: list[str], trading_date: date, output: Callable = print
    ) -> dict[str, dict[str, Any]]:
        """Resolve each vt_symbol's exchange-defined session times and instrument status for one date."""
        if not self.inited and not self.init(output):
            raise RuntimeError("RQData 数据服务初始化失败")
        order_book_id_by_vt_symbol: dict[str, str] = {}
        for vt_symbol in vt_symbols:
            symbol, separator, exchange_value = vt_symbol.rpartition(".")
            if not separator:
                continue
            try:
                exchange = Exchange(exchange_value)
            except ValueError:
                continue
            order_book_id_by_vt_symbol[vt_symbol] = to_rq_symbol(symbol, exchange, self.symbols)
        if not order_book_id_by_vt_symbol:
            return {}

        order_book_ids = list(dict.fromkeys(order_book_id_by_vt_symbol.values()))
        trading_hours = self._trading_hours_by_order_book_id(order_book_ids, trading_date)
        statuses = self._instrument_status_by_order_book_id(order_book_ids, trading_date)
        return {
            vt_symbol: {
                "sessions": _parse_trading_hours(trading_hours.get(order_book_id, "")),
                "status": statuses.get(order_book_id, "Unknown"),
            }
            for vt_symbol, order_book_id in order_book_id_by_vt_symbol.items()
        }

    def get_all_futures_trading_sessions(
        self, trading_date: date, output: Callable = print
    ) -> dict[str, dict[str, Any]]:
        """Return session periods and status for every listed domestic futures contract."""
        if not self.inited and not self.init(output):
            raise RuntimeError("RQData 数据服务初始化失败")
        candidates: list[tuple[str, Any]] = []
        for row in self.instruments.itertuples(index=False):
            if "future" not in str(getattr(row, "type", "")).lower():
                continue
            order_book_id = str(getattr(row, "order_book_id", "")).strip()
            if not STANDARD_FUTURES_PATTERN.fullmatch(order_book_id):
                continue
            listed_date = self._optional_date(getattr(row, "listed_date", None))
            de_listed_date = self._optional_date(getattr(row, "de_listed_date", None))
            if listed_date and listed_date > trading_date:
                continue
            if de_listed_date and de_listed_date < trading_date:
                continue
            candidates.append((order_book_id, row))

        exchange_by_symbol: dict[str, str] = {}
        if "exchange" not in self.instruments.columns:
            exchange_by_symbol = self.get_instrument_exchanges(
                [order_book_id for order_book_id, _ in candidates]
            )
        contracts: dict[str, tuple[str, str]] = {}
        for order_book_id, row in candidates:
            raw_exchange = exchange_by_symbol.get(
                order_book_id, str(getattr(row, "exchange", ""))
            )
            exchange = RQ_EXCHANGE_TO_VT.get(raw_exchange)
            if exchange is None:
                continue
            symbol = order_book_id if exchange in {"CFFEX", "CZCE"} else order_book_id.lower()
            contracts[order_book_id] = (symbol, exchange)

        result: dict[str, dict[str, Any]] = {}
        order_book_ids = list(contracts)
        for offset in range(0, len(order_book_ids), 200):
            batch = order_book_ids[offset : offset + 200]
            trading_hours = self._trading_hours_by_order_book_id(batch, trading_date)
            statuses = self._instrument_status_by_order_book_id(batch, trading_date)
            for order_book_id in batch:
                symbol, exchange = contracts[order_book_id]
                result[f"{symbol}.{exchange}"] = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "sessions": _parse_trading_hours(trading_hours.get(order_book_id, "")),
                    "status": statuses.get(order_book_id, "Unknown"),
                }
        return result

    @staticmethod
    def _optional_date(value: Any) -> date | None:
        text = str(value).strip()
        if not text or text.lower() in {"nat", "nan", "none"} or text.startswith("0000-"):
            return None
        if hasattr(value, "date"):
            parsed = value.date()
            if isinstance(parsed, date):
                return parsed
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None

    @staticmethod
    def _trading_hours_by_order_book_id(order_book_ids: list[str], trading_date: date) -> dict[str, str]:
        # frequency="tick" returns the exchange-defined session times rather than bar boundaries.
        frame = cast(
            DataFrame | None,
            get_trading_periods(order_book_ids, start_date=trading_date, end_date=trading_date, frequency="tick"),
        )
        if frame is None or frame.empty:
            return {}
        result: dict[str, str] = {}
        for index, row in frame.iterrows():
            order_book_id = index[0] if isinstance(index, tuple) else index
            result[str(order_book_id)] = str(row["trading_hours"])
        return result

    @staticmethod
    def _instrument_status_by_order_book_id(
        order_book_ids: list[str], trading_date: date
    ) -> dict[str, str]:
        resolved = instruments(order_book_ids)
        if not isinstance(resolved, list):
            resolved = [resolved]
        statuses: dict[str, str] = {}
        for item in resolved:
            if item is None:
                continue
            status = str(getattr(item, "status", "")).strip()
            if status and status.lower() not in {"nan", "none"}:
                statuses[item.order_book_id] = status
                continue
            listed_date = RqdataDatafeed._optional_date(getattr(item, "listed_date", None))
            de_listed_date = RqdataDatafeed._optional_date(
                getattr(item, "de_listed_date", None)
            )
            if listed_date and listed_date > trading_date:
                statuses[item.order_book_id] = "NotListed"
            elif de_listed_date and de_listed_date < trading_date:
                statuses[item.order_book_id] = "Delisted"
            else:
                statuses[item.order_book_id] = "Active"
        return statuses

    def query_bar_history(self, req: HistoryRequest, output: Callable = print) -> list[BarData] | None:
        """查询K线数据"""
        # 期货品种且代码中没有数字（非具体合约），则查询主力连续
        if req.exchange in FUTURES_EXCHANGES and req.symbol.isalpha():
            return self._query_dominant_history(req, output)
        else:
            return self._query_bar_history(req, output)

    def _query_bar_history(self, req: HistoryRequest, output: Callable = print) -> list[BarData] | None:
        """查询K线数据"""
        if not self.inited:
            n: bool = self.init(output)
            if not n:
                return []

        symbol: str = req.symbol
        exchange: Exchange = req.exchange
        interval: Interval = cast(Interval, req.interval)
        start: datetime = to_china_tz(req.start)
        end: datetime = to_china_tz(cast(datetime, req.end))

        # 股票期权不添加交易所后缀
        if exchange in [Exchange.SSE, Exchange.SZSE] and symbol in self.symbols:
            rq_symbol: str = symbol
        else:
            rq_symbol = to_rq_symbol(symbol, exchange, self.symbols)

        # 检查查询的代码在范围内
        if rq_symbol not in self.symbols:
            output(f"RQData查询K线数据失败：不支持的合约代码{req.vt_symbol}")
            return []

        rq_interval: str | None = INTERVAL_VT2RQ.get(interval, None)
        if not rq_interval:
            output(f"RQData查询K线数据失败：不支持的时间周期{interval.value}")
            return []

        # 为了将米筐时间戳（K线结束时点）转换为VeighNa时间戳（K线开始时点）
        adjustment: timedelta = INTERVAL_ADJUSTMENT_MAP[interval]

        # 只对衍生品合约才查询持仓量数据
        fields: list = ["open", "high", "low", "close", "volume", "total_turnover"]
        if not symbol.isdigit():
            fields.append("open_interest")

        # 对于股票查询前复权K线数据
        if rq_symbol.endswith(".XSHG") or rq_symbol.endswith(".XSHE"):
            adjust_type: str = "pre_volume"
        else:
            adjust_type = "none"

        df = cast(DataFrame | None, get_price(
            rq_symbol,
            frequency=rq_interval,
            fields=fields,
            start_date=start,
            end_date=get_next_trading_date(end),        # 为了查询夜盘数据
            adjust_type=adjust_type
        ))

        data: list[BarData] = []

        if df is not None:
            # 填充NaN为0
            df.fillna(0, inplace=True)

            for raw_row in df.itertuples():
                row: Any = raw_row
                row_index: tuple[str, Timestamp] = cast(tuple[str, Timestamp], row.Index)
                dt: datetime = row_index[1].to_pydatetime() - adjustment
                dt = dt.replace(tzinfo=CHINA_TZ)

                if dt >= end:
                    break

                bar: BarData = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    datetime=dt,
                    open_price=round_to(row.open, 0.000001),
                    high_price=round_to(row.high, 0.000001),
                    low_price=round_to(row.low, 0.000001),
                    close_price=round_to(row.close, 0.000001),
                    volume=row.volume,
                    turnover=row.total_turnover,
                    open_interest=getattr(row, "open_interest", 0),
                    gateway_name="RQ"
                )

                data.append(bar)

        return data

    def query_tick_history(self, req: HistoryRequest, output: Callable = print) -> list[TickData] | None:
        """查询Tick数据"""
        if not self.inited:
            n: bool = self.init(output)
            if not n:
                return []

        symbol: str = req.symbol
        exchange: Exchange = req.exchange
        start: datetime = to_china_tz(req.start)
        end: datetime = to_china_tz(cast(datetime, req.end))

        # 股票期权不添加交易所后缀
        if exchange in [Exchange.SSE, Exchange.SZSE] and symbol in self.symbols:
            rq_symbol: str = symbol
        else:
            rq_symbol = to_rq_symbol(symbol, exchange, self.symbols)

        if rq_symbol not in self.symbols:
            output(f"RQData查询Tick数据失败：不支持的合约代码{req.vt_symbol}")
            return []

        # 只对衍生品合约才查询持仓量数据
        fields: list = [
            "open",
            "high",
            "low",
            "last",
            "prev_close",
            "volume",
            "total_turnover",
            "limit_up",
            "limit_down",
            "b1",
            "b2",
            "b3",
            "b4",
            "b5",
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "b1_v",
            "b2_v",
            "b3_v",
            "b4_v",
            "b5_v",
            "a1_v",
            "a2_v",
            "a3_v",
            "a4_v",
            "a5_v",
        ]
        if not symbol.isdigit():
            fields.append("open_interest")

        df = cast(DataFrame | None, get_price(
            rq_symbol,
            frequency="tick",
            fields=fields,
            start_date=start,
            end_date=get_next_trading_date(end),        # 为了查询夜盘数据
            adjust_type="none"
        ))

        data: list[TickData] = []

        if df is not None:
            # 填充NaN为0
            df.fillna(0, inplace=True)

            for raw_row in df.itertuples():
                row: Any = raw_row
                row_index: tuple[str, Timestamp] = cast(tuple[str, Timestamp], row.Index)
                dt: datetime = row_index[1].to_pydatetime()
                dt = dt.replace(tzinfo=CHINA_TZ)

                if dt >= end:
                    break

                tick: TickData = TickData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=dt,
                    open_price=row.open,
                    high_price=row.high,
                    low_price=row.low,
                    pre_close=row.prev_close,
                    last_price=row.last,
                    volume=row.volume,
                    turnover=row.total_turnover,
                    open_interest=getattr(row, "open_interest", 0),
                    limit_up=row.limit_up,
                    limit_down=row.limit_down,
                    bid_price_1=row.b1,
                    bid_price_2=row.b2,
                    bid_price_3=row.b3,
                    bid_price_4=row.b4,
                    bid_price_5=row.b5,
                    ask_price_1=row.a1,
                    ask_price_2=row.a2,
                    ask_price_3=row.a3,
                    ask_price_4=row.a4,
                    ask_price_5=row.a5,
                    bid_volume_1=row.b1_v,
                    bid_volume_2=row.b2_v,
                    bid_volume_3=row.b3_v,
                    bid_volume_4=row.b4_v,
                    bid_volume_5=row.b5_v,
                    ask_volume_1=row.a1_v,
                    ask_volume_2=row.a2_v,
                    ask_volume_3=row.a3_v,
                    ask_volume_4=row.a4_v,
                    ask_volume_5=row.a5_v,
                    gateway_name="RQ"
                )

                data.append(tick)

        return data

    def _query_dominant_history(self, req: HistoryRequest, output: Callable = print) -> list[BarData] | None:
        """查询期货主力K线数据"""
        if not self.inited:
            n: bool = self.init(output)
            if not n:
                return []

        symbol: str = req.symbol
        exchange: Exchange = req.exchange
        interval: Interval = cast(Interval, req.interval)
        start: datetime = to_china_tz(req.start)
        end: datetime = to_china_tz(cast(datetime, req.end))

        rq_interval: str | None = INTERVAL_VT2RQ.get(interval, None)
        if not rq_interval:
            output(f"RQData查询K线数据失败：不支持的时间周期{interval.value}")
            return []

        # 为了将米筐时间戳（K线结束时点）转换为VeighNa时间戳（K线开始时点）
        adjustment: timedelta = INTERVAL_ADJUSTMENT_MAP[interval]

        # 只对衍生品合约才查询持仓量数据
        fields: list = ["open", "high", "low", "close", "volume", "total_turnover"]
        if not symbol.isdigit():
            fields.append("open_interest")

        df = cast(DataFrame | None, get_dominant_price(
            symbol.upper(),                         # 合约代码用大写
            frequency=rq_interval,
            fields=fields,
            start_date=start,
            end_date=get_next_trading_date(end),    # 为了查询夜盘数据
            adjust_type="pre",                      # 前复权
            adjust_method="prev_close_ratio"        # 切换前一日收盘价比例复权
        ))

        data: list[BarData] = []

        if df is not None:
            # 填充NaN为0
            df.fillna(0, inplace=True)

            for raw_row in df.itertuples():
                row: Any = raw_row
                row_index: tuple[str, Timestamp] = cast(tuple[str, Timestamp], row.Index)
                dt: datetime = row_index[1].to_pydatetime() - adjustment
                dt = dt.replace(tzinfo=CHINA_TZ)

                if dt >= end:
                    break

                bar: BarData = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    datetime=dt,
                    open_price=round_to(row.open, 0.000001),
                    high_price=round_to(row.high, 0.000001),
                    low_price=round_to(row.low, 0.000001),
                    close_price=round_to(row.close, 0.000001),
                    volume=row.volume,
                    turnover=row.total_turnover,
                    open_interest=getattr(row, "open_interest", 0),
                    gateway_name="RQ"
                )

                data.append(bar)

        return data
