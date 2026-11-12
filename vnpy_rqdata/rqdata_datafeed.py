import os
import re
from math import gcd
from datetime import date, datetime, time, timedelta
from typing import Any, cast
from collections.abc import Callable
from contextlib import nullcontext
from time import monotonic, sleep
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
STANDARD_OPTIONS_PATTERN = re.compile(r"^([A-Za-z]+)([0-9]{4})-?([CPcp])-?([0-9]+(?:\.[0-9]+)?)$")
TRADING_SESSION_CONTRACT_DETAIL_BATCH_SIZE = "TRADING_SESSION_CONTRACT_DETAIL_BATCH_SIZE"
TRADING_SESSION_CONTRACT_DETAIL_REQUEST_INTERVAL_MS = "TRADING_SESSION_CONTRACT_DETAIL_REQUEST_INTERVAL_MS"
HISTORY_TICK_FIELDS = [
    "open", "high", "low", "last", "prev_close", "volume", "total_turnover",
    "limit_up", "limit_down", "b1", "b2", "b3", "b4", "b5",
    "a1", "a2", "a3", "a4", "a5", "b1_v", "b2_v", "b3_v", "b4_v", "b5_v",
    "a1_v", "a2_v", "a3_v", "a4_v", "a5_v",
]


def contract_detail_batch_size() -> int:
    try:
        value = int(os.getenv(TRADING_SESSION_CONTRACT_DETAIL_BATCH_SIZE, "200"))
    except ValueError:
        raise ValueError(f"{TRADING_SESSION_CONTRACT_DETAIL_BATCH_SIZE} must be an integer") from None
    if not 1 <= value <= 100_000:
        raise ValueError(f"{TRADING_SESSION_CONTRACT_DETAIL_BATCH_SIZE} must be between 1 and 100000")
    return value


def contract_detail_request_interval_ms() -> int:
    try:
        value = int(os.getenv(TRADING_SESSION_CONTRACT_DETAIL_REQUEST_INTERVAL_MS, "200"))
    except ValueError:
        raise ValueError(f"{TRADING_SESSION_CONTRACT_DETAIL_REQUEST_INTERVAL_MS} must be an integer") from None
    if not 0 <= value <= 60_000:
        raise ValueError(f"{TRADING_SESSION_CONTRACT_DETAIL_REQUEST_INTERVAL_MS} must be between 0 and 60000")
    return value


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
        option = re.fullmatch(r"([A-Za-z]+)([0-9]{3,4})-?([CPcp])-?([0-9]+(?:\.[0-9]+)?)", symbol)
        if option is not None:
            product, delivery, right, strike = option.groups()
            compact = f"{product}{delivery}{right}{strike}".upper()
            if exchange is not Exchange.CZCE or len(delivery) == 4:
                return compact
            candidates = [f"{product}{decade}{delivery}{right}{strike}".upper() for decade in ("2", "1")]
            return next((candidate for candidate in candidates if candidate in all_symbols), candidates[-1])
        for count, word in enumerate(symbol):  # noqa: B007
            if word.isdigit():
                break

        product: str = symbol[:count]
        time_str: str = symbol[count:]

        # 期货
        if time_str.isdigit():
            # 只有郑商所需要特殊处理
            if exchange is not Exchange.CZCE or len(time_str) == 4:
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
        self.request_controller: Any | None = None
        self.request_account = self.username or "rqdata"
        self._history_calendar_scope: tuple[str, str | None, str | None] | None = None
        self._next_trading_dates: dict[date, date] = {}

    def set_history_calendar_scope(
        self, job_id: str | None, calendar_snapshot_id: str | None = None,
        calendar_rule_version: str | None = None,
    ) -> None:
        scope = (job_id, calendar_snapshot_id, calendar_rule_version) if job_id else None
        if scope != getattr(self, "_history_calendar_scope", None):
            self._next_trading_dates = {}
        self._history_calendar_scope = scope

    def _next_trading_date(self, end: datetime) -> date:
        if getattr(self, "_history_calendar_scope", None) is None:
            return self._request_call("calendar_next_date", get_next_trading_date, end)
        day = end.date()
        cache = self._next_trading_dates
        if day in cache:
            return cache[day]
        result = self._request_call("calendar_next_date", get_next_trading_date, end)
        if isinstance(result, date) and result > day:
            if len(cache) >= 1024:
                cache.pop(next(iter(cache)))
            cache[day] = result
        return result

    def _request_call(
        self, endpoint: str, function: Callable[..., Any], *args: Any,
        ensure_active: Callable[[], None] | None = None, **kwargs: Any,
    ) -> Any:
        if ensure_active is not None:
            ensure_active()
        controller = getattr(self, "request_controller", None)
        request = getattr(controller, "request", None)
        if controller is not None and not callable(request):
            controller.throttle(
                self.request_account, endpoint, ensure_active=ensure_active,
            )
        scope = request(self.request_account, endpoint, ensure_active=ensure_active) if callable(request) else nullcontext(None)
        with scope as check:
            if callable(check):
                check()
            if ensure_active is not None:
                ensure_active()
            result = function(*args, **kwargs)
            capture = getattr(self, "calendar_raw_callback", None)
            if capture is not None:
                capture(endpoint, args, kwargs, result)
            if callable(check):
                check()
            if ensure_active is not None:
                ensure_active()
            return result

    def init(self, output: Callable = print, *, load_instruments: bool = True) -> bool:
        """初始化"""
        if self.inited:
            return True

        self._next_trading_dates = {}
        if not self.username:
            output("RQData数据服务初始化失败：用户名为空！")
            return False

        if not self.password:
            output("RQData数据服务初始化失败：密码为空！")
            return False

        try:
            self._request_call(
                "session_init", init,
                self.username,
                self.password,
                ("rqdatad-pro.ricequant.com", 16011),
                use_pool=True,
                max_pool_size=1,
                auto_load_plugins=False
            )

            self.instruments = self._request_call(
                "catalog_futures", all_instruments,
            ) if load_instruments else DataFrame(columns=["order_book_id"])
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

    @staticmethod
    def validate_trading_session_sync_settings() -> int:
        return contract_detail_batch_size()

    def get_instrument_exchanges(
        self, order_book_ids: list[str], *, ensure_active: Callable[[], None] | None = None,
    ) -> dict[str, str]:
        """Resolve RQData exchanges in one batch and cache the result."""
        missing = [item for item in order_book_ids if item not in self.instrument_exchanges]
        if missing:
            resolved = self._query_contract_details(
                missing, **({"ensure_active": ensure_active} if ensure_active is not None else {}),
            )
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

    def _query_contract_details(
        self, order_book_ids: list[str], *, ensure_active: Callable[[], None] | None = None,
    ) -> Any:
        cache = getattr(self, "_session_sync_cache", None)
        previous = (cache.get("last_detail_query_at") if cache is not None
                    else getattr(self, "_last_contract_detail_query_at", None))
        interval = contract_detail_request_interval_ms() / 1000
        if previous is not None and interval:
            remaining = interval - (monotonic() - previous)
            if remaining > 0:
                if ensure_active is not None:
                    ensure_active()
                sleep(remaining)
        if ensure_active is not None:
            ensure_active()
        started = monotonic()
        result = self._request_call("instrument_details", instruments, order_book_ids)
        if ensure_active is not None:
            ensure_active()
        self._last_contract_detail_query_at = started
        if cache is not None:
            cache["last_detail_query_at"] = started
        return result

    def is_trading_day(self, value: date) -> bool:
        """Return whether the date is in RQData's Chinese trading calendar."""
        if not self.inited and not self.init():
            raise RuntimeError("RQData 数据服务初始化失败")
        return len(self._request_call("calendar_days", get_trading_dates, value, value)) > 0

    def begin_trading_session_sync(self) -> None:
        self._session_sync_cache = {
            "catalogs": {}, "details": {}, "calendars": {},
            "last_detail_query_at": getattr(self, "_last_contract_detail_query_at", None),
        }

    def end_trading_session_sync(self) -> None:
        self._session_sync_cache = None

    def get_trading_days(
        self, start: date, end: date, *, ensure_active: Callable[[], None] | None = None,
    ) -> tuple[date, ...]:
        """Return a date range using a single RQData calendar request."""
        if not self.inited and not self.init():
            raise RuntimeError("RQData 数据服务初始化失败")
        cache = getattr(self, "_session_sync_cache", None)
        if cache is not None:
            for (cached_start, cached_end), days in cache["calendars"].items():
                if cached_start <= start and cached_end >= end:
                    return tuple(day for day in days if start <= day <= end)
        if ensure_active is not None:
            ensure_active()
        raw_days = self._request_call("calendar_days", get_trading_dates, start, end,
                                      ensure_active=ensure_active)
        days = tuple(
            cast(Any, value).date() if hasattr(value, "date") else value
            for value in raw_days
        )
        if ensure_active is not None:
            ensure_active()
        if cache is not None:
            cache["calendars"][(start, end)] = days
        return days

    def get_trading_session_snapshot(
        self, vt_symbols: list[str], trading_date: date, output: Callable = print,
        *, ensure_active: Callable[[], None] | None = None,
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
        active_options = {"ensure_active": ensure_active} if ensure_active is not None else {}
        trading_hours = self._trading_hours_by_order_book_id(order_book_ids, trading_date, **active_options)
        if ensure_active is not None:
            ensure_active()
        statuses = self._instrument_status_by_order_book_id(
            order_book_ids, trading_date, **active_options,
        )
        return {
            vt_symbol: {
                "sessions": _parse_trading_hours(trading_hours.get(order_book_id, "")),
                "status": statuses.get(order_book_id, "Unknown"),
            }
            for vt_symbol, order_book_id in order_book_id_by_vt_symbol.items()
        }

    def get_all_futures_trading_sessions(
        self, trading_date: date, output: Callable = print,
        *, ensure_active: Callable[[], None] | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        batch_size: int | None = None,
        checkpoint_batch_size: int | None = None,
        batch_checkpoint: Callable[[int, list[str], dict[str, dict[str, Any]] | None],
                                   dict[str, dict[str, Any]] | None] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return session periods and status for every listed domestic futures contract."""
        return self._get_all_contract_trading_sessions(
            trading_date, "futures", output, ensure_active=ensure_active, progress=progress,
            batch_size=batch_size, checkpoint_batch_size=checkpoint_batch_size,
            batch_checkpoint=batch_checkpoint,
        )

    def get_all_options_trading_sessions(
        self, trading_date: date, output: Callable = print,
        *, ensure_active: Callable[[], None] | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        batch_size: int | None = None,
        checkpoint_batch_size: int | None = None,
        batch_checkpoint: Callable[[int, list[str], dict[str, dict[str, Any]] | None],
                                   dict[str, dict[str, Any]] | None] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return actual session periods and status for domestic derivatives options."""
        return self._get_all_contract_trading_sessions(
            trading_date, "options", output, ensure_active=ensure_active, progress=progress,
            batch_size=batch_size, checkpoint_batch_size=checkpoint_batch_size,
            batch_checkpoint=batch_checkpoint,
        )

    def _get_all_contract_trading_sessions(
        self, trading_date: date, asset_class: str, output: Callable,
        *, ensure_active: Callable[[], None] | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        batch_size: int | None = None,
        checkpoint_batch_size: int | None = None,
        batch_checkpoint: Callable[[int, list[str], dict[str, dict[str, Any]] | None],
                                   dict[str, dict[str, Any]] | None] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if batch_size is None:
            batch_size = contract_detail_batch_size()
        elif type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if checkpoint_batch_size is not None and (type(checkpoint_batch_size) is not int or checkpoint_batch_size < 1):
            raise ValueError("checkpoint_batch_size must be a positive integer")
        checkpoint_size = gcd(batch_size, checkpoint_batch_size or batch_size)
        active_options = {"ensure_active": ensure_active} if ensure_active is not None else {}
        if ensure_active is not None:
            ensure_active()
        if not self.inited and not self.init(output):
            raise RuntimeError("RQData 数据服务初始化失败")
        cache = getattr(self, "_session_sync_cache", None)
        refreshed = cache["catalogs"].get(asset_class) if cache is not None else None
        if refreshed is None:
            if progress is not None:
                progress({"stage": "catalog", "message": f"下载 {asset_class} 合约目录"})
            refreshed = self._request_call(
                "catalog_options" if asset_class == "options" else "catalog_futures",
                all_instruments, **({"type": "Option"} if asset_class == "options" else {}),
                ensure_active=ensure_active,
            )
            if cache is not None:
                cache["catalogs"][asset_class] = refreshed
        if refreshed is None or refreshed.empty:
            raise RuntimeError(f"RQData returned an empty {asset_class} catalog")
        if asset_class == "futures":
            self.instruments = refreshed
            self.symbols = refreshed["order_book_id"].to_numpy()
        candidates: list[tuple[str, Any]] = []
        for row in refreshed.itertuples(index=False):
            if ("option" if asset_class == "options" else "future") not in str(getattr(row, "type", "")).lower():
                continue
            order_book_id = str(getattr(row, "order_book_id", "")).strip()
            pattern = STANDARD_OPTIONS_PATTERN if asset_class == "options" else STANDARD_FUTURES_PATTERN
            if not pattern.fullmatch(order_book_id):
                continue
            candidates.append((order_book_id, row))

        exchange_by_symbol: dict[str, str] = {}
        if "exchange" not in refreshed.columns:
            exchange_by_symbol = self.get_instrument_exchanges(
                [order_book_id for order_book_id, _ in candidates], **active_options,
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
            if asset_class == "options":
                option = STANDARD_OPTIONS_PATTERN.fullmatch(order_book_id)
                if option is None:
                    continue
                product, delivery, right, strike = option.groups()
                product = product.upper() if exchange in {"CFFEX", "CZCE"} else product.lower()
                symbol = f"{product}{delivery}-{right.upper()}-{strike}" if exchange in {"CFFEX", "DCE", "GFEX"} else f"{product}{delivery}{right.upper()}{strike}"
            contracts[order_book_id] = (symbol, exchange)

        result: dict[str, dict[str, Any]] = {}
        metadata = {order_book_id: row for order_book_id, row in candidates}
        calendar = self.get_trading_days(trading_date - timedelta(days=40), trading_date, **active_options)
        previous_date = next((value for value in reversed(calendar) if value < trading_date), None)
        is_trading_date = trading_date in calendar
        order_book_ids = sorted(contracts) if batch_checkpoint is not None else list(contracts)
        order_book_id_by_vt_symbol = {
            f"{symbol}.{exchange}": order_book_id
            for order_book_id, (symbol, exchange) in contracts.items()
        } if batch_checkpoint is not None else {}
        for offset in range(0, len(order_book_ids), batch_size):
            if ensure_active is not None:
                ensure_active()
            batch = order_book_ids[offset : offset + batch_size]
            batch_number = offset // batch_size + 1
            batch_vt_symbols = [f"{contracts[symbol][0]}.{contracts[symbol][1]}" for symbol in batch]
            pending_checkpoints: list[tuple[int, list[str]]] = []
            if batch_checkpoint is not None:
                for local_offset in range(0, len(batch), checkpoint_size):
                    checkpoint_index = (offset + local_offset) // checkpoint_size
                    checkpoint_symbols = batch_vt_symbols[local_offset:local_offset + checkpoint_size]
                    cached = batch_checkpoint(checkpoint_index, checkpoint_symbols, None)
                    if cached is not None:
                        result.update(cached)
                    else:
                        pending_checkpoints.append((checkpoint_index, checkpoint_symbols))
                if not pending_checkpoints:
                    continue
                try:
                    detail_batch = [
                        order_book_id_by_vt_symbol[vt_symbol]
                        for _, checkpoint_symbols in pending_checkpoints
                        for vt_symbol in checkpoint_symbols
                    ]
                except KeyError as exc:
                    raise RuntimeError("RQData checkpoint references an unknown contract") from exc
            else:
                detail_batch = batch
            if progress is not None:
                progress({
                    "stage": "contract_details",
                    "completed": offset,
                    "total": len(order_book_ids),
                    "message": f"查询{asset_class}合约详情，第 {batch_number} 批（已处理 {offset}/{len(order_book_ids)}）",
                })
            details_cache = cache["details"] if cache is not None else {}
            missing = [symbol for symbol in detail_batch if symbol not in details_cache]
            if missing:
                resolved = self._query_contract_details(missing, **active_options)
                if not isinstance(resolved, list):
                    resolved = [resolved]
                details_cache.update({item.order_book_id: item for item in resolved if item is not None})
            details = {symbol: details_cache[symbol] for symbol in detail_batch if symbol in details_cache}
            active: set[str] = set()
            period_groups: dict[tuple[str, str], list[str]] = {}
            batch_result: dict[str, dict[str, Any]] = {}
            for order_book_id in detail_batch:
                item = details.get(order_book_id, metadata[order_book_id])
                listed = self._optional_date(getattr(item, "listed_date", None))
                last = self._optional_date(getattr(item, "de_listed_date", None))
                if listed is None or last is None or listed > last:
                    raise ValueError(f"RQData missing/invalid lifecycle: {order_book_id}")
                if listed <= trading_date <= last:
                    active.add(order_book_id)
                    underlying = str(getattr(item, "underlying_symbol", "") or "").strip()
                    if underlying.lower() in {"", "nan", "none"}:
                        underlying = order_book_id
                    group = (contracts[order_book_id][1], underlying)
                    period_groups.setdefault(group, []).append(order_book_id)
            period_cache = cache.setdefault("trading_hours", {}) if cache is not None else {}
            missing_groups = {
                group: symbols for group, symbols in period_groups.items()
                if (trading_date, asset_class, *group) not in period_cache
            }
            if progress is not None:
                progress({
                    "stage": "trading_hours_query", "completed": offset, "total": len(order_book_ids),
                    "message": f"查询{asset_class}交易时段，第 {batch_number} 批（本批 {len(batch)} 个合约，在市 {len(active)} 个，待查 {len(missing_groups)} 个品种，已处理 {offset}/{len(order_book_ids)}）",
                })
            queried_hours = self._trading_hours_by_order_book_id(
                [symbols[0] for symbols in missing_groups.values()], trading_date, **active_options,
            ) if missing_groups and is_trading_date else {}
            for group, symbols in missing_groups.items():
                if symbols[0] in queried_hours:
                    period_cache[(trading_date, asset_class, *group)] = queried_hours[symbols[0]]
            trading_hours = {
                symbol: period_cache[(trading_date, asset_class, *group)]
                for group, symbols in period_groups.items()
                if is_trading_date and (trading_date, asset_class, *group) in period_cache
                for symbol in symbols
            }
            if progress is not None:
                progress({
                    "stage": "trading_hours",
                    "completed": min(offset + len(batch), len(order_book_ids)),
                    "total": len(order_book_ids),
                    "message": f"已处理 {min(offset + len(batch), len(order_book_ids))}/{len(order_book_ids)} 个{asset_class}合约",
                })
            for order_book_id in detail_batch:
                symbol, exchange = contracts[order_book_id]
                item = details.get(order_book_id, metadata[order_book_id])
                listed = self._optional_date(getattr(item, "listed_date", None))
                last = self._optional_date(getattr(item, "de_listed_date", None))
                status = str(getattr(item, "status", "") or "Active")
                if status.lower() in {"nan", "none", "unknown", "delisted", "de_listed", "notlisted"}:
                    status = "Active"
                if listed and trading_date < listed:
                    status = "NotListed"
                elif last and trading_date > last:
                    status = "Delisted"
                raw_hours = trading_hours.get(order_book_id, "")
                sessions = _parse_trading_hours(raw_hours)
                if raw_hours and len(sessions) != len(raw_hours.split(",")):
                    raise ValueError(f"RQData invalid trading periods: {order_book_id}")
                if is_trading_date and order_book_id in active and status.lower() in {"active", "trading"} and not sessions:
                    raise ValueError(f"RQData missing trading periods: {order_book_id}")
                batch_result[f"{symbol}.{exchange}"] = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "asset_class": asset_class,
                    "sessions": sessions,
                    "status": status,
                    "listed_date": listed,
                    "last_trading_date": last,
                    "individual_last_trading_date": self._optional_date(getattr(item, "individual_last_trading_date", None)),
                    "institutional_last_trading_date": self._optional_date(getattr(item, "institutional_last_trading_date", None)) or last,
                    "delivery_date": self._optional_date(getattr(item, "delivery_date", None)),
                    "previous_trading_date": previous_date,
                }
            if batch_checkpoint is not None:
                for checkpoint_index, checkpoint_symbols in pending_checkpoints:
                    checkpoint_result = {
                        symbol: batch_result[symbol] for symbol in checkpoint_symbols
                        if symbol in batch_result
                    }
                    if len(checkpoint_result) != len(checkpoint_symbols):
                        raise RuntimeError("RQData catalog batch omitted checkpointed contracts")
                    saved = batch_checkpoint(checkpoint_index, checkpoint_symbols, checkpoint_result)
                    result.update(saved if saved is not None else checkpoint_result)
            else:
                result.update(batch_result)
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

    def _trading_hours_by_order_book_id(
        self, order_book_ids: list[str], trading_date: date,
        *, ensure_active: Callable[[], None] | None = None,
    ) -> dict[str, str]:
        # frequency="tick" returns the exchange-defined session times rather than bar boundaries.
        if ensure_active is not None:
            ensure_active()
        frame = cast(
            DataFrame | None,
            self._request_call("trading_periods", get_trading_periods, order_book_ids,
                               start_date=trading_date, end_date=trading_date, frequency="tick",
                               ensure_active=ensure_active),
        )
        if ensure_active is not None:
            ensure_active()
        if frame is None or frame.empty:
            return {}
        result: dict[str, str] = {}
        for index, row in frame.iterrows():
            order_book_id = index[0] if isinstance(index, tuple) else index
            result[str(order_book_id)] = str(row["trading_hours"])
        return result

    def _instrument_status_by_order_book_id(
        self, order_book_ids: list[str], trading_date: date,
        *, ensure_active: Callable[[], None] | None = None,
    ) -> dict[str, str]:
        resolved = self._query_contract_details(
            order_book_ids, **({"ensure_active": ensure_active} if ensure_active is not None else {}),
        )
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

    def query_history_batch(
        self, requests: list[HistoryRequest], output: Callable = print, *, before_convert: Callable | None = None,
    ) -> list[list[BarData] | list[TickData] | None]:
        if not requests:
            return []
        if not self.inited and not self.init(output):
            raise RuntimeError("RQData batch initialization failed")
        first = requests[0]
        if any(
            req.exchange not in FUTURES_EXCHANGES or req.symbol.isalpha()
            or (req.interval, to_china_tz(req.start), to_china_tz(cast(datetime, req.end)))
            != (first.interval, to_china_tz(first.start), to_china_tz(cast(datetime, first.end)))
            for req in requests
        ):
            raise ValueError("Batch history requires concrete derivatives with matching interval and range")
        symbols = [to_rq_symbol(req.symbol, req.exchange, self.symbols) for req in requests]
        if len(set(symbols)) != len(symbols):
            raise ValueError("Batch history requires unique supported contracts")
        if any(symbol not in self.symbols for symbol in symbols):
            results = []
            for index, req in enumerate(requests):
                if before_convert is not None:
                    before_convert(index)
                results.append(self.query_tick_history(req, output) if req.interval == Interval.TICK
                               else self.query_bar_history(req, output))
            return results
        frequency = "tick" if first.interval == Interval.TICK else INTERVAL_VT2RQ.get(first.interval)
        if frequency is None:
            raise ValueError("Unsupported batch history interval")
        fields = list(HISTORY_TICK_FIELDS) if frequency == "tick" else [
            "open", "high", "low", "close", "volume", "total_turnover",
        ]
        fields.append("open_interest")
        source_end = self._next_trading_date(to_china_tz(cast(datetime, first.end)))
        frame = self._request_call(
            "history_tick" if frequency == "tick" else "history_bar", get_price, symbols,
            frequency=frequency, fields=fields, start_date=to_china_tz(first.start),
            end_date=source_end, adjust_type="none",
        )
        if frame is not None and not set(frame.index.get_level_values(0)).issubset(symbols):
            raise ValueError("RQData returned an unexpected contract in batch history")
        results = []
        for index, (req, symbol) in enumerate(zip(requests, symbols)):
            if before_convert is not None:
                before_convert(index)
            part = frame.loc[frame.index.get_level_values(0) == symbol].copy() if frame is not None else None
            source = (part, source_end, symbols)
            results.append(self._query_tick_history(req, output, source=source) if frequency == "tick"
                           else self._query_bar_history(req, output, source=source))
        return results

    def _query_bar_history(
        self, req: HistoryRequest, output: Callable = print, *, source: tuple | None = None,
    ) -> list[BarData] | None:
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
            if re.fullmatch(r"[A-Z]+[0-9]{4}[CP][0-9]+(?:\.[0-9]+)?", rq_symbol):
                raise ValueError(f"RQData 未返回或不支持该期权合约: {req.vt_symbol}")
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

        df = source[0] if source is not None else cast(DataFrame | None, self._request_call("history_bar", get_price,
            rq_symbol,
            frequency=rq_interval,
            fields=fields,
            start_date=start,
            end_date=self._next_trading_date(end),
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
        return self._query_tick_history(req, output)

    def _query_tick_history(
        self, req: HistoryRequest, output: Callable = print, *, source: tuple | None = None,
    ) -> list[TickData] | None:
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
            if re.fullmatch(r"[A-Z]+[0-9]{4}[CP][0-9]+(?:\.[0-9]+)?", rq_symbol):
                raise ValueError(f"RQData 未返回或不支持该期权合约: {req.vt_symbol}")
            output(f"RQData查询Tick数据失败：不支持的合约代码{req.vt_symbol}")
            return []

        # 只对衍生品合约才查询持仓量数据
        fields: list = list(HISTORY_TICK_FIELDS)
        if not symbol.isdigit():
            fields.append("open_interest")

        df = source[0] if source is not None else cast(DataFrame | None, self._request_call("history_tick", get_price,
            rq_symbol,
            frequency="tick",
            fields=fields,
            start_date=start,
            end_date=self._next_trading_date(end),
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

        df = cast(DataFrame | None, self._request_call("history_dominant", get_dominant_price,
            symbol.upper(),                         # 合约代码用大写
            frequency=rq_interval,
            fields=fields,
            start_date=start,
            end_date=self._next_trading_date(end),
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
