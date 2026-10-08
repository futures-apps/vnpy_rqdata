from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

from vnpy_rqdata.rqdata_datafeed import RqdataDatafeed


class FakeInstrument:
    def __init__(
        self,
        order_book_id: str,
        listed_date: str,
        de_listed_date: str,
        *,
        status: str | None = None,
    ) -> None:
        self.order_book_id = order_book_id
        self.listed_date = listed_date
        self.de_listed_date = de_listed_date
        if status is not None:
            self.status = status


class TradingSessionStatusTest(unittest.TestCase):
    @patch("vnpy_rqdata.rqdata_datafeed.instruments")
    def test_missing_status_is_derived_from_listing_dates(self, mock_instruments) -> None:
        mock_instruments.return_value = [
            FakeInstrument("FU2611", "2025-11-03", "2026-10-30"),
            FakeInstrument("FU2609", "2025-09-01", "2026-08-31"),
            FakeInstrument("FU2701", "2026-10-01", "2026-12-31"),
        ]

        statuses = RqdataDatafeed._instrument_status_by_order_book_id(
            ["FU2611", "FU2609", "FU2701"],
            date(2026, 9, 28),
        )

        self.assertEqual("Active", statuses["FU2611"])
        self.assertEqual("Delisted", statuses["FU2609"])
        self.assertEqual("NotListed", statuses["FU2701"])

    @patch("vnpy_rqdata.rqdata_datafeed.instruments")
    def test_explicit_status_is_preserved(self, mock_instruments) -> None:
        mock_instruments.return_value = FakeInstrument(
            "FU2611",
            "2025-11-03",
            "2026-10-30",
            status="Suspended",
        )

        statuses = RqdataDatafeed._instrument_status_by_order_book_id(
            ["FU2611"],
            date(2026, 9, 28),
        )

        self.assertEqual("Suspended", statuses["FU2611"])


if __name__ == "__main__":
    unittest.main()
