from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

import geomosaic_hg.clients.gdelt as gdelt_module
import geomosaic_hg.clients.http as http_module
from geomosaic_hg.clients.http import HTTPClientError
from geomosaic_hg.clients.gdelt import GDELTDOCClient, gdelt_doc_article_to_asset
from geomosaic_hg.external_assets import collect_existing_external_assets


class GDELTDOCClientTest(unittest.TestCase):
    def test_gdelt_doc_search_tolerates_invalid_json_backslash_escapes(self) -> None:
        body = b'{"articles":[{"url":"https://example.test/a","title":"Hong Kong \\\\ policy","seendate":"20200701021500"},{"url":"https://example.test/b","title":"Bad \\& escape"}]}'
        calls = []

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return body

        def fake_urlopen(request, timeout=30):
            calls.append(request.full_url)
            return FakeResponse()

        client = GDELTDOCClient()
        with patch.object(http_module, "urlopen", fake_urlopen):
            rows = client.search_articles("Hong Kong", max_records=2)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], "Hong Kong \\ policy")
        self.assertEqual(rows[1]["title"], "Bad \\& escape")
        self.assertIn("format=json", calls[0])

    def test_gdelt_doc_search_uses_artlist_json_parameters(self) -> None:
        calls = []

        def fake_get_json(url, params, timeout=30):
            calls.append((url, params, timeout))
            return SimpleNamespace(data={"articles": [{"url": "https://example.test/a", "title": "Fixture"}]})

        client = GDELTDOCClient(timeout=9)
        with patch.object(gdelt_module, "get_json", fake_get_json):
            rows = client.search_articles(
                "Hong Kong national security law",
                max_records=5,
                start_datetime="20200701000000",
                end_datetime="20200702000000",
            )

        self.assertEqual(rows, [{"url": "https://example.test/a", "title": "Fixture"}])
        url, params, timeout = calls[0]
        self.assertEqual(url, "https://api.gdeltproject.org/api/v2/doc/doc")
        self.assertEqual(timeout, 9)
        self.assertEqual(params["mode"], "artlist")
        self.assertEqual(params["format"], "json")
        self.assertEqual(params["query"], "Hong Kong national security law")
        self.assertEqual(params["maxrecords"], 5)
        self.assertEqual(params["startdatetime"], "20200701000000")
        self.assertEqual(params["enddatetime"], "20200702000000")

    def test_gdelt_doc_retries_rate_limited_requests(self) -> None:
        calls = []
        sleeps = []

        def fake_get_json(url, params, timeout=30):
            calls.append(params)
            if len(calls) == 1:
                raise HTTPClientError("HTTP 429 fixture", status=429)
            return SimpleNamespace(data={"articles": [{"url": "https://example.test/a", "title": "Fixture"}]})

        client = GDELTDOCClient(rate_limit_seconds=5, max_retries=2, retry_backoff_seconds=7)
        with patch.object(gdelt_module, "get_json", fake_get_json), patch.object(gdelt_module.time, "sleep", lambda seconds: sleeps.append(seconds)):
            rows = client.search_articles("Hong Kong", max_records=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [7])

    def test_gdelt_doc_retries_transient_network_failures(self) -> None:
        calls = []
        sleeps = []

        def fake_get_json(url, params, timeout=30):
            calls.append(params)
            if len(calls) == 1:
                raise HTTPClientError("Request failed for fixture: timed out")
            return SimpleNamespace(data={"articles": [{"url": "https://example.test/a", "title": "Fixture"}]})

        client = GDELTDOCClient(rate_limit_seconds=5, max_retries=2, retry_backoff_seconds=7)
        with patch.object(gdelt_module, "get_json", fake_get_json), patch.object(gdelt_module.time, "sleep", lambda seconds: sleeps.append(seconds)):
            rows = client.search_articles("Iraq", max_records=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [7])

    def test_gdelt_doc_article_to_asset_is_news_pointer_not_structured_event(self) -> None:
        article = {
            "url": "https://news.example.test/hk",
            "title": "Hong Kong national security law takes effect",
            "seendate": "20200701021500",
            "domain": "news.example.test",
            "language": "English",
            "sourcecountry": "Hong Kong",
            "socialimage": "https://news.example.test/hk.jpg",
        }

        asset = gdelt_doc_article_to_asset(article, "hongkong", query="Hong Kong national security law", temporal_relation="event_window")

        self.assertEqual(asset.asset_source, "GDELT_DOC")
        self.assertEqual(asset.modality, "text")
        self.assertEqual(asset.source_layer, "news")
        self.assertFalse(asset.redistribution_flag)
        self.assertEqual(asset.publish_time, "2020-07-01T02:15:00Z")
        self.assertEqual(asset.extra["collection_channel"], "gdelt_doc_search")
        self.assertEqual(asset.extra["record_type"], "news_pointer")
        self.assertEqual(asset.extra["curation_level"], "machine_indexed_news_pointer")
        self.assertEqual(asset.extra["source_temporal_coverage"], "event_window")
        self.assertEqual(asset.extra["temporal_relation"], "event_window")
        self.assertEqual(asset.extra["socialimage"], "https://news.example.test/hk.jpg")

    def test_collect_existing_preserves_gdelt_doc_adapter_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            merged = root / "external_assets.jsonl"
            row = gdelt_doc_article_to_asset(
                {
                    "url": "https://news.example.test/ukraine",
                    "title": "Ukraine sovereignty coverage",
                    "seendate": "20220224080000",
                    "domain": "news.example.test",
                    "language": "English",
                    "sourcecountry": "United States",
                },
                "ukraine",
                query="Ukraine sovereignty",
                temporal_relation="event_window",
            )
            (raw / "gdelt_doc_ukraine.jsonl").write_text(json.dumps(row.__dict__, default=list) + "\n", encoding="utf-8")

            summary = collect_existing_external_assets(raw, merged, {"ukraine"})

            rows = [json.loads(line) for line in merged.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["by_source"], {"GDELT_DOC": 1})
            self.assertEqual(rows[0]["extra"]["collection_channel"], "gdelt_doc_search")
            self.assertEqual(rows[0]["extra"]["active_policy"], "pointer_enrichment")

    def test_gdelt_doc_shell_runner_dry_run_sequences_events(self) -> None:
        script = PROJECT_ROOT / "script" / "collect_gdelt_doc_pointers.sh"
        result = subprocess.run(
            [
                str(script),
                "--dry-run",
                "--events",
                "hongkong,ukraine",
                "--max-records",
                "12",
                "--window-days",
                "3",
                "--sleep-seconds",
                "0",
                "--gdelt-retries",
                "4",
                "--gdelt-retry-backoff-seconds",
                "15",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("script/fetch_gdelt_doc_assets.py --event hongkong", result.stdout)
        self.assertIn("script/fetch_gdelt_doc_assets.py --event ukraine", result.stdout)
        self.assertIn("--max-records 12", result.stdout)
        self.assertIn("--gdelt-retries 4", result.stdout)

    def test_gdelt_doc_shell_runner_continues_after_event_failure(self) -> None:
        script = PROJECT_ROOT / "script" / "collect_gdelt_doc_pointers.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_python = root / "fake_python"
            log_path = root / "calls.log"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"$FAKE_PYTHON_LOG\"\n"
                "if [[ \"$*\" == *'--event iraq'* ]]; then exit 9; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            result = subprocess.run(
                [
                    str(script),
                    "--events",
                    "crimea,iraq,libya",
                    "--max-records",
                    "1",
                    "--sleep-seconds",
                    "0",
                    "--output-dir",
                    str(root / "raw"),
                ],
                cwd=PROJECT_ROOT,
                env={"PYTHON_BIN": str(fake_python), "FAKE_PYTHON_LOG": str(log_path)},
                text=True,
                capture_output=True,
                check=False,
            )

            calls = log_path.read_text(encoding="utf-8")
            self.assertEqual(result.returncode, 1)
            self.assertIn("--event crimea", calls)
            self.assertIn("--event iraq", calls)
            self.assertIn("--event libya", calls)
            self.assertIn("GDELT DOC collection failed for events: iraq", result.stderr)


if __name__ == "__main__":
    unittest.main()
