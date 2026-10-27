import asyncio
import copy
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import ClientResponseError
from pyytlounge.models import State

from iSponsorBlockTV.api_helpers import ApiHelper, merge_iterators
from iSponsorBlockTV.conditional_ttl_cache import AsyncConditionalTTL
from iSponsorBlockTV.main import DeviceListener
from iSponsorBlockTV.ytlounge import PlaybackState, YtLoungeApi


class Response:
    def __init__(self, status, payload):
        self.status, self.payload = status, payload
        self.json = AsyncMock(return_value=payload)
        self.read = AsyncMock(return_value=b"")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status >= 400:
            raise ClientResponseError(None, (), status=self.status)


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def helper(session):
    return ApiHelper(
        SimpleNamespace(
            apikey="test",
            skip_categories=["sponsor"],
            channel_whitelist=[],
            skip_count_tracking=False,
            devices=[],
            minimum_skip_length=0,
            sponsorblock_api_url="https://example.test/api/",
        ),
        session,
    )


class SegmentTests(unittest.TestCase):
    def test_nested_overlapping_adjacent_and_separate_segments(self):
        raw = [
            {"segment": [20, 25], "UUID": "d", "locked": 1},
            {"segment": [2, 3], "UUID": "b", "locked": 1},
            {"segment": [0, 10], "UUID": "a", "locked": 1},
            {"segment": [10.5, 15], "UUID": "c", "locked": 0},
        ]
        original = copy.deepcopy(raw)
        segments, locked = ApiHelper.process_segments({"segments": raw}, 0)
        self.assertEqual(
            segments,
            [
                {"start": 0, "end": 15, "UUID": ["a", "b", "c"]},
                {"start": 20, "end": 25, "UUID": ["d"]},
            ],
        )
        self.assertFalse(locked)
        self.assertEqual(raw, original)

    def test_threshold_applies_to_combined_short_segments(self):
        segments, locked = ApiHelper.process_segments(
            {
                "segments": [
                    {"segment": [0, 0.8], "UUID": "a", "locked": 1},
                    {"segment": [0.9, 1.7], "UUID": "b", "locked": 1},
                ]
            },
            1,
        )
        self.assertEqual(len(segments), 1)
        self.assertTrue(locked)

    def test_one_second_gap_does_not_merge(self):
        result, _ = ApiHelper.process_segments(
            {
                "segments": [
                    {"segment": [0, 2], "UUID": "a", "locked": 1},
                    {"segment": [3, 4], "UUID": "b", "locked": 1},
                ]
            },
            0,
        )
        self.assertEqual(len(result), 2)


class AsyncPerformanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_fetch_survives_one_cancelled_caller(self):
        calls = 0
        started, release = asyncio.Event(), asyncio.Event()

        @AsyncConditionalTTL()
        async def fetch(key):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return [key], False

        first = asyncio.create_task(fetch("video"))
        second = asyncio.create_task(fetch("video"))
        await started.wait()
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        release.set()
        self.assertEqual(await second, ["video"])
        self.assertEqual(await fetch("video"), ["video"])
        self.assertEqual(calls, 1)

    async def test_cache_expiry_and_locked_entries(self):
        calls = 0

        @AsyncConditionalTTL(time_to_live=300)
        async def fetch(locked):
            nonlocal calls
            calls += 1
            return calls, locked

        with patch("iSponsorBlockTV.conditional_ttl_cache.time.monotonic", return_value=100):
            self.assertEqual(await fetch(False), 1)
            self.assertEqual(await fetch(True), 2)
        with patch("iSponsorBlockTV.conditional_ttl_cache.time.monotonic", return_value=401):
            self.assertEqual(await fetch(False), 3)
            self.assertEqual(await fetch(True), 2)

    async def test_transient_errors_are_not_cached(self):
        response = Response(200, [{"videoID": "video", "segments": []}])
        session = Session([Response(503, None), response])
        api = helper(session)
        with self.assertRaises(ClientResponseError):
            await api.get_segments("video")
        self.assertEqual(await api.get_segments("video"), [])
        self.assertEqual(len(session.calls), 2)

    async def test_404_is_cached_with_expiration_without_parsing_json(self):
        response = Response(404, None)
        session = Session([response, Response(200, [])])
        api = helper(session)
        with patch("iSponsorBlockTV.conditional_ttl_cache.time.monotonic", return_value=100):
            self.assertEqual(await api.get_segments("missing"), [])
            self.assertEqual(await api.get_segments("missing"), [])
        response.json.assert_not_awaited()
        with patch("iSponsorBlockTV.conditional_ttl_cache.time.monotonic", return_value=401):
            self.assertEqual(await api.get_segments("missing"), [])
        self.assertEqual(len(session.calls), 2)

    async def test_channel_search_batches_statistics_and_preserves_order(self):
        session = Session(
            [
                Response(
                    200,
                    {
                        "items": [
                            {"snippet": {"channelId": str(i), "channelTitle": str(i)}}
                            for i in range(5)
                        ]
                    },
                ),
                Response(
                    200,
                    {
                        "items": [
                            {"id": str(i), "statistics": {"subscriberCount": "1234"}}
                            for i in reversed(range(5))
                        ]
                    },
                ),
            ]
        )
        result = await helper(session).search_channels("query")
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[1][1]["params"]["id"], "0,1,2,3,4")
        self.assertEqual(result, [(str(i), str(i), "1_234") for i in range(5)])

    async def test_merge_iterator_closes_all_generators_on_early_exit(self):
        stopped = [asyncio.Event(), asyncio.Event()]

        async def source(index):
            try:
                if index:
                    await asyncio.Event().wait()
                yield index
            finally:
                stopped[index].set()

        merged = merge_iterators([source(0), source(1)])
        self.assertEqual(await anext(merged), 0)
        await merged.aclose()
        self.assertTrue(all(event.is_set() for event in stopped))

    async def test_playback_task_uses_snapshot(self):
        listener = DeviceListener.__new__(DeviceListener)
        listener.task = None
        listener.process_playstatus = AsyncMock()
        state = PlaybackState()
        state.videoId = "first"
        await listener(state)
        state.videoId = "second"
        await listener.task
        self.assertEqual(listener.process_playstatus.call_args.args[0].videoId, "first")

    async def test_paused_playback_does_not_fetch_segments(self):
        listener = DeviceListener.__new__(DeviceListener)
        listener.api_helper = AsyncMock()
        await listener.process_playstatus(SimpleNamespace(state=State.Paused, videoId="video"), 0)
        listener.api_helper.get_segments.assert_not_called()

    async def test_skip_timing_subtracts_network_delay_at_double_speed(self):
        listener = DeviceListener.__new__(DeviceListener)
        listener.lounge_controller = SimpleNamespace(playback_speed=2)
        listener.offset = 0.2
        listener.skip = AsyncMock()
        with patch("iSponsorBlockTV.main.time.monotonic", return_value=103):
            await listener.time_to_segment([{"start": 20, "end": 30, "UUID": ["segment"]}], 10, 100)
        self.assertAlmostEqual(listener.skip.call_args.args[0], 1.8)

    async def test_duplicate_prefetch_and_skip_have_one_inflight_task(self):
        config = SimpleNamespace(join_name="test", mute_ads=False, skip_ads=True, auto_play=True)
        api = YtLoungeApi("screen", config, AsyncMock(), logging.getLogger("test"))
        api.skip_ad = AsyncMock(return_value=True)
        for _ in range(20):
            api._prefetch_segments("video")
            api._start_ad({"isSkipEnabled": "true"})
        await asyncio.gather(*api._prefetch_tasks.values(), *api._skip_tasks)
        self.assertEqual(api.api_helper.get_segments.await_count, 1)
        self.assertEqual(api.skip_ad.await_count, 1)
        await api.reset_connection()


class ImportTests(unittest.TestCase):
    def test_service_does_not_import_setup_or_discovery_dependencies(self):
        import os
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import iSponsorBlockTV.helpers; "
                "assert not any(name in sys.modules for name in "
                "['textual', 'pychromecast', 'zeroconf', 'ssdp', 'xmltodict']); "
                "from iSponsorBlockTV import setup_wizard, config_setup, "
                "chromecast_client, dial_client",
            ],
            env=os.environ.copy(),
            check=True,
        )


class DiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_dial_cancellation_joins_inflight_probes(self):
        from unittest.mock import Mock
        from iSponsorBlockTV import dial_client

        started = asyncio.Event()
        probes = []

        async def process(handler, session, api, active, pending, queue, complete, seen):
            probe = asyncio.create_task(asyncio.Event().wait())
            probes.append(probe)
            pending.add(probe)
            started.set()
            await asyncio.Event().wait()

        loop = asyncio.get_running_loop()
        transport = Mock()
        with (
            patch.object(
                loop, "create_datagram_endpoint", new=AsyncMock(return_value=(transport, None))
            ),
            patch.object(dial_client, "get_ip", return_value="127.0.0.1"),
            patch.object(dial_client.network, "get_best_family", return_value=(2, None)),
            patch.object(dial_client, "_send_search_requests", new=AsyncMock()),
            patch.object(dial_client, "_process_devices", side_effect=process),
        ):
            discovery = dial_client.discover(None, None)
            task = asyncio.create_task(anext(discovery))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(probes[0].cancelled())
        transport.close.assert_called_once()
