import asyncio
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pyytlounge

from iSponsorBlockTV.main import DeviceListener
from iSponsorBlockTV.ytlounge import YtLoungeApi


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def api(self):
        config = SimpleNamespace(join_name="test", mute_ads=True, skip_ads=True, auto_play=True)
        return YtLoungeApi("screen", config, AsyncMock(), logging.getLogger("test"))

    async def event(self, api, name, data):
        # Exercise our raw event handling independently of typed dependency payload parsing.
        with patch.object(pyytlounge.YtLoungeApi, "_process_event", new=AsyncMock()):
            await api._process_event(name, [data])

    async def test_skip_does_not_unmute_until_ad_ends(self):
        api = self.api()
        api._command = AsyncMock(return_value=True)
        api.skip_ad = AsyncMock(return_value=True)
        await self.event(api, "onAdStateChange", {"adState": "1", "isSkipEnabled": "true"})
        await api._mute_task
        await asyncio.gather(*api._skip_tasks)
        self.assertEqual(api._command.call_args.args[1]["muted"], "true")
        await self.event(api, "onAdStateChange", {"adState": "0", "currentTime": "0"})
        await api._mute_task
        self.assertEqual(api._command.call_args.args[1]["muted"], "false")

    async def test_content_state_does_not_unmute_active_ad(self):
        api = self.api()
        api._command = AsyncMock(return_value=True)
        api._start_ad({})
        await api._mute_task
        await self.event(api, "onStateChange", {"state": "1"})
        self.assertTrue(api._ad_active)
        self.assertEqual(api._command.await_count, 1)

    async def test_failed_mute_does_not_change_cache(self):
        api = self.api()
        api.volume_state = {"volume": "25", "muted": "false"}
        api._command = AsyncMock(return_value=False)
        with self.assertRaises(ConnectionError):
            await api.mute(True, override=True)
        self.assertEqual(api.volume_state["muted"], "false")

    async def test_mute_failure_retries(self):
        api = self.api()
        api._command = AsyncMock(side_effect=[False, True])
        with patch("iSponsorBlockTV.ytlounge.asyncio.sleep", new=AsyncMock()):
            api._start_ad({})
            await api._mute_task
        self.assertEqual(api._command.await_count, 2)
        self.assertEqual(api.volume_state["muted"], "true")

    async def test_ad_end_during_mute_orders_final_unmute(self):
        api = self.api()
        entered, release = asyncio.Event(), asyncio.Event()

        async def command(*args):
            if args[1]["muted"] == "true":
                entered.set()
                await release.wait()
            return True

        api._command = AsyncMock(side_effect=command)
        api._start_ad({})
        await entered.wait()
        api._end_ad()
        release.set()
        await api._mute_task
        self.assertEqual(
            [c.args[1]["muted"] for c in api._command.call_args_list], ["true", "false"]
        )

    async def test_preserves_user_mute(self):
        api = self.api()
        api.volume_state = {"muted": "true", "volume": "30"}
        api._command = AsyncMock()
        api._start_ad({})
        api._end_ad()
        self.assertIsNone(api._mute_task)
        api._command.assert_not_called()

    async def test_tv_unmute_during_ad_is_corrected(self):
        api = self.api()
        api._command = AsyncMock(return_value=True)
        api._start_ad({})
        await api._mute_task
        await self.event(api, "onVolumeChanged", {"muted": "false", "volume": "20"})
        await api._mute_task
        self.assertEqual(api._command.await_count, 2)
        self.assertEqual(api._command.call_args.args[1]["volume"], "20")

    async def test_reset_discards_session_and_cancels_workers(self):
        api = self.api()
        api._sid, api._gsession = "old", "old"
        api.auth.lounge_id_token = "expired"
        api.subscribe_task = asyncio.create_task(asyncio.Event().wait())
        sub = api.subscribe_task
        await api.reset_connection()
        self.assertTrue(sub.cancelled())
        self.assertFalse(api.connected())
        self.assertFalse(api.linked())

    async def test_outage_retries_and_reconnects_after_watchdog_cancel(self):
        listener = DeviceListener.__new__(DeviceListener)
        listener.cancelled, listener.task = False, None
        listener.name, listener.logger = "test", logging.getLogger("test")
        controller = self.api()
        listener.lounge_controller = controller

        async def refresh():
            controller.auth.lounge_id_token = "fresh"
            return True

        attempts = 0

        async def refresh_with_outage():
            nonlocal attempts
            attempts += 1
            if attempts <= 3:
                raise OSError("offline")
            return await refresh()

        controller.refresh_auth = AsyncMock(side_effect=refresh_with_outage)
        controller.is_available = AsyncMock(side_effect=[False, True, True, True])
        controller.connect = AsyncMock(side_effect=[False, True, True])
        subscriptions = 0

        async def subscribe(callback):
            nonlocal subscriptions
            subscriptions += 1
            task = asyncio.create_task(asyncio.Event().wait())
            controller.subscribe_task = task
            if subscriptions == 1:
                task.cancel()  # Simulate watchdog expiring on a silent stream.
            else:
                listener.cancelled = True
                task.cancel()
            return task

        controller.subscribe_monitored = AsyncMock(side_effect=subscribe)
        with patch("iSponsorBlockTV.main.asyncio.sleep", new=AsyncMock()):
            await asyncio.wait_for(listener.loop(), timeout=2)
        self.assertEqual(subscriptions, 2)
        self.assertGreaterEqual(attempts, 7)
        self.assertFalse(controller.connected())

    async def test_watchdog_cancels_silent_subscription(self):
        api = self.api()
        api.subscribe_task = asyncio.create_task(asyncio.Event().wait())
        clock = [0]

        async def tick(_):
            clock[0] += 70
            if clock[0] > 70:
                api.watchdog_running = False

        with patch("iSponsorBlockTV.ytlounge.asyncio.sleep", side_effect=tick):
            loop = asyncio.get_running_loop()
            with patch.object(loop, "time", side_effect=lambda: clock[0]):
                await api._watchdog()
        await asyncio.gather(api.subscribe_task, return_exceptions=True)
        self.assertTrue(api.subscribe_task.cancelled())

    async def test_real_dependency_ad_events(self):
        api = self.api()
        api._command = AsyncMock(return_value=True)
        await api._process_event(
            "onAdStateChange", [{"adState": "1", "currentTime": "0", "isSkipEnabled": "false"}]
        )
        await api._mute_task
        await api._process_event(
            "onAdStateChange", [{"adState": "0", "currentTime": "0", "isSkipEnabled": "false"}]
        )
        await api._mute_task
        self.assertEqual(api._command.call_args.args[1]["muted"], "false")

    async def test_listener_cancellation_is_not_swallowed(self):
        listener = DeviceListener.__new__(DeviceListener)
        listener.cancelled, listener.task = False, None
        listener.name, listener.logger = "test", logging.getLogger("test")
        controller = self.api()
        listener.lounge_controller = controller
        started = asyncio.Event()

        async def refresh():
            started.set()
            await asyncio.Event().wait()

        controller.refresh_auth = refresh
        task = asyncio.create_task(listener.loop())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


if __name__ == "__main__":
    unittest.main()
