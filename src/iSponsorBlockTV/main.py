import asyncio
from copy import copy
import logging
import time
from signal import SIGINT, SIGTERM, signal
from typing import Optional

import aiohttp

from . import api_helpers, ytlounge
from .debug_helpers import AiohttpTracer


class DeviceListener:
    def __init__(self, api_helper, config, device, debug: bool, web_session):
        self.task: Optional[asyncio.Task] = None
        self.api_helper = api_helper
        self.offset = device.offset
        self.name = device.name
        self.cancelled = False
        self.logger = logging.getLogger(f"iSponsorBlockTV-{device.screen_id}")
        self.web_session = web_session
        self.lounge_controller = ytlounge.YtLoungeApi(
            device.screen_id, config, api_helper, self.logger
        )

    # Each connection attempt obtains a fresh token; no daily refresh task is needed.
    async def loop(self):
        controller = self.lounge_controller
        try:
            while not self.cancelled:
                try:
                    await asyncio.wait_for(controller.refresh_auth(), timeout=30)
                    if not controller.linked():
                        raise ConnectionError("Could not refresh lounge token")
                    if not await asyncio.wait_for(controller.is_available(), timeout=30):
                        self.logger.debug("Waiting for device to be available")
                    elif await asyncio.wait_for(controller.connect(), timeout=30):
                        self.logger.info(
                            "Connected to device %s (%s)", controller.screen_name, self.name
                        )
                        sub = await controller.subscribe_monitored(self)
                        # Shield distinguishes watchdog cancellation from listener shutdown.
                        try:
                            await asyncio.shield(sub)
                        except asyncio.CancelledError:
                            if asyncio.current_task().cancelling():
                                raise
                            self.logger.warning("Subscription stalled; reconnecting")
                except Exception:
                    self.logger.warning("Connection failed; retrying in 10 seconds", exc_info=True)
                finally:
                    await controller.reset_connection()
                    if self.task:
                        self.task.cancel()
                        await asyncio.gather(self.task, return_exceptions=True)
                        self.task = None
                if not self.cancelled:
                    await asyncio.sleep(10)
        finally:
            await controller.reset_connection()

    # Method called on playback state change
    async def __call__(self, state):
        time_start = time.monotonic()
        if self.task:
            self.task.cancel()
        # The lounge reuses its PlaybackState object between events.
        self.task = asyncio.create_task(self.process_playstatus(copy(state), time_start))

    # Processes the playback state change
    async def process_playstatus(self, state, time_start):
        if state.state.value != 1 or not state.videoId:
            return
        try:
            segments = await self.api_helper.get_segments(state.videoId)
            self.logger.debug("Playing video %s with %d segments", state.videoId, len(segments))
            if segments:
                await self.time_to_segment(segments, state.currentTime, time_start)
        except Exception:
            self.logger.warning("Could not process video %s", state.videoId, exc_info=True)

    # Finds the next segment to skip to and skips to it
    async def time_to_segment(self, segments, position, time_start):
        start_next_segment = None
        next_segment = None
        for segment in segments:
            segment_start = segment["start"]
            segment_end = segment["end"]
            is_within_start_range = (
                position < 1 < segment_end and segment_start <= position < segment_end
            )
            is_beyond_current_position = segment_start > position

            if is_within_start_range or is_beyond_current_position:
                next_segment = segment
                start_next_segment = position if is_within_start_range else segment_start
                break
        if start_next_segment is not None:
            time_to_next = (
                (start_next_segment - position) / self.lounge_controller.playback_speed
                - (time.monotonic() - time_start)
                - self.offset
            )
            await self.skip(time_to_next, next_segment["end"], next_segment["UUID"])

    # Skips to the next segment (waits for the time to pass)
    async def skip(self, time_to, position, uuids):
        await asyncio.sleep(time_to)
        self.logger.info("Skipping segment: seeking to %s", position)
        await asyncio.gather(
            asyncio.create_task(self.lounge_controller.seek_to(position)),
            asyncio.create_task(self.api_helper.mark_viewed_segments(uuids)),
        )

    async def cancel(self):
        self.cancelled = True
        await self.lounge_controller.reset_connection()
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def initialize_web_session(self):
        await self.lounge_controller.change_web_session(self.web_session)


async def finish(devices, web_session, tcp_connector):
    await asyncio.gather(*(device.cancel() for device in devices), return_exceptions=True)
    await web_session.close()
    await tcp_connector.close()


def handle_signal(signum, frame):
    raise KeyboardInterrupt()


async def main_async(config, debug, http_tracing):
    loop = asyncio.get_running_loop()
    tasks = []  # Save the tasks so the interpreter doesn't garbage collect them
    devices = []  # Save the devices to close them later
    if debug:
        loop.set_debug(True)

    tcp_connector = aiohttp.TCPConnector(ttl_dns_cache=300)

    # Configure session with tracing if enabled
    if http_tracing:
        root_logger = logging.getLogger("aiohttp_trace")
        tracer = AiohttpTracer(root_logger)
        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(tracer.on_request_start)
        trace_config.on_response_chunk_received.append(tracer.on_response_chunk_received)
        trace_config.on_request_end.append(tracer.on_request_end)
        trace_config.on_request_exception.append(tracer.on_request_exception)
        web_session = aiohttp.ClientSession(
            trust_env=config.use_proxy, connector=tcp_connector, trace_configs=[trace_config]
        )
    else:
        web_session = aiohttp.ClientSession(trust_env=config.use_proxy, connector=tcp_connector)

    api_helper = api_helpers.ApiHelper(config, web_session)
    for i in config.devices:
        device = DeviceListener(api_helper, config, i, debug, web_session)
        devices.append(device)
        await device.initialize_web_session()
        tasks.append(loop.create_task(device.loop()))
    signal(SIGTERM, handle_signal)
    signal(SIGINT, handle_signal)
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("Cancelling tasks and exiting...")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await finish(devices, web_session, tcp_connector)
        print("Exited")


def main(config, debug, http_tracing):
    asyncio.run(main_async(config, debug, http_tracing))
