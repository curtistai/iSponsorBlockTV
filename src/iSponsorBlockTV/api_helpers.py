import html
from hashlib import sha256
from asyncio import FIRST_COMPLETED, Task, create_task, gather, wait
from typing import AsyncIterator, Collection, TypeVar

from aiohttp import ClientSession, ClientTimeout
from cache import AsyncLRU
from pyytlounge.wrapper import api_base

from . import constants
from .conditional_ttl_cache import AsyncConditionalTTL


_T = TypeVar("_T")


def list_to_tuple(function):
    def wrapper(*args):
        args = [tuple(x) if isinstance(x, list) else x for x in args]
        result = function(*args)
        result = tuple(result) if isinstance(result, list) else result
        return result

    return wrapper


async def _await_next(iterator: AsyncIterator[_T]) -> _T:
    return await iterator.__anext__()


def _as_task(iterator: AsyncIterator[_T]) -> Task[_T]:
    return create_task(_await_next(iterator))


async def merge_iterators(iterators: Collection[AsyncIterator[_T]]) -> AsyncIterator[_T]:
    pending_tasks = {_as_task(iterator): iterator for iterator in iterators}
    try:
        while pending_tasks:
            done, _ = await wait(pending_tasks, return_when=FIRST_COMPLETED)
            for task in done:
                iterator = pending_tasks.pop(task)
                try:
                    yield task.result()
                except StopAsyncIteration:
                    continue
                except Exception:
                    continue
                pending_tasks[_as_task(iterator)] = iterator
    finally:
        for task in pending_tasks:
            task.cancel()
        await gather(*pending_tasks, return_exceptions=True)
        await gather(
            *(iterator.aclose() for iterator in iterators if hasattr(iterator, "aclose")),
            return_exceptions=True,
        )


# Class that handles all the api calls and their cache
class ApiHelper:
    def __init__(self, config, web_session: ClientSession) -> None:
        self.apikey = config.apikey
        self.skip_categories = config.skip_categories
        self.channel_whitelist = config.channel_whitelist
        self.skip_count_tracking = config.skip_count_tracking
        self.web_session = web_session
        self.num_devices = len(config.devices)
        self.minimum_skip_length = config.minimum_skip_length
        self.sponsorblock_api_url = config.sponsorblock_api_url

    @staticmethod
    def _normalize_pairing_code(pairing_code):
        return str(pairing_code).replace("-", "").replace(" ", "")

    async def pair_with_code(self, pairing_code):
        normalized_code = self._normalize_pairing_code(pairing_code)
        pair_url = f"{api_base}/pairing/get_screen"
        pair_data = {"pairing_code": normalized_code}

        async with self.web_session.post(url=pair_url, data=pair_data) as response:
            if response.status != 200:
                return None
            try:
                pair_response = await response.json()
            except BaseException:
                return None

        screen = pair_response.get("screen")
        if not screen:
            return None

        screen_id = screen.get("screenId")
        if not screen_id:
            return None

        return {
            "screen_id": screen_id,
            "name": screen.get("name"),
            "lounge_token": screen.get("loungeToken"),
        }

    # Not used anymore, maybe it can stay here a little longer
    @AsyncLRU(maxsize=10)
    async def get_vid_id(self, title, artist, api_key, web_session):
        params = {"q": title + " " + artist, "key": api_key, "part": "snippet"}
        url = constants.Youtube_api + "search"
        async with web_session.get(url, params=params) as resp:
            data = await resp.json()

        if "error" in data:
            return

        for i in data["items"]:
            if i["id"]["kind"] != "youtube#video":
                continue
            title_api = html.unescape(i["snippet"]["title"])
            artist_api = html.unescape(i["snippet"]["channelTitle"])
            if title_api == title and artist_api == artist:
                return i["id"]["videoId"], i["snippet"]["channelId"]
        return

    @AsyncLRU(maxsize=100)
    async def is_whitelisted(self, vid_id):
        if self.apikey and self.channel_whitelist:
            channel_id = await self.__get_channel_id(vid_id)
            # check if channel id is in whitelist
            for i in self.channel_whitelist:
                if i["id"] == channel_id:
                    return True
        return False

    async def __get_channel_id(self, vid_id):
        params = {"id": vid_id, "key": self.apikey, "part": "snippet"}
        url = constants.Youtube_api + "videos"
        async with self.web_session.get(url, params=params) as resp:
            data = await resp.json()

        if "error" in data:
            return
        data = data["items"][0]
        if data["kind"] != "youtube#video":
            return
        return data["snippet"]["channelId"]

    @AsyncLRU(maxsize=10)
    async def search_channels(self, channel):
        channels = []
        params = {
            "q": channel,
            "key": self.apikey,
            "part": "snippet",
            "type": "channel",
            "maxResults": "5",
        }
        url = constants.Youtube_api + "search"
        async with self.web_session.get(url, params=params) as resp:
            data = await resp.json()
        if "error" in data:
            return channels

        items = data.get("items", [])
        if not items:
            return channels
        # YouTube accepts multiple channel IDs in one statistics request.
        params = {
            "id": ",".join(i["snippet"]["channelId"] for i in items),
            "key": self.apikey,
            "part": "statistics",
        }
        async with self.web_session.get(constants.Youtube_api + "channels", params=params) as resp:
            channel_data = await resp.json()
        statistics = {i["id"]: i.get("statistics", {}) for i in channel_data.get("items", [])}
        for item in items:
            channel_id = item["snippet"]["channelId"]
            stats = statistics.get(channel_id, {})
            count = stats.get("subscriberCount")
            sub_count = (
                "Hidden"
                if stats.get("hiddenSubscriberCount") or count is None
                else format(int(count), "_")
            )
            channels.append((channel_id, item["snippet"]["channelTitle"], sub_count))
        return channels

    @list_to_tuple  # Convert list to tuple so it can be used as a key in the cache
    @AsyncConditionalTTL(time_to_live=300, maxsize=10)  # 5 minutes for non-locked segments
    async def get_segments(self, vid_id):
        if not self.skip_categories:
            return ([], True)  # Categories explicitly empty, skip segment fetching
        if await self.is_whitelisted(vid_id):
            return (
                [],
                True,
            )  # Return empty list and True to indicate
            # that the cache should last forever
        vid_id_hashed = sha256(vid_id.encode("utf-8")).hexdigest()[
            :4
        ]  # Hashes video id and gets the first 4 characters
        params = {
            "category": self.skip_categories,
            "actionType": constants.SponsorBlock_actiontype,
            "service": constants.SponsorBlock_service,
        }
        headers = {"Accept": "application/json"}
        url = self.sponsorblock_api_url + "skipSegments/" + vid_id_hashed
        async with self.web_session.get(
            url, headers=headers, params=params, timeout=ClientTimeout(total=15)
        ) as response:
            if response.status == 404:
                return [], False  # A video can acquire segments later.
            response.raise_for_status()  # Never cache transient server/rate-limit errors.
            response_json = await response.json()
        for item in response_json:
            if str(item["videoID"]) == str(vid_id):
                return self.process_segments(item, self.minimum_skip_length)
        return [], False

    @staticmethod
    def process_segments(response, minimum_skip_length):
        """Merge intervals in O(n log n), without mutating the API response."""
        raw = response.get("segments", []) if isinstance(response, dict) else []
        ignore_ttl = bool(raw) and all(item.get("locked") == 1 for item in raw)
        segments = []
        for item in sorted(raw, key=lambda item: item["segment"][0]):
            start, end = item["segment"]
            if segments and start - segments[-1]["end"] < 1:
                previous = segments[-1]
                previous["end"] = max(previous["end"], end)
                previous["UUID"].append(item["UUID"])
            else:
                segments.append({"start": start, "end": end, "UUID": [item["UUID"]]})
        return [s for s in segments if s["end"] - s["start"] > minimum_skip_length], ignore_ttl

    async def mark_viewed_segments(self, uuids):
        """Marks the segments as viewed in the SponsorBlock API
        if skip_count_tracking is enabled.
        Lets the contributor know that someone skipped the segment (thanks)"""
        if self.skip_count_tracking:
            for i in uuids:
                url = self.sponsorblock_api_url + "viewedVideoSponsorTime/"
                params = {"UUID": i}
                async with self.web_session.post(url, params=params) as response:
                    await response.read()

    async def discover_youtube_devices_dial(self, active=True):
        """Discovers YouTube devices using DIAL and Chromecast discovery.

        Yields devices as they are discovered instead of waiting for the full
        discovery cycle to finish.
        """
        from . import chromecast_client, dial_client

        seen_screen_ids = set()
        async for device in merge_iterators(
            [
                dial_client.discover(self.web_session, self, active=active),
                chromecast_client.discover(active=active),
            ]
        ):
            screen_id = device.get("screen_id")
            if not screen_id or screen_id in seen_screen_ids:
                continue
            seen_screen_ids.add(screen_id)
            yield device
