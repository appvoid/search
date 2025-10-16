from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Optional

import aiohttp
import requests
from bs4 import BeautifulSoup

from .config import Settings
from .types import SearchResult
from .utils import get_title_from_url, truncate_text

GOOGLE_SEARCH_URL = "https://www.google.com/search"


class SearchScraper:
    """Synchronous Google search scraper with optional content extraction."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        }

    def search(self, query: str) -> List[SearchResult]:
        params = {"q": query}
        try:
            response = self.session.get(
                GOOGLE_SEARCH_URL,
                params=params,
                headers=self.headers,
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            if self.settings.debug_mode:
                print(f"> Failed to perform search for '{query}': {exc}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        collections = soup.find_all("div", {"class": "g"})
        results: list[SearchResult] = []
        futures = []

        max_results = self.settings.search_results_per_query
        with ThreadPoolExecutor(max_workers=max(1, max_results)) as executor:
            for group in collections:
                if len(results) >= max_results:
                    break

                anchors = group.find_all("a")
                if not anchors:
                    continue

                link = anchors[0].get("href")
                if not link:
                    continue

                title_node = group.find("h3")
                title = title_node.text.strip() if title_node else get_title_from_url(link)
                description_node = group.find("div", {"data-sncf": "2"})
                description_text = (
                    description_node.text.strip() if description_node else None
                )

                futures.append(
                    executor.submit(
                        self._process_result,
                        title,
                        link,
                        description_text,
                    )
                )

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    if self.settings.debug_mode:
                        print(f"> Error processing result: {exc}")
                    continue

                if result:
                    results.append(result)
                if len(results) >= max_results:
                    break

        return results

    def search_many(self, queries: Iterable[str]) -> List[SearchResult]:
        queries = list(queries)
        if not queries:
            return []

        results: list[SearchResult] = []
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            future_map = {executor.submit(self.search, query): query for query in queries}
            for future in as_completed(future_map):
                try:
                    batch = future.result()
                except Exception as exc:  # noqa: BLE001
                    if self.settings.debug_mode:
                        print(f"> Error searching for '{future_map[future]}': {exc}")
                    continue
                results.extend(batch)
        return results

    def _process_result(
        self, title: str, link: str, description_text: Optional[str]
    ) -> Optional[SearchResult]:
        content = self._extract_content(link)
        if not any([description_text, content]):
            if self.settings.debug_mode:
                print(f"> Skipping '{title}' because no useful content was found")
            return None

        description = truncate_text(description_text, self.settings.max_content_length)
        content = truncate_text(content, self.settings.max_content_length)

        if self.settings.debug_mode and content:
            print(f'Reading > "{title}"')
            print(content)

        return SearchResult(title=title, link=link, description=description, content=content)

    def _extract_content(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(
                url, headers=self.headers, timeout=self.settings.request_timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            if self.settings.debug_mode:
                print(f"> Error fetching content for {url}: {exc}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text(strip=True) for p in paragraphs)
        return content.strip() if content else None


class AsyncSearchScraper:
    """Asynchronous Google search scraper."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        }

    async def search(self, session: aiohttp.ClientSession, query: str) -> List[SearchResult]:
        params = {"q": query}
        try:
            async with session.get(
                GOOGLE_SEARCH_URL,
                params=params,
                headers=self.headers,
                timeout=self.settings.request_timeout,
            ) as response:
                if response.status != 200:
                    if self.settings.debug_mode:
                        print(
                            f"> Failed to perform search for '{query}': HTTP {response.status}"
                        )
                    return []
                html = await response.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if self.settings.debug_mode:
                print(f"> Async search request failed for '{query}': {exc}")
            return []

        soup = BeautifulSoup(html, "html.parser")
        groups = soup.find_all("div", {"class": "g"})
        max_results = self.settings.search_results_per_query
        results: list[SearchResult] = []
        tasks: list[asyncio.Task[Optional[SearchResult]]] = []

        for group in groups:
            if len(tasks) >= max_results:
                break
            anchors = group.find_all("a")
            if not anchors:
                continue

            link = anchors[0].get("href")
            if not link:
                continue

            title_node = group.find("h3")
            title = title_node.text.strip() if title_node else get_title_from_url(link)
            description_node = group.find("div", {"data-sncf": "2"})
            description_text = description_node.text.strip() if description_node else None

            tasks.append(
                asyncio.create_task(
                    self._process_result_async(session, title, link, description_text)
                )
            )

        for task in asyncio.as_completed(tasks):
            result = await task
            if result:
                results.append(result)
            if len(results) >= max_results:
                break

        return results

    async def search_many(
        self, session: aiohttp.ClientSession, queries: Iterable[str]
    ) -> List[SearchResult]:
        queries = list(queries)
        if not queries:
            return []

        tasks = [asyncio.create_task(self.search(session, query)) for query in queries]
        task_to_query = {task: query for task, query in zip(tasks, queries)}
        results: list[SearchResult] = []

        for task in asyncio.as_completed(tasks):
            query = task_to_query.get(task, "<unknown>")
            try:
                batch = await task
            except Exception as exc:  # noqa: BLE001
                if self.settings.debug_mode:
                    print(f"> Error searching for '{query}': {exc}")
                continue
            results.extend(batch)
        return results

    async def _process_result_async(
        self,
        session: aiohttp.ClientSession,
        title: str,
        link: str,
        description_text: Optional[str],
    ) -> Optional[SearchResult]:
        content = await self._extract_content_async(session, link)
        if not any([description_text, content]):
            if self.settings.debug_mode:
                print(f"> Skipping '{title}' because no useful content was found")
            return None

        description = truncate_text(description_text, self.settings.max_content_length)
        content = truncate_text(content, self.settings.max_content_length)

        if self.settings.debug_mode and content:
            print(f'Reading > "{title}"')
            print(content)

        return SearchResult(title=title, link=link, description=description, content=content)

    async def _extract_content_async(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[str]:
        try:
            async with session.get(
                url,
                headers=self.headers,
                timeout=self.settings.request_timeout,
            ) as response:
                if response.status != 200:
                    return None
                html = await response.text()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None

        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text(strip=True) for p in paragraphs)
        return content.strip() if content else None
