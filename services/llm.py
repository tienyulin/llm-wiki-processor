import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

MINIMAX_API_URL = "https://api.minimax.io/v1/text/chatcompletion_v2"
MINIMAX_MODEL = "MiniMax-M2.7"


class MinimaxClient:
    """Client for the Minimax LLM API."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def extract_json(self, content: str) -> dict:
        """Remove <think>...</think> tags and extract JSON from LLM response."""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    async def _call(self, prompt: str, temperature: float) -> str:
        """Make a single call to the Minimax API and return the assistant message content."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                MINIMAX_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": MINIMAX_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    async def generate_wiki(self, markdowns: dict) -> dict:
        """
        First run: analyze all markdowns and generate a complete wiki JSON.
        """
        combined = "\n\n".join(
            f"## File: {fname}\n{content}" for fname, content in markdowns.items()
        )
        prompt = (
            "Analyze the following API documentation markdown files and generate a structured wiki.\n\n"
            f"{combined}\n\n"
            "Task:\n"
            "1. Extract all API endpoints (method, path, description, parameters)\n"
            "2. Group by module/service\n"
            '3. Generate JSON structure: {"apis": {"module": {"endpoint": {...}}}, "metadata": {}}\n\n'
            "Output ONLY valid JSON, no markdown."
        )
        logger.info(f"Calling Minimax for initial wiki generation ({len(combined)} chars)")
        content = await self._call(prompt, temperature=0.3)
        wiki = self.extract_json(content)
        logger.info("Successfully generated initial wiki")
        return wiki

    async def update_wiki(self, current_wiki: dict, changed_markdowns: dict, changes: dict) -> dict:
        """
        Incremental update: merge changes into existing wiki.
        """
        changed_content = "\n\n".join(
            f"## File: {fname}\n{content}"
            for fname, content in changed_markdowns.items()
        )
        wiki_summary = json.dumps(current_wiki, ensure_ascii=False, indent=2)[:2000]

        prompt = (
            f"Current Wiki (summarized):\n{wiki_summary}\n\n"
            f"Changes: {json.dumps(changes)}\n\n"
            f"New/Modified Markdowns:\n{changed_content}\n\n"
            "Task:\n"
            "1. For new files: Extract APIs and add to wiki\n"
            "2. For modified files: Update related APIs\n"
            "3. For deleted files: Remove from wiki\n"
            "4. Maintain module structure and semantic relationships\n\n"
            "Output ONLY the updated wiki JSON."
        )
        logger.info(f"Calling Minimax for incremental update (changes: {changes})")
        content = await self._call(prompt, temperature=0.2)
        updated_wiki = self.extract_json(content)
        logger.info("Successfully performed incremental wiki update")
        return updated_wiki
