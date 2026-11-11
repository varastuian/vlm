import asyncio
import os
from pathlib import Path

from browser_use_sdk.v4 import AsyncBrowserUse
PROJECT = Path("/media/varas/Data/Code/paper/vlm")

async def main():
    # Create a key at cloud.browser-use.com/new-api-key
    async with AsyncBrowserUse(api_key=os.environ["BROWSER_USE_API_KEY"]) as client:
        workspace = await client.workspaces.create(name="vlm-project")

        await client.workspaces.upload(
            workspace.id,
            PROJECT / "main3.py",
            # Add other files here:
            # PROJECT / "config.py",
        )
        run = await client.runs.create(
            task="is this file can give me inspiration to find a good way to detect changes in satellite images?",
            model="claude-opus-5-5",
            workspace_id=workspace.id,
            browser_settings={
                "proxy_country_code": "us",
                "record": False,
            },
        )
        result = await client.runs.wait_for_completion(run.id)
        print(result)

asyncio.run(main())