import asyncio
import os
import time
from pathlib import Path

from browser_use_sdk.v4 import AsyncBrowserUse

PROJECT = Path("/media/varas/Data/Code/paper/vlm")

async def main():
    async with AsyncBrowserUse(api_key=os.environ["BROWSER_USE_API_KEY"]) as client:
        workspace = await client.workspaces.create(name="vlm-project")

        await client.workspaces.upload(
            workspace.id,
            PROJECT / "main3.py",
        )

        # 1. Better task: Ask it to fix the code and save a new file.
        # 2. Add a judge to get a clean, readable verdict.
        run = await client.runs.create(
            task=(


                
                "Review the uploaded file 'main3.py'. Identify all bugs and missing functionality. "
                "Then, write a complete, corrected version of the script. "
                "Save the fixed file to the workspace as 'main3_fixed.py'. "
                "Ensure all issues mentioned in the code are addressed."
            ),
            model="claude-opus-5-5",
            workspace_id=workspace.id,
            judge={"context": "The agent must fix all bugs and save a corrected file named 'main3_fixed.py'."},
            browser_settings={
                "proxy_country_code": "us",
                "record": False,
            },
        )

        print(f"Run ID: {run.id}")
        print("Waiting for completion...")
        
        result = await client.runs.wait_for_completion(run.id)

        # Print the judge verdict for a clean assessment
        if result.judgement:
            print("\n--- JUDGE VERDICT ---")
            print(f"Successful: {result.judgement.get('verdict')}")
            print(f"Reasoning: {result.judgement.get('reasoning', 'N/A')}")
        else:
            print("\n--- RAW RESULT ---")
            print(result.result)

        # 3. Retrieve the fixed file from the workspace
        print("\n--- WORKSPACE FILES ---")
        files = await client.workspaces.files(workspace.id, include_urls=True)
        for f in files.files:
            print(f"File: {f.path}")
            if "fixed" in f.path and f.url:
                print(f"  Download URL: {f.url}")

if __name__ == "__main__":
    asyncio.run(main())