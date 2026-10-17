
import json
import os
from pathlib import Path

from groq import Groq


INPUT_FILE = Path("text.txt")
OUTPUT_FILE = Path("responses.json")
MODEL_NAME = "openai/gpt-oss-20b"


def load_prompts(file_path):
    """Read prompts from a text file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    prompts = file_path.read_text(encoding="utf-8").splitlines()
    return [prompt.strip() for prompt in prompts if prompt.strip()]


def get_ai_response(client, prompt):
    """Send one prompt to Groq and return the AI response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    """Run the prompt-to-response workflow."""
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing. Set it in Command Prompt first."
        )

    client = Groq()
    prompts = load_prompts(INPUT_FILE)
    results = []

    for prompt in prompts:
        print(f"Sending prompt: {prompt}")

        try:
            answer = get_ai_response(client, prompt)
            results.append({
                "prompt": prompt,
                "response": answer,
            })
        except Exception as error:
            print(f"Request failed: {error}")
            results.append({
                "prompt": prompt,
                "error": str(error),
            })

    OUTPUT_FILE.write_text(
        json.dumps(results, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nFinished. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
