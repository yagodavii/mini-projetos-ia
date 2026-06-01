import argparse
import os

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple terminal chatbot with OpenAI.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    history = [
        {"role": "system", "content": "You are a helpful assistant in a CLI chat."}
    ]

    print("Chatbot started. Type 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        history.append({"role": "user", "content": user_input})

        response = client.responses.create(model=args.model, input=history)
        assistant_text = response.output_text.strip()

        print(f"Bot: {assistant_text}")
        history.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
