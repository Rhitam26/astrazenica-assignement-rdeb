"""Optional CLI client sharing the HTTP application's exact runtime."""

import argparse

from src.assistant.service import production_service
from src.shared.config import get_settings


def main():
    parser = argparse.ArgumentParser(description="Chat with the indexed documents")
    parser.add_argument("question", nargs="*")
    args = parser.parse_args()
    with production_service(get_settings()) as service:
        cid = None
        while True:
            try:
                question = " ".join(args.question) or input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            response = service.chat(question, cid)
            cid = response.conversation_id
            print(response.answer)
            for citation in response.citations:
                print(f"[{citation.marker}] {citation.doc_title}: pages {citation.page_numbers}")
            if args.question:
                break


if __name__ == "__main__":
    main()
