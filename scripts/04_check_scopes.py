from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCOPES_DIR = PROJECT_ROOT / "data" / "scopes"


def main():
    scope_files = list(SCOPES_DIR.glob("*_scope.txt"))

    if not scope_files:
        print("No scope files found.")
        return

    for file_path in scope_files:
        text = file_path.read_text(encoding="utf-8")
        print("\n" + "=" * 80)
        print(file_path.name)
        print("=" * 80)
        print(f"Characters: {len(text)}")
        print(f"Words: {len(text.split())}")
        print(text[:500])


if __name__ == "__main__":
    main()