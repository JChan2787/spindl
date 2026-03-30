#!/usr/bin/env python3
"""
Migration script: YAML persona files → ST V2 Character Cards.

Converts legacy persona YAML files to the new Character Card V2 JSON format.

Usage:
    python scripts/migrate_persona.py spindle
    python scripts/migrate_persona.py --all
    python scripts/migrate_persona.py spindle --dry-run

This script:
1. Reads persona YAML from personas/{id}.yaml
2. Maps fields to ST V2 Character Card format
3. Writes JSON to characters/{id}/card.json
4. Optionally backs up original YAML
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml


def load_yaml_persona(persona_path: Path) -> dict:
    """Load persona from YAML file."""
    with open(persona_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def migrate_persona_to_card(persona: dict) -> dict:
    """
    Convert legacy persona dict to ST V2 Character Card format.

    Mapping:
        YAML Field              → ST V2 Location
        ─────────────────────────────────────────
        id                      → extensions.spindl.id
        name                    → data.name
        voice                   → extensions.spindl.voice
        language                → extensions.spindl.language
        appearance              → extensions.spindl.appearance
        personality             → data.personality
        rules                   → extensions.spindl.rules
        system_prompt           → data.system_prompt
        summarization_prompt    → extensions.spindl.summarization_prompt
        generation              → extensions.spindl.generation
        summarization_generation→ extensions.spindl.summarization_generation
    """
    # Build spindl extensions block
    nano_ext = {}

    if "id" in persona:
        nano_ext["id"] = persona["id"]
    if "voice" in persona:
        nano_ext["voice"] = persona["voice"]
    if "language" in persona:
        nano_ext["language"] = persona["language"]
    if "appearance" in persona:
        nano_ext["appearance"] = persona["appearance"]
    if "rules" in persona:
        nano_ext["rules"] = persona["rules"]
    if "summarization_prompt" in persona:
        nano_ext["summarization_prompt"] = persona["summarization_prompt"]
    if "generation" in persona:
        nano_ext["generation"] = persona["generation"]
    if "summarization_generation" in persona:
        nano_ext["summarization_generation"] = persona["summarization_generation"]

    # Build extensions block
    extensions = {}
    if nano_ext:
        extensions["spindl"] = nano_ext

    # Build the Character Card V2 structure
    card = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            # V1 Legacy fields (required for compatibility)
            "name": persona.get("name", "Unknown"),
            "description": "",
            "personality": persona.get("personality", ""),
            "scenario": "",
            "first_mes": "",
            "mes_example": "",
            # V2 New fields
            "creator_notes": "",
            "system_prompt": persona.get("system_prompt", ""),
            "post_history_instructions": "",
            "alternate_greetings": [],
            "tags": [],
            "creator": "spindl migration",
            "character_version": "1.0",
            "extensions": extensions,
        },
    }

    return card


def migrate_persona(
    persona_id: str,
    personas_dir: Path,
    characters_dir: Path,
    dry_run: bool = False,
    backup: bool = True,
) -> bool:
    """
    Migrate a single persona from YAML to Character Card V2.

    Returns True if migration was successful.
    """
    persona_path = personas_dir / f"{persona_id}.yaml"

    if not persona_path.exists():
        print(f"ERROR: Persona file not found: {persona_path}")
        return False

    # Load YAML persona
    try:
        persona = load_yaml_persona(persona_path)
    except Exception as e:
        print(f"ERROR: Failed to parse {persona_path}: {e}")
        return False

    # Convert to Character Card
    card = migrate_persona_to_card(persona)

    # Output path
    char_dir = characters_dir / persona_id
    card_path = char_dir / "card.json"

    if dry_run:
        print(f"\n[DRY RUN] Would migrate: {persona_path}")
        print(f"[DRY RUN] Would create: {card_path}")
        print(f"\n[DRY RUN] Card content:")
        print(json.dumps(card, indent=2))
        return True

    # Create character directory
    char_dir.mkdir(parents=True, exist_ok=True)

    # Write Character Card JSON
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)

    print(f"[OK] Migrated: {persona_path} -> {card_path}")

    # Backup original YAML
    if backup:
        backup_path = persona_path.with_suffix(".yaml.bak")
        shutil.copy2(persona_path, backup_path)
        print(f"  Backup: {backup_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate YAML personas to ST V2 Character Cards"
    )
    parser.add_argument(
        "persona_id",
        nargs="?",
        help="Persona ID to migrate (filename without .yaml)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrate all personas in the personas directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without writing files",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of original YAML files",
    )
    parser.add_argument(
        "--personas-dir",
        type=Path,
        default=Path("./personas"),
        help="Path to personas directory (default: ./personas)",
    )
    parser.add_argument(
        "--characters-dir",
        type=Path,
        default=Path("./characters"),
        help="Path to characters output directory (default: ./characters)",
    )

    args = parser.parse_args()

    if not args.persona_id and not args.all:
        parser.error("Either persona_id or --all is required")

    personas_dir = args.personas_dir.resolve()
    characters_dir = args.characters_dir.resolve()

    if not personas_dir.exists():
        print(f"ERROR: Personas directory not found: {personas_dir}")
        sys.exit(1)

    # Collect personas to migrate
    if args.all:
        persona_ids = [
            p.stem for p in personas_dir.glob("*.yaml") if p.is_file()
        ]
        if not persona_ids:
            print(f"No persona files found in {personas_dir}")
            sys.exit(0)
    else:
        persona_ids = [args.persona_id]

    print(f"Personas directory: {personas_dir}")
    print(f"Characters directory: {characters_dir}")
    print(f"Personas to migrate: {len(persona_ids)}")
    if args.dry_run:
        print("Mode: DRY RUN (no files will be written)")
    print()

    # Migrate each persona
    success_count = 0
    for persona_id in persona_ids:
        if migrate_persona(
            persona_id,
            personas_dir,
            characters_dir,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        ):
            success_count += 1

    print()
    print(f"Migration complete: {success_count}/{len(persona_ids)} successful")

    if success_count < len(persona_ids):
        sys.exit(1)


if __name__ == "__main__":
    main()
