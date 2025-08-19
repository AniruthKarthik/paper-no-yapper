#!/usr/bin/env python3
import os
import re
import subprocess
import sys

# Your GitHub repository URL
REPO_URL = "https://github.com/AniruthKarthik/paper-no-yapper"

README_PATH = "README.md"
VERSION_PATTERN = r"v(\d+)\.(\d+)\.(\d+)"  # Major.Minor.Patch format


def run_cmd(cmd):
    """Run a shell command and return output or exit on error."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error running command: {cmd}\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def get_last_tag():
    """Return the last git tag."""
    try:
        return run_cmd("git describe --tags --abbrev=0")
    except:
        return None


def parse_commit_message():
    """Get commit message and extract version if present."""
    msg = run_cmd("git log -1 --pretty=%B").strip()
    match = re.search(VERSION_PATTERN, msg)
    if match:
        return msg, match.group(0)
    return msg, None


def update_readme(new_version, old_version):
    """Update README.md with new version and previous version link."""
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Update current version number
    content = re.sub(r"(\*\*Version:\*\*\s*)v\d+\.\d+\.\d+",
                     f"\\1{new_version}", content)

    # Add link to previous version
    if old_version:
        old_link = f"* [{old_version}]({REPO_URL}/releases/tag/{old_version}) — Released before {new_version}"
        if "## 📜 Version History" in content:
            # Add to existing history, just after the header
            content = re.sub(r"(## 📜 Version History\s+)",
                             f"\\1{old_link}\n", content)
        else:
            # Create a new section if not present
            content += f"\n## 📜 Version History\n{old_link}\n"

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    commit_msg, new_version = parse_commit_message()
    if not new_version:
        print("ℹ No version bump detected in commit message. Aborting.")
        sys.exit(0)

    last_tag = get_last_tag()
    print(f"📌 Last tag: {last_tag}, New version: {new_version}")

    # Update README
    update_readme(new_version, last_tag)

    # Stage and amend commit with updated README
    run_cmd(f'git add {README_PATH}')
    run_cmd(f'git commit --amend --no-edit')

    # Create new tag
    run_cmd(f"git tag {new_version}")

    # Push commit and tag
    run_cmd("git push origin HEAD")
    run_cmd(f"git push origin {new_version}")

    print(f"✅ Updated README and tagged {new_version}")


if __name__ == "__main__":
    main()

