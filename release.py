#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from datetime import datetime

REPO_URL = "https://github.com/AniruthKarthik/paper-no-yapper"

README_PATH = "README.md"
VERSION_PATTERN = r"v(\d+)\.(\d+)\.(\d+)"  # Major.Minor.Patch format

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()

def get_last_tag():
    try:
        return run_cmd("git describe --tags --abbrev=0")
    except:
        return None

def parse_commit_message():
    msg = run_cmd("git log -1 --pretty=%B").strip()
    match = re.search(VERSION_PATTERN, msg)
    if match:
        return msg, match.group(0)
    return msg, None

def update_readme(new_version, old_version):
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Update current version
    content = re.sub(r"(## Current Version:\s*)v\d+\.\d+\.\d+",
                     f"\\1{new_version}", content)

    # Add link to previous version
    if old_version:
        old_link = f"- [{old_version}]({REPO_URL}/tree/{old_version}) — Released before {new_version}"
        if "## Previous Versions" in content:
            content = re.sub(r"(## Previous Versions\s+)",
                             f"\\1{old_link}\n", content)
        else:
            content += f"\n## Previous Versions\n{old_link}\n"

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    commit_msg, new_version = parse_commit_message()
    if not new_version:
        print("No version bump detected in commit message. Aborting.")
        sys.exit(0)

    last_tag = get_last_tag()
    print(f"Last tag: {last_tag}, New version: {new_version}")

    # Update README
    update_readme(new_version, last_tag)

    # Commit README changes
    run_cmd(f'git add {README_PATH}')
    run_cmd(f'git commit --amend --no-edit')

    # Tag the new version
    run_cmd(f"git tag {new_version}")

    # Push commit and tag
    run_cmd("git push origin HEAD")
    run_cmd(f"git push origin {new_version}")

    print(f"✅ Updated README, committed, and tagged {new_version}")

if __name__ == "__main__":
    main()

