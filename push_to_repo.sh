#!/bin/bash
# Push the whole project to the repository.
# Run from project root:  bash push_to_repo.sh

set -e
cd "/Users/ivantarkhanov/Desktop/Python Practice/Projects/Inflation allocation"

# Stage all tracked and new files (respects .gitignore)
git add -A

# Show what will be committed
git status

# Commit with a descriptive message
git commit -m "Align README with canonical backtest; 60/40 benchmark fix; regime metrics and transparency"

# Push to remote (default branch; use origin main or origin master as needed)
git push
