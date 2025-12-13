# Git作業原則

- **mainブランチに直接pushしない**
- 必ず別ブランチ→PR作成
- 例: `git checkout -b feat/xxx` → commit → `git push origin feat/xxx` → `gh pr create`