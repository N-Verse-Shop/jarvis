---
name: git-status-check
description: Швидкий зріз git-стану всіх активних Nexus Studio репо — що uncommitted, що behind, що ahead.
status: active
version: 1.0.0
author: Danylo Molyanko
tags: [git, dev, daily, nexus]
tools: [run_shortcut]
risk: low
locale: uk
---

# Git Status Check (Nexus Studio fleet)

Голосовий зріз стану всіх активних проєктів. Інженеру швидше запитати
Jarvis ніж перемикатися між терміналами.

## Тригер

- «який стан гіту»
- «що там по гіту»
- «git fleet status»
- «що неcommitted»
- «який репо потребує уваги»

## Sources of truth

Active Nexus Studio repos (з `~/.claude/CLAUDE.md`):

- `~/Desktop/Nexus.../Ibons_Finish-main/` — IBONS Hydrogen (Shopify)
- `~/Projects/jarvis-isair/` — Jarvis sandbox
- `~/Projects/nexus-mobile/` — Nexus Mobile (RN + Expo)
- `~/Projects/founder-cockpit/` — Founder Cockpit (Render)

## Протокол

Для кожного репо в списку:

1. `cd <repo>; git status --porcelain` → парс uncommitted (M/A/D/??/UU).
2. `git rev-list --count @{u}..HEAD 2>/dev/null` → ahead count.
3. `git rev-list --count HEAD..@{u} 2>/dev/null` → behind count.
4. Якщо `git stash list` non-empty → відмітити stash count.

Не вимагай мережі. Якщо `@{u}` відсутній — пропусти ahead/behind.

## Output

Голосове резюме (≤4 речення):

> IBONS: 3 uncommitted, 1 ahead. Jarvis: clean, 2 behind. Mobile: clean.
> Founder Cockpit: 1 stash.
> На увагу — IBONS (uncommitted) і Jarvis (need pull).

Якщо все чисто:

> Всі четверо чисті. Push нічого не треба.

## Edge cases

- Submodules — ігноруй status у submodules; рахуй тільки top-level.
- Detached HEAD — згадай голосом окремо.
- Repo missing on disk — мовчки пропусти, не голось «не знайдено».
- Сторонні зміни (M? format) — рахуй у uncommitted.
