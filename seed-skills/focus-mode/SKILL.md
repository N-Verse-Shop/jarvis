---
name: focus-mode
description: Увімкни Do Not Disturb + закрий розсіювачі + почни таймер фокусу (Pomodoro-style).
status: active
version: 1.0.0
author: Danylo Molyanko
tags: [productivity, focus, daily]
tools: [run_shortcut, focus_app]
risk: low
locale: uk
---

# Focus Mode

CEO режим. Закриває розсіювачі, ставить DND, відкриває один проєкт.

## Тригер

- «фокус режим»
- «не турбувати»
- «давай зосередимось»
- «увімкни DND»
- «pomodoro 25»

## Протокол

1. `run_shortcut("Set Focus to Work")` — macOS Focus mode "Work".
2. Закрий розсіювачі: Slack badge → mute, Discord → quit (не quit назавжди,
   лише hide); Telegram → mute 1h.
3. Якщо користувач сказав конкретний проєкт ("фокус на IBONS") —
   `focus_app("Cursor")` + `open_url("file:///path/to/project")`.
4. Якщо явно сказали Pomodoro — `run_shortcut("Pomodoro 25 min")` →
   таймер 25 хв + automation alarm.

## Output

Голос, ≤2 речення:

> Фокус-mode увімкнено. DND до 17:00. IBONS відкрито в Cursor.

При Pomodoro:

> Pomodoro 25 хв пішов. Розбуджу о 17:25.

## Edge cases

- Якщо Focus mode "Work" не існує — створи через Shortcuts один раз,
  потім повтори.
- Якщо вже у фокусі — не подвоюй, скажи «вже у фокусі, продовжуй».
- Користувач сказав "30 хвилин" замість 25 — підстройся, але запам'ятай
  через `add_fact("user.pref.pomodoro_min", "30")`.
