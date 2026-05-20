---
name: deploy-check
description: Перевір статус деплоїв Render / Cloudflare / Hetzner для активних проєктів Nexus.
status: active
version: 1.0.0
author: Danylo Molyanko
tags: [devops, deploy, monitoring, nexus]
tools: [run_shortcut, query_calendar]
risk: low
locale: uk
---

# Deploy Check (Nexus Studio infra)

Швидке голосове perevirka де що зараз задеплоєно і нічого не зламано.

## Тригер

- «що з деплойами»
- «що в продакшні»
- «який стан інфри»
- «cloudflare статус»
- «hetzner alive»

## Endpoints to ping (HTTP HEAD only — fail-open if creds missing)

| Service          | URL                                           | Expected |
|------------------|-----------------------------------------------|----------|
| Ibons production | https://ibons.de                              | 200      |
| Founder Cockpit  | https://founder-cockpit.onrender.com/health   | 200      |
| Jarvis Hetzner   | https://ollama.jarvis-bridge.tail.../health   | 200      |
| Tailscale tunnel | tailscale status (local CLI)                  | online   |

## Протокол

1. Виконай HEAD на кожен URL з timeout=3s (через `curl -sI`).
2. Парс HTTP code + response time.
3. Якщо `tailscale` CLI присутня — `tailscale status --json` для Hetzner check.
4. Не падай якщо щось недосяжне — рахуй це окремою катеорією.

## Output

Швидко, ≤3 речення:

> Всі чотири живі. Найповільніше — Cockpit 380ms. Тейлскейл online.

При проблемі:

> Увага: Ibons production падає 502 уже п'ять секунд. Cockpit і Jarvis OK.

## Edge cases

- Без інтернету — мовчки «оффлайн локально».
- 401/403 на health → НЕ помилка деплою (auth gate), згадай окремо.
- Timeout → клас "degraded", не "down" — пропонуй повторну перевірку через 30s.
