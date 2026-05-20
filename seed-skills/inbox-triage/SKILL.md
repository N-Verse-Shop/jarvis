---
name: inbox-triage
description: Швидко проскани inbox Mail / Slack / Linear на termiнові речі — клієнт / партнер / payments.
status: active
version: 1.0.0
author: Danylo Molyanko
tags: [productivity, communication, daily, nexus]
tools: [focus_app, query_calendar]
risk: low
locale: uk
---

# Inbox Triage

Не перечитуй все. Скажи лише те що потребує дії від CEO Nexus Studio
протягом наступних 24 годин.

## Тригер

- «що в інбоксі»
- «що нового»
- «хто чекає відповіді»
- «inbox triage»
- «що сьогодні треба відповісти»

## Categories of "urgent"

Tier 1 — потрібна відповідь сьогодні (≤24h):

- Активний клієнт Nexus (Ibons, Bauplanung-Jungblut, Founder Cockpit…).
- Партнер з підписаним контрактом.
- Payment / invoice issue.
- Сповіщення про падіння проду.

Tier 2 — гарна відповідь до кінця тижня:

- Запит про нову роботу від unknown sender.
- Network LinkedIn / community.

Tier 3 — ігнорувати, але згадати у щоденному recap:

- Newsletters.
- GitHub watch notifications.
- Marketing pitches.

## Протокол

1. `focus_app("Mail")` — переключитися (якщо користувач голосом запитав
   тригер, він хоче побачити).
2. Тригер AppleScript `mail-list-unread` (через run_shortcut) → отримай
   summary: from / subject / preview / received_at.
3. Класифікуй кожен лист по tiers вище.
4. Голос: ≤3 речення про Tier 1, rough count про Tier 2.

## Output

> Tier 1 — два листи: Дмитро з Ibons (payment confirmation) і Jungblut
> щодо документації фасадів. Tier 2 — чотири, переважно потенційні
> ліди. Newsletters ігнорую.

## Edge cases

- Mail.app зачинений — не відкривай нічого автоматично, скажи голосом
  «потрібно відкрити Mail».
- VIP-список (sender = client / partner) бере пріоритет навіть на
  short subject.
- Slack — потенційно у v2; зараз не торкаймось.
