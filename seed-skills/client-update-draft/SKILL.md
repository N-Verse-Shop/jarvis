---
name: client-update-draft
description: Згенеруй чернетку короткого weekly-update для активного клієнта у тон Nexus Studio.
status: active
version: 1.0.0
author: Danylo Molyanko
tags: [communication, clients, content, nexus]
tools: [new_note]
risk: low
locale: uk
---

# Client Update Draft

CEO готує weekly update для клієнта. Jarvis робить чернетку з фактів
поточного тижня: що зроблено / що далі / питання до клієнта.

## Тригер

- «зроби update для Ibons»
- «напиши weekly клієнту»
- «чернетка update»
- «status report Bauplanung»

## Tone — Nexus Studio voice

- B2B DACH ринок: чітко, по факту, без води і смайликів.
- Стиль Linear / Stripe: короткі речення, активні дієслова.
- Не використовуй "ми пишемо/розробляємо", використовуй "ми зробили / йде у production".
- Завжди закінчуй питанням клієнту — це драйвить decision velocity.

## Структура

```
Hi <client first name>,

Quick update (week N).

✓ Done:
- bullet 1
- bullet 2
- bullet 3

→ Up next (this week):
- bullet 1
- bullet 2

Questions for you:
- Q1
- Q2

Calendar invite for review: <Friday 14:00 CET if not set>.

— Danylo
```

## Протокол

1. Запитай у користувача (якщо неявно): для якого клієнта?
2. Витягни з Nexus-Brain (`/Users/test/Documents/Nexus-Brain/01-CLIENTS/<client>/`)
   останні daily notes + проєкт MASTER.md.
3. Виділи 3-5 "done" та 2-3 "up next".
4. Сформулюй 1-2 живі questions (не риторичні — реально те що блокує далі).
5. Скинь у Notes через `new_note(title="Update <client> W<n>", body="…")`.

## Output

Голос:

> Чернетку зберіг у Notes "Update Ibons W19". Три виконаних, дві в роботі,
> два питання. Перевір — і я надішлю.

## Edge cases

- Якщо клієнтського MASTER.md нема — попроси імпровізувати з калькуляційного
  Hexpand timeline (Nexus-Brain/02-PROJECTS).
- Якщо нічого не done за тиждень — НЕ вигадуй; напиши чесно
  "Architecture work, no shippable deliverables this week. Recovery
  plan attached."
- Мова листа = мова клієнта (DE для bauplanung-jungblut, UA для українських).

## References

- [`weekly_template`](references/weekly_template.md) — повна Markdown
  заготовка з усіма секціями.
- [`tone_examples`](references/tone_examples.md) — три приклади
  гарного / поганого формулювання done-bullet.
