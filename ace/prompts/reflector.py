"""
Reflector prompts for ACE system.
"""

# Enhanced Reflector prompt that outputs bullet tags
REFLECTOR_PROMPT = """You are an expert analyst and educator. Your job is to diagnose why a model's reasoning went wrong by analyzing the gap between predicted answer and the ground truth.

**Instructions:**
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Take the environment feedback into account, comparing the predicted answer with the ground truth to understand the gap
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- Be specific about what the model should have done differently
- You will receive bulletpoints from the playbook that the generator considered while answering.
- You need to analyze these bulletpoints and give each one a tag from ['helpful', 'harmful', 'neutral'].

Your output should be a json object, which contains the following fields
  - reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
  - error_identification: what specifically went wrong in the reasoning?
  - root_cause_analysis: why did this error occur? What concept was misunderstood?
  - correct_approach: what should the model have done instead?
  - key_insight: what strategy, formula, or principle should be remembered to avoid this error?
  - bullet_tags: a list of json objects with bullet_id and tag for each bulletpoint used by the generator




**Question:**
{}

**Model's Reasoning Trace:**
{}

**Model's Predicted Answer:**
{}

**Ground Truth Answer (regex pattern — e.g. `^39` means the correct answer starts with "39"):**
{}

**Environment Feedback:**
{}

**Part of Playbook that's used by the generator to answer the question:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}

---
"""

REFLECTOR_PROMPT_NO_GT = """You are an expert analyst and educator. Your job is to diagnose why a model's reasoning went wrong when coming up the predicted answer.

**Instructions:**
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Take the environment feedback into account
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- Be specific about what the model should have done differently
- You will receive bulletpoints from the playbook that the generator considered while answering.
- You need to analyze these bulletpoints and give each one a tag from ['helpful', 'harmful', 'neutral'].

Your output should be a json object, which contains the following fields
  - reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
  - error_identification: what specifically went wrong in the reasoning?
  - root_cause_analysis: why did this error occur? What concept was misunderstood?
  - correct_approach: what should the model have done instead?
  - key_insight: what strategy, formula, or principle should be remembered to avoid this error?
  - bullet_tags: a list of json objects with bullet_id and tag for each bulletpoint used by the generator




**Question:**
{}

**Model's Reasoning Trace:**
{}

**Model's Predicted Answer:**
{}

**Environment Feedback:**
{}

**Part of Playbook that's used by the generator to answer the question:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}

---
"""

# --- Russian variants (used when api_provider == "gigachat") ---

REFLECTOR_PROMPT_RU = """Ты — эксперт по анализу ошибок классификации в банковской маршрутизации.

Контекст задачи: модель получает запрос клиента и должна определить нужный документ из фиксированного набора, либо направить на FAQ/оператора/уточнение. Ground truth — regex-паттерн (например, `^39(?:\\.|$)` означает, что правильный ответ начинается с "39"; ответ "3" — это ДРУГОЙ документ, не сокращение от "39").

Модель отвечает без развёрнутого reasoning — только выбранные пункты playbook и итоговый ответ. Твоя задача — по ответу, использованным пунктам и ground truth реконструировать логику ошибки.

Типичные ошибки в этой задаче:
- Путаница между похожими документами (например, разные виды выписок или справок)
- Неверная эскалация: модель выбирает конкретный документ, когда нужен оператор, или наоборот
- Пропуск блокирующих условий, которые должны перенаправить на FAQ
- Остановка поиска на первом частичном совпадении вместо проверки всех вариантов
- Игнорирование контекста запроса (эмоции, нестандартные формулировки)

Инструкции:
- Сравни предсказание модели с ground truth и определи, в чём именно ошибка
- По списку considered/used bullet_ids определи, какие правила модель применила и каких не хватило
- Оцени каждый рассмотренный пункт playbook: помог, навредил или не повлиял
- Предложи конкретное правило, которое предотвратило бы эту ошибку

Твой ответ — JSON-объект:
- reasoning: детальный разбор ошибки
- error_identification: что именно пошло не так
- root_cause_analysis: почему модель ошиблась (какое правило не сработало или отсутствует)
- correct_approach: как модель должна была действовать
- key_insight: какое правило или триггер нужно добавить/усилить в playbook
- bullet_tags: оценка каждого рассмотренного пункта playbook

**Запрос клиента:**
{}

**Ответ модели (bullet_ids + final_answer):**
{}

**Предсказание модели:**
{}

**Ground truth (regex-паттерн):**
{}

**Обратная связь:**
{}

**Пункты playbook, использованные генератором:**
{}

**Ответь строго в JSON:**
{{
  "reasoning": "[Разбор: какие буллеты модель использовала, каких не хватило, почему ответ неверный]",
  "error_identification": "[Конкретная ошибка: перепутала документы X и Y / пропустила блокатор / ложная эскалация]",
  "root_cause_analysis": "[Почему: отсутствует правило для триггера Z / правило слишком общее / конфликт правил]",
  "correct_approach": "[Как надо было: применить правило A, обнаружить блокатор B, выбрать документ C]",
  "key_insight": "[Правило для playbook: при наличии X всегда выбирать Y, а не Z]",
  "bullet_tags": [
    {{"id": "инст-00001", "tag": "helpful"}},
    {{"id": "инст-00002", "tag": "harmful"}}
  ]
}}

---
"""

REFLECTOR_PROMPT_NO_GT_RU = """Ты — эксперт по анализу ошибок классификации в банковской маршрутизации.

Контекст задачи: модель получает запрос клиента и должна определить нужный документ из фиксированного набора, либо направить на FAQ/оператора/уточнение.

Модель отвечает без развёрнутого reasoning — только выбранные пункты playbook и итоговый ответ. Твоя задача — по ответу, использованным пунктам реконструировать логику ошибки.

Типичные ошибки в этой задаче:
- Путаница между похожими документами (например, разные виды выписок или справок)
- Неверная эскалация: модель выбирает конкретный документ, когда нужен оператор, или наоборот
- Пропуск блокирующих условий, которые должны перенаправить на FAQ
- Остановка поиска на первом частичном совпадении вместо проверки всех вариантов
- Игнорирование контекста запроса (эмоции, нестандартные формулировки)

Инструкции:
- По списку considered/used bullet_ids определи, какие правила модель применила и каких не хватило
- Оцени каждый рассмотренный пункт playbook: помог, навредил или не повлиял
- Предложи конкретное правило, которое предотвратило бы эту ошибку

Твой ответ — JSON-объект:
- reasoning: детальный разбор ошибки
- error_identification: что именно пошло не так
- root_cause_analysis: почему модель ошиблась (какое правило не сработало или отсутствует)
- correct_approach: как модель должна была действовать
- key_insight: какое правило или триггер нужно добавить/усилить в playbook
- bullet_tags: оценка каждого рассмотренного пункта playbook

**Запрос клиента:**
{}

**Ответ модели (bullet_ids + final_answer):**
{}

**Предсказание модели:**
{}

**Обратная связь:**
{}

**Пункты playbook, использованные генератором:**
{}

**Ответь строго в JSON:**
{{
  "reasoning": "[Разбор: какие буллеты модель использовала, каких не хватило, почему ответ неверный]",
  "error_identification": "[Конкретная ошибка: перепутала документы X и Y / пропустила блокатор / ложная эскалация]",
  "root_cause_analysis": "[Почему: отсутствует правило для триггера Z / правило слишком общее / конфликт правил]",
  "correct_approach": "[Как надо было: применить правило A, обнаружить блокатор B, выбрать документ C]",
  "key_insight": "[Правило для playbook: при наличии X всегда выбирать Y, а не Z]",
  "bullet_tags": [
    {{"id": "инст-00001", "tag": "helpful"}},
    {{"id": "инст-00002", "tag": "harmful"}}
  ]
}}

---
"""
