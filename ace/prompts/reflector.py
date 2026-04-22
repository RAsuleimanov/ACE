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

**Ground Truth Answer:**
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

REFLECTOR_PROMPT_RU = """Ты — эксперт-аналитик и преподаватель. Твоя задача — диагностировать, почему рассуждение модели пошло неверно, анализируя разрыв между предсказанным ответом и правильным (ground truth).

**Инструкции:**
- Тщательно проанализируй цепочку рассуждений модели, чтобы определить, где произошла ошибка
- Учитывай обратную связь среды, сравнивая предсказанный ответ с правильным
- Определи конкретные концептуальные ошибки, ошибки в расчётах или неправильно применённые стратегии
- Дай практичные рекомендации, которые помогут модели избежать этой ошибки в будущем
- Сосредоточься на корневой причине, а не на поверхностных проявлениях
- Укажи конкретно, что модель должна была сделать иначе
- Ты получишь пункты плейбука, которые генератор рассматривал при ответе
- Присвой каждому пункту тег из ['helpful', 'harmful', 'neutral']

Твой ответ — JSON-объект со следующими полями:
  - reasoning: цепочка рассуждений, детальный анализ
  - error_identification: что конкретно пошло не так в рассуждении?
  - root_cause_analysis: почему произошла эта ошибка? Какая концепция была неверно понята?
  - correct_approach: что модель должна была сделать иначе?
  - key_insight: какую стратегию или принцип нужно запомнить, чтобы избежать этой ошибки?
  - bullet_tags: список JSON-объектов с bullet_id и tag для каждого рассмотренного пункта




**Вопрос:**
{}

**Цепочка рассуждений модели:**
{}

**Предсказанный ответ модели:**
{}

**Правильный ответ (Ground Truth, regex-паттерн — например '^39' значит правильный ответ начинается с "39"):**
{}

**Обратная связь среды:**
{}

**Пункты плейбука, использованные генератором:**
{}

**Ответь строго в таком JSON-формате:**
{{
  "reasoning": "[Цепочка рассуждений, детальный анализ]",
  "error_identification": "[Что конкретно пошло не так?]",
  "root_cause_analysis": "[Почему произошла ошибка?]",
  "correct_approach": "[Что нужно было сделать иначе?]",
  "key_insight": "[Какой принцип запомнить?]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}

---
"""

REFLECTOR_PROMPT_NO_GT_RU = """Ты — эксперт-аналитик и преподаватель. Твоя задача — диагностировать, почему рассуждение модели пошло неверно при формулировании предсказанного ответа.

**Инструкции:**
- Тщательно проанализируй цепочку рассуждений модели, чтобы определить, где произошла ошибка
- Учитывай обратную связь среды
- Определи конкретные концептуальные ошибки, ошибки в расчётах или неправильно применённые стратегии
- Дай практичные рекомендации, которые помогут модели избежать этой ошибки в будущем
- Сосредоточься на корневой причине, а не на поверхностных проявлениях
- Укажи конкретно, что модель должна была сделать иначе
- Ты получишь пункты плейбука, которые генератор рассматривал при ответе
- Присвой каждому пункту тег из ['helpful', 'harmful', 'neutral']

Твой ответ — JSON-объект со следующими полями:
  - reasoning: цепочка рассуждений, детальный анализ
  - error_identification: что конкретно пошло не так в рассуждении?
  - root_cause_analysis: почему произошла эта ошибка?
  - correct_approach: что модель должна была сделать иначе?
  - key_insight: какую стратегию или принцип нужно запомнить?
  - bullet_tags: список JSON-объектов с bullet_id и tag для каждого рассмотренного пункта




**Вопрос:**
{}

**Цепочка рассуждений модели:**
{}

**Предсказанный ответ модели:**
{}

**Обратная связь среды:**
{}

**Пункты плейбука, использованные генератором:**
{}

**Ответь строго в таком JSON-формате:**
{{
  "reasoning": "[Цепочка рассуждений, детальный анализ]",
  "error_identification": "[Что конкретно пошло не так?]",
  "root_cause_analysis": "[Почему произошла ошибка?]",
  "correct_approach": "[Что нужно было сделать иначе?]",
  "key_insight": "[Какой принцип запомнить?]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}

---
"""
