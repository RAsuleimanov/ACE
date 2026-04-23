"""
Curator prompts for ACE system.
"""

# Curator prompt for intelligent playbook management
CURATOR_PROMPT = """You are a master curator of knowledge. Your job is to improve an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions.
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used. So you need to come up with content that can aid the playbook user to create predictions that likely align with ground truth.
- Ground truth answers in reflections are regex patterns (e.g. `^39(?:\\.|$)` means the correct answer must start with "39"). Treat them as the authoritative correct label. If the model answered "3" but ground truth is `^39`, that is a REAL error (wrong document), not a formatting mismatch.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Prefer precise lifecycle operations over redundancy
- Use ADD for genuinely new guidance
- Use UPDATE when an existing bullet is partly right but should be rewritten
- Use MERGE when multiple bullets overlap and can be combined into one stronger bullet
- Use ARCHIVE when a bullet is stale, repeatedly neutral, or harmful
- Do NOT regenerate the entire playbook
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable
- Length limit: each ADD/UPDATE `content` must be ≤ 500 characters. Prefer ~300. If guidance is genuinely multi-topic and won't fit, do NOT cram it into one bullet — instead emit ARCHIVE on the old bullet + 2-3 ADD operations (one rule per bullet). Composition > monoliths.
- Don't inflate long bullets: when UPDATE targets a bullet already ≥ 500 characters, the new content must NOT exceed the original length — stay same or shrink. For shorter bullets some growth is acceptable if a genuinely new rule is added, but still respect the 500-char cap.
- No decoration: plain Russian sentences only. Do NOT use ⭐, ⛔, ✅, emoji emphasis, ALL-CAPS headers ("КРИТИЧЕСКОЕ", "АБСОЛЮТНЫЙ", "ОБЯЗАТЕЛЬНОЕ", "ВАЖНОЕ"), or stacked modal particles. Emphasis inflates length without adding signal and pollutes the generator's attention.
- Body only: for ADD/UPDATE/MERGE `content`, return only the bullet body text. Do NOT include bullet IDs, metadata, leading patterns like `[инст-00001] ::`, or full serialized playbook lines.
- Self-contained: write bullets so they stand on their own. Do NOT reference playbook bullet IDs inside `content`; inline the rule instead of writing things like `[инст-00036]`.
- Archive/merge safety: before emitting ARCHIVE or MERGE, ensure no surviving active bullet still depends on the archived/source bullet IDs. If dependencies exist, rewrite those bullets first or skip the ARCHIVE/MERGE.
- No hallucinated IDs: do not invent unseen bullet IDs. Any invented or unseen ID reference will be rejected.


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook
  - type: one of ADD, UPDATE, MERGE, ARCHIVE

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet
2. UPDATE: Rewrite an existing bullet
    - bullet_id: the bullet to rewrite
    - content: the replacement content
3. MERGE: Combine multiple related bullets into a stronger bullet
    - source_ids: the bullets to merge
    - section: the target section for the merged bullet
    - content: the merged bullet content
4. ARCHIVE: Remove a bullet from the active prompt while preserving it for auditability
    - bullet_id: the bullet to archive
    - reason: why the bullet should be archived

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{
      "type": "UPDATE",
      "bullet_id": "calc-00001",
      "content": "[Rewrite the existing calculation guidance...]"
    }},
    {{
      "type": "ARCHIVE",
      "bullet_id": "mis-00054",
      "reason": "repeatedly neutral and stale"
    }}
  ]
}}

---
"""

CURATOR_PROMPT_NO_GT = """You are a master curator of knowledge. Your job is to improve an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions.
- The reflection is generated using environment feedback that will NOT be available when the playbook is being used.
- Ground truth answers in reflections are regex patterns (e.g. `^39(?:\\.|$)` means the correct answer must start with "39"). Treat them as the authoritative correct label. If the model answered "3" but ground truth is `^39`, that is a REAL error (wrong document), not a formatting mismatch.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Prefer precise lifecycle operations over redundancy
- Use ADD for genuinely new guidance
- Use UPDATE when an existing bullet is partly right but should be rewritten
- Use MERGE when multiple bullets overlap and can be combined into one stronger bullet
- Use ARCHIVE when a bullet is stale, repeatedly neutral, or harmful
- Do NOT regenerate the entire playbook
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable
- Length limit: each ADD/UPDATE `content` must be ≤ 500 characters. Prefer ~300. If guidance is genuinely multi-topic and won't fit, do NOT cram it into one bullet — instead emit ARCHIVE on the old bullet + 2-3 ADD operations (one rule per bullet). Composition > monoliths.
- Don't inflate long bullets: when UPDATE targets a bullet already ≥ 500 characters, the new content must NOT exceed the original length — stay same or shrink. For shorter bullets some growth is acceptable if a genuinely new rule is added, but still respect the 500-char cap.
- No decoration: plain Russian sentences only. Do NOT use ⭐, ⛔, ✅, emoji emphasis, ALL-CAPS headers ("КРИТИЧЕСКОЕ", "АБСОЛЮТНЫЙ", "ОБЯЗАТЕЛЬНОЕ", "ВАЖНОЕ"), or stacked modal particles. Emphasis inflates length without adding signal and pollutes the generator's attention.
- Body only: for ADD/UPDATE/MERGE `content`, return only the bullet body text. Do NOT include bullet IDs, metadata, leading patterns like `[инст-00001] ::`, or full serialized playbook lines.
- Self-contained: write bullets so they stand on their own. Do NOT reference playbook bullet IDs inside `content`; inline the rule instead of writing things like `[инст-00036]`.
- Archive/merge safety: before emitting ARCHIVE or MERGE, ensure no surviving active bullet still depends on the archived/source bullet IDs. If dependencies exist, rewrite those bullets first or skip the ARCHIVE/MERGE.
- No hallucinated IDs: do not invent unseen bullet IDs. Any invented or unseen ID reference will be rejected.


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook
  - type: one of ADD, UPDATE, MERGE, ARCHIVE

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet
2. UPDATE: Rewrite an existing bullet
    - bullet_id: the bullet to rewrite
    - content: the replacement content
3. MERGE: Combine multiple related bullets into a stronger bullet
    - source_ids: the bullets to merge
    - section: the target section for the merged bullet
    - content: the merged bullet content
4. ARCHIVE: Remove a bullet from the active prompt while preserving it for auditability
    - bullet_id: the bullet to archive
    - reason: why the bullet should be archived

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{
      "type": "UPDATE",
      "bullet_id": "calc-00001",
      "content": "[Rewrite the existing calculation guidance...]"
    }},
    {{
      "type": "ARCHIVE",
      "bullet_id": "mis-00054",
      "reason": "repeatedly neutral and stale"
    }}
  ]
}}

---
"""

# --- Russian variants (used when api_provider == "gigachat") ---

CURATOR_PROMPT_RU = """Ты — мастер-куратор знаний. Твоя задача — улучшить существующий плейбук на основе рефлексии по предыдущей попытке.

**Контекст:**
- Созданный тобой плейбук будет использоваться для помощи в ответах на аналогичные вопросы.
- Рефлексия создана с использованием правильных ответов (ground truth), которые НЕ будут доступны при использовании плейбука. Поэтому нужно сформулировать контент, который поможет пользователю плейбука давать предсказания, совпадающие с правильным ответом.
- Правильные ответы в рефлексиях — это regex-паттерны (например `^39(?:\\.|$)` означает, что правильный ответ должен начинаться с "39"). Трактуй их как авторитетную правильную метку. Если модель ответила "3", а ground truth — `^39`, это РЕАЛЬНАЯ ошибка (неверный документ), а не расхождение в формате.

**Инструкции:**
- Проанализируй текущий плейбук и рефлексию по предыдущей попытке
- Предпочитай точечные операции жизненного цикла, а не избыточность
- Используй ADD для принципиально нового руководства
- Используй UPDATE, когда существующий пункт частично верен, но нуждается в переписывании
- Используй MERGE, когда несколько пунктов пересекаются и могут быть объединены в один более сильный
- Используй ARCHIVE, когда пункт устарел, стабильно нейтрален или вреден
- НЕ пересоздавай весь плейбук целиком
- Качество важнее количества — сфокусированный, хорошо организованный плейбук лучше исчерпывающего
- Если нечего добавлять — верни пустой список для поля operations
- Будь кратким и конкретным — каждое добавление должно быть практичным
- Лимит длины: каждый content для ADD/UPDATE должен быть <= 500 символов. Предпочтительно ~300. Если руководство многотемное и не помещается, НЕ сжимай в один пункт — лучше ARCHIVE старый + 2-3 ADD (одно правило на пункт). Композиция > монолиты.
- Не раздувай длинные пункты: при UPDATE пункта длиннее 500 символов новый контент НЕ должен превышать длину оригинала — оставь таким же или сократи. Для коротких пунктов допустим рост при добавлении реально нового правила, но с соблюдением лимита 500 символов.
- Без украшательств: только простые русские предложения. НЕ используй эмодзи, заголовки КАПСОМ ("КРИТИЧЕСКОЕ", "АБСОЛЮТНЫЙ", "ОБЯЗАТЕЛЬНОЕ", "ВАЖНОЕ"), нагромождение модальных частиц. Акценты раздувают длину без сигнала и засоряют внимание генератора.
- Только тело: для content в ADD/UPDATE/MERGE возвращай только текст пункта. НЕ включай ID, метаданные, паттерны вида `[инст-00001] ::` или полные сериализованные строки плейбука.
- Самодостаточность: пиши пункты так, чтобы они были понятны без контекста. НЕ ссылайся на ID других пунктов внутри content; встрой правило, а не пиши `[инст-00036]`.
- Безопасность архивирования/слияния: перед ARCHIVE или MERGE убедись, что ни один активный пункт не зависит от архивируемых/исходных ID. Если зависимости есть — сначала перепиши зависимые пункты или пропусти ARCHIVE/MERGE.
- Без выдуманных ID: не изобретай несуществующие ID пунктов. Любая ссылка на несуществующий ID будет отклонена.


**Контекст обучения:**
- Общий бюджет токенов: {token_budget} токенов
- Прогресс обучения: Шаг {current_step} из {total_samples}

**Текущая статистика плейбука:**
{playbook_stats}

**Последняя рефлексия:**
{recent_reflection}

**Текущий плейбук:**
{current_playbook}

**Контекст вопроса:**
{question_context}

**Доступные операции:**
1. ADD — создать новый пункт
2. UPDATE — переписать существующий пункт
3. MERGE — объединить несколько связанных пунктов в один более сильный
4. ARCHIVE — удалить пункт из активного промта с сохранением для аудита

---
"""

CURATOR_PROMPT_NO_GT_RU = """Ты — мастер-куратор знаний. Твоя задача — улучшить существующий плейбук на основе рефлексии по предыдущей попытке.

**Контекст:**
- Созданный тобой плейбук будет использоваться для помощи в ответах на аналогичные вопросы.
- Рефлексия создана с использованием обратной связи среды, которая НЕ будет доступна при использовании плейбука.
- Правильные ответы в рефлексиях — это regex-паттерны (например `^39(?:\\.|$)` означает, что правильный ответ должен начинаться с "39"). Трактуй их как авторитетную правильную метку. Если модель ответила "3", а ground truth — `^39`, это РЕАЛЬНАЯ ошибка (неверный документ), а не расхождение в формате.

**Инструкции:**
- Проанализируй текущий плейбук и рефлексию по предыдущей попытке
- Предпочитай точечные операции жизненного цикла, а не избыточность
- Используй ADD для принципиально нового руководства
- Используй UPDATE, когда существующий пункт частично верен, но нуждается в переписывании
- Используй MERGE, когда несколько пунктов пересекаются и могут быть объединены в один более сильный
- Используй ARCHIVE, когда пункт устарел, стабильно нейтрален или вреден
- НЕ пересоздавай весь плейбук целиком
- Качество важнее количества — сфокусированный, хорошо организованный плейбук лучше исчерпывающего
- Если нечего добавлять — верни пустой список для поля operations
- Будь кратким и конкретным — каждое добавление должно быть практичным
- Лимит длины: каждый content для ADD/UPDATE должен быть <= 500 символов. Предпочтительно ~300. Если руководство многотемное и не помещается, НЕ сжимай в один пункт — лучше ARCHIVE старый + 2-3 ADD (одно правило на пункт). Композиция > монолиты.
- Не раздувай длинные пункты: при UPDATE пункта длиннее 500 символов новый контент НЕ должен превышать длину оригинала — оставь таким же или сократи. Для коротких пунктов допустим рост при добавлении реально нового правила, но с соблюдением лимита 500 символов.
- Без украшательств: только простые русские предложения. НЕ используй эмодзи, заголовки КАПСОМ ("КРИТИЧЕСКОЕ", "АБСОЛЮТНЫЙ", "ОБЯЗАТЕЛЬНОЕ", "ВАЖНОЕ"), нагромождение модальных частиц. Акценты раздувают длину без сигнала и засоряют внимание генератора.
- Только тело: для content в ADD/UPDATE/MERGE возвращай только текст пункта. НЕ включай ID, метаданные, паттерны вида `[инст-00001] ::` или полные сериализованные строки плейбука.
- Самодостаточность: пиши пункты так, чтобы они были понятны без контекста. НЕ ссылайся на ID других пунктов внутри content; встрой правило, а не пиши `[инст-00036]`.
- Безопасность архивирования/слияния: перед ARCHIVE или MERGE убедись, что ни один активный пункт не зависит от архивируемых/исходных ID. Если зависимости есть — сначала перепиши зависимые пункты или пропусти ARCHIVE/MERGE.
- Без выдуманных ID: не изобретай несуществующие ID пунктов. Любая ссылка на несуществующий ID будет отклонена.


**Контекст обучения:**
- Общий бюджет токенов: {token_budget} токенов
- Прогресс обучения: Шаг {current_step} из {total_samples}

**Текущая статистика плейбука:**
{playbook_stats}

**Последняя рефлексия:**
{recent_reflection}

**Текущий плейбук:**
{current_playbook}

**Контекст вопроса:**
{question_context}

**Доступные операции:**
1. ADD — создать новый пункт
2. UPDATE — переписать существующий пункт
3. MERGE — объединить несколько связанных пунктов в один более сильный
4. ARCHIVE — удалить пункт из активного промта с сохранением для аудита

---
"""
