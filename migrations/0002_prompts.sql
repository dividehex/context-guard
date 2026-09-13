-- The anomaly window is counted in prompts: a completion that begins a new
-- user prompt advances the counter. LiteLLM sources begin one per completion
-- (so prompt == turn, as before); Claude Code begins one per user message,
-- so an agentic tool loop of many completions is one prompt.
ALTER TABLE conversations ADD COLUMN prompts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE anomalies ADD COLUMN prompt INTEGER NOT NULL DEFAULT 0;
ALTER TABLE health_results ADD COLUMN prompt INTEGER NOT NULL DEFAULT 0;
UPDATE conversations SET prompts = turns;
UPDATE anomalies SET prompt = turn;
UPDATE health_results SET prompt = turn;
CREATE INDEX idx_anomalies_conv_prompt ON anomalies(conversation_id, prompt);
