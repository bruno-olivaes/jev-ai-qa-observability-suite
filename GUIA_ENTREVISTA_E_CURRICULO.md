# Guia Completo de Entrevista e Defesa Técnica — Requisitos da Vaga de AI QA / Engenheiro de Confiabilidade de IA

Este guia conecta **experiência real já construída** (`Velith`, `olvAI`, `Promobrace`) com a **implementação prática executada neste repositório** ([github.com/bruno-olivaes/jev-ai-qa-observability-suite](https://github.com/bruno-olivaes/jev-ai-qa-observability-suite)) e nos workflows de CI dos repositórios públicos.

---

## 1. Observabilidade de LLMs (Arize Phoenix / OpenTelemetry / OpenInference)
- **Onde está no código:** [`src/observability_phoenix_datadog.py`](src/observability_phoenix_datadog.py) + artifact [`results/phoenix_otel_traces.json`](results/phoenix_otel_traces.json).
- **O que foi implementado:** Instrumentação real via `opentelemetry-sdk` usando semântica **OpenInference** compatível com **Arize Phoenix** (`openinference.span.kind`: `"GUARDRAIL"`, `"LLM"`, `"EVALUATOR"`, `llm.model_name`, `input.value`, `output.value`, `jev.confidence`, `eval.score`, `eval.faithfulness`, `latency_ms`), com suporte opcional a exportação OTLP (`PHOENIX_COLLECTOR_ENDPOINT`) e exportação determinística em disco (**206 spans** capturados na auditoria).
- **Como falar na entrevista:**
  > *"Além de telemetria de latência e portões de qualidade no Velith/olvAI, instrumentei uma pipeline LangGraph + classificador JEV usando OpenTelemetry com atributos OpenInference (padrão do Arize Phoenix), correlacionando `trace_id` e `span_id` entre triagem, geração LLM, avaliação e latência ponta a ponta."*

## 2. Conhecimento de JEV (Sistema 1 para Decisão Estruturada & QA)
- **Onde está no código:** [`src/jev_engine.py`](src/jev_engine.py) e [`src/jev_qa_metrics.py`](src/jev_qa_metrics.py) + relatório [`results/jev_calibration_drift_report.json`](results/jev_calibration_drift_report.json).
- **O que foi implementado:**
  - Contrato estrito **Pydantic v2 (`extra="forbid"`)**: rejeita com HTTP `422` qualquer campo fora do schema — zero geração de texto livre, zero alucinação de JSON.
  - Classificador calibrado por temperatura (`T=1.15`) com 4 classes (`LOW_RISK_AUTO`, `POLICY_CHECK_NEEDED`, `HIGH_RISK_FRAUD`, `PROMPT_INJECTION`) e escalonamento automático para `HUMAN_REVIEW` se `confidence < 0.72`.
  - Bateria completa de **QA para modelos de decisão**:
    - **Matriz de confusão + Precision/Recall/Macro-F1** (120 casos balanceados -> Accuracy `1.0000`, Macro-F1 `1.0000`)
    - **Calibração de confiança:** ECE = `0.0801`, Brier Score = `0.0199`
    - **Detecção de Drift:** **PSI** (`1.5491`, `CRITICAL_DRIFT` detectado entre tráfego limpo vs. turno com excesso de fraude/disputa) e **Jensen-Shannon Divergence** (`0.2317`) no mix de classes
    - **Latência SLA:** In-process P95 = `0.0432 ms`; HTTP end-to-end sob carga k6 = **P95 `12.34 ms`**, **P99 `39.26 ms`**.
- **Como testar um modelo Sistema 1 (pergunta clássica de QA):**
  1. Validar contrato e invariantes (`extra="forbid"`, soma de probabilidades $= 1.0$, tipos enumerados).
  2. Medir **calibração** (ECE e Brier Score — confiança `0.90` precisa acertar ~90% das vezes, senão o limiar de escalonamento humano falha).
  3. Monitorar **drift** contínuo de scores (PSI $> 0.25$) e de distribuição de decisões (JS divergence) antes que a matriz de confusão degrade.
  4. Testar carga com metas rígidas de cauda (**P95 < 500 ms**, **P99 < 800 ms**).

## 3. LangGraph (Fluxos Multi-Nó com Estado & Ciclo Condicional)
- **Onde está no código:** [`src/langgraph_workflow.py`](src/langgraph_workflow.py) e [`tests/test_langgraph_workflow.py`](tests/test_langgraph_workflow.py).
- **O que foi implementado:** `StateGraph(WorkflowState)` real com nós tipados:
  - `jev_triage_node` → roteia prompt injection/fraude direto para `blocked_node` (`BLOCKED_BY_JEV`, sem gastar chamada de LLM!)
  - `generator_node` → recebe `GeneratorCallable` plugável/mockável para testes determinísticos em `pytest`
  - `evaluator_node` → aplica o portão Velith/DeepEval (`>= 7.0/10`)
  - Aresta condicional `route_after_evaluation` → reenvia crítica estruturada (`critique`) ao `generator_node` enquanto `score < 7.0` e `retry_count < max_retries`; aprova (`APPROVED`) quando atinge o corte (testado recuperando de alucinação para `9.33/10` na tentativa 2) ou escala (`ESCALATED_LOW_SCORE`).

## 4. Frameworks de Avaliação (DeepEval / RAGAS)
- **Onde está no código:** [`src/eval_framework.py`](src/eval_framework.py), [`data/golden_eval_cases.json`](data/golden_eval_cases.json) e [`tests/test_deepeval_ragas_suite.py`](tests/test_deepeval_ragas_suite.py).
- **O que foi implementado:** Métricas compatíveis com `deepeval.test_case.LLMTestCase` e `BaseMetric` (`FaithfulnessMetric`, `AnswerRelevancyMetric`, `HallucinationMetric`, `ContextualPrecisionMetric` ao estilo RAGAS) executáveis em CI **sem depender de chave paga da OpenAI** (`0.70s` para toda a suíte). Separa **3/3 golden cases** (score `9.04–9.71/10`) de **3/3 casos adversariais** (alucinação numérica `2.36/10`, inversão de negação `3.00/10` e resposta fora de tópico `0.00/10`).

## 5. Testes de Carga com Grafana k6
- **Onde está no código:** [`load_tests/jev_decision_k6.js`](load_tests/jev_decision_k6.js) e [`results/k6_summary.json`](results/k6_summary.json).
- **Resultado real executado nesta máquina (`k6 v2.4.0` contra FastAPI/Uvicorn):**
  - **20 VUs concorrentes**, **2.232 requisições HTTP**, **202,51 req/s**
  - **`http_req_failed`: `0.00%`** (`0` falhas)
  - **`checks`: `100.00%`** (`8.928 / 8.928` validações de schema JSON e faixa `[0,1]`)
  - **Latência HTTP:** média `4.9 ms`, **P95 = `12.34 ms`** (threshold `< 500 ms`), **P99 = `39.26 ms`** (threshold `< 800 ms`).

## 6. Monitoramento com Datadog (APM, Log Pipelines & Alerts)
- **Onde está no código:** [`monitoring/datadog_monitors.json`](monitoring/datadog_monitors.json), [`monitoring/datadog_log_pipeline.json`](monitoring/datadog_log_pipeline.json) e logs correlacionados em [`results/datadog_structured_logs.jsonl`](results/datadog_structured_logs.jsonl).
- **O que você domina para explicar:**
  - Correlação de logs estruturados JSON com APM via `dd.trace_id` e `dd.span_id` (para abrir qualquer erro de decisão direto no flamegraph da requisição).
  - Pipeline de logs com `date-remapper`, `status-remapper`, `trace-id-remapper` e `category-processor` separando banda de confiança baixa (`@jev.confidence:<0.72`).
  - Três monitores essenciais em produção de IA: (1) P95 APM duration > `500 ms`, (2) spike de `status:ERROR` / HTTP `422`, (3) degradação média de `jev.eval.faithfulness < 0.70`.

## 7. GitHub Actions CI/CD
- **Onde está ativo:** `.github/workflows/ci.yml` criado e configurado em **3 repositórios públicos**:
  1. `jev-ai-qa-observability-suite` (`pytest tests/ -v` + `python run_full_qa_audit.py`)
  2. `qiskit-lombscargle-zne` (`pytest tests/ -v`)
  3. `explosive-sync-markov` (`pytest tests/ -v`)
