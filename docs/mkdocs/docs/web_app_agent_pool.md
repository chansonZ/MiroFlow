# Agent Pool (Web Application)

The MiroFlow web application uses an **agent pool** to avoid rebuilding agents
for every submitted query.  Constructing an agent involves initialising the LLM
client, loading tool configurations, and wiring up the skill system — all of
which can take several seconds.  With the pool, agents are created once at
startup and reused across many requests.

---

## How It Works

1. **Warmup** – When the web application starts, a background thread builds
   `AGENT_POOL_SIZE` agents (default **10**) using the default agent config
   (`config/agent_web_demo.yaml`).  The app is immediately available to serve
   requests while the pool fills.

2. **Acquire** – When a task is submitted with the default config, the executor
   takes an idle agent from the pool instead of calling `build_agent_from_config`.

3. **Overflow** – If the pool is empty (all agents are in use), a temporary
   *overflow* agent is created on demand.  This preserves the old "always build"
   behaviour as a safe fallback.

4. **Release** – After a task finishes (success **or** failure), the pool agent
   is returned so future requests can reuse it.  Overflow agents are discarded.

5. **State isolation** – Agent state lives entirely in
   [`AgentContext`](../miroflow/agents/context.py), which is created fresh for
   every task.  The agent object itself is stateless between runs, so reuse is
   safe.

---

## Configuration

All settings can be supplied as **environment variables** (e.g. in `.env`):

| Variable | Default | Description |
|---|---|---|
| `AGENT_POOL_SIZE` | `10` | Number of agents to pre-build at startup. Set to `0` to disable the pool entirely. |
| `AGENT_POOL_MAX_OVERFLOW` | `-1` (unlimited) | Maximum number of extra agents that may be created concurrently when the pool is empty. `-1` means no limit. |
| `AGENT_POOL_STRATEGY` | `overflow_create` | What to do when the pool is empty. Currently only `overflow_create` is supported (creates a temporary agent on demand). |

### Example `.env` snippet

```dotenv
AGENT_POOL_SIZE=5
AGENT_POOL_MAX_OVERFLOW=20
AGENT_POOL_STRATEGY=overflow_create
```

---

## Observability

Pool metrics are logged at startup and during every acquire/release cycle.
You can also inspect pool statistics programmatically:

```python
from web_app.api.dependencies import get_agent_pool

pool = get_agent_pool()
if pool:
    print(pool.stats)
# {
#   "config_path": "config/agent_web_demo.yaml",
#   "pool_size": 10,
#   "available": 8,
#   "overflow_active": 1,
#   "overflow_total": 3,
#   "total_acquired": 42,
#   "total_released": 41,
#   "avg_build_time_s": 4.217
# }
```

Key metrics:

| Metric | Meaning |
|---|---|
| `available` | Idle agents ready for immediate use |
| `overflow_active` | Overflow agents currently executing a task |
| `overflow_total` | Cumulative overflow agents created since startup |
| `avg_build_time_s` | Average time (seconds) to build one agent |

A high `overflow_total` relative to `total_acquired` suggests the pool size
should be increased.

---

## Non-default Configs

The pool is pre-warmed for the **default config** only
(`config/agent_web_demo.yaml` by default, or whatever
`AppConfig.default_config` is set to).  Tasks submitted with a different
agent config are always handled by the overflow path (i.e. an agent is built
on demand).  This keeps the warmup predictable and avoids building agents for
configs that may never be used.
