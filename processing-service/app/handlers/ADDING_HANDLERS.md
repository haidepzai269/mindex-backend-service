# Adding New Task Handlers

## Steps

1. Create handler file in `app/handlers/`:

```python
from app.handlers.base import BaseHandler, TaskError

class MyHandler(BaseHandler):
    async def process(self, payload: dict) -> dict:
        # payload contains task-specific input
        value = payload.get("key")
        if not value:
            raise TaskError("Missing key in payload")
        # Process and return result dict
        return {"output": value}
```

2. Register in `app/handlers/__init__.py`:

```python
from app.handlers.my_handler import MyHandler

def register_all() -> None:
    # ... existing handlers ...
    registry.register("my_task_type", MyHandler())
```

3. Go backend can immediately use it:

```go
result, err := utils.Processing.CallAsync("my_task_type", payload, timeout)
// or
result, err := utils.Processing.CallSync("my_task_type", payload, timeout)
```

No changes needed to queue consumer, HTTP router, or Go client code.
