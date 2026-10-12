"""Pure task-DAG scheduling helpers, independent from any model or graph runtime."""

from devma.config import MAX_PARALLEL_TASKS


def ready_tasks(tasks: list[dict], results: dict) -> list[dict]:
    ready = []
    for task in tasks:
        if task["task_id"] in results:
            continue
        if not all(
            results.get(dep, {}).get("status") == "completed"
            for dep in task["depends_on"]
        ):
            continue
        # Serialize overlapping resources if either task can write. Read/read
        # tasks may run in parallel, even when they use the SAME Solver type.
        if any(
            set(task["resource_keys"]) & set(other["resource_keys"])
            and "write" in (task["access"], other["access"])
            for other in ready
        ):
            continue
        ready.append(task)
        if len(ready) >= MAX_PARALLEL_TASKS:
            break
    return ready


def blocked_results(tasks: list[dict], results: dict) -> dict:
    combined, blocked = dict(results), {}
    changed = True
    while changed:
        changed = False
        for task in tasks:
            if task["task_id"] in combined:
                continue
            failed_deps = [
                dep
                for dep in task["depends_on"]
                if dep in combined and combined[dep]["status"] != "completed"
            ]
            if failed_deps:
                result = {
                    "task_id": task["task_id"],
                    "active_solver": task["target_solver"],
                    "status": "blocked",
                    "output": "前置任务尚未完成，暂不执行该步骤。",
                    "blocked_by": failed_deps,
                    "tool_calls": [],
                    "evidence": [],
                    "facts": {},
                    "plan": [],
                    "metrics": {},
                }
                combined[task["task_id"]] = blocked[task["task_id"]] = result
                changed = True
    return blocked


def invalidated_tasks(tasks: list[dict], changed_ids: list[str]) -> set[str]:
    """Invalidate the changed results AND all their downstream consumers."""
    invalid = set(changed_ids)
    for _ in tasks:
        invalid.update(
            t["task_id"] for t in tasks if invalid.intersection(t["depends_on"])
        )
    return invalid
