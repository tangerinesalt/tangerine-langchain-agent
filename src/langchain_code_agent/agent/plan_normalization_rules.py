from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from textwrap import indent

from langchain_code_agent.models.plan import Plan, PlanStep
from langchain_code_agent.models.task import Task


JsonRepairRule = Callable[[str], str]
PlanNormalizationRule = Callable[[list[PlanStep], "PlanNormalizationContext"], list[PlanStep]]


@dataclass(frozen=True, slots=True)
class PlanNormalizationContext:
    task: Task
    workspace_root: Path


def apply_json_text_repairs(text: str) -> str:
    repaired = text
    for rule in JSON_TEXT_REPAIR_RULES:
        repaired = rule(repaired)
    return repaired


def apply_plan_normalization_rules(plan: Plan, *, task: Task, workspace_root: Path) -> Plan:
    context = PlanNormalizationContext(task=task, workspace_root=workspace_root)
    steps = list(plan.steps)
    for rule in PLAN_NORMALIZATION_RULES:
        steps = rule(steps, context)
    return Plan(summary=plan.summary.strip(), steps=steps)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text.strip()
    return text[start : end + 1]


def _escape_invalid_backslashes(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaping = False

    for index, char in enumerate(text):
        if escaping:
            result.append(char)
            escaping = False
            continue

        if char == '"':
            result.append(char)
            in_string = not in_string
            continue

        if char == "\\" and in_string:
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if next_char in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
                result.append(char)
                escaping = True
            else:
                result.append("\\\\")
            continue

        result.append(char)

    return "".join(result)


def _normalize_step_fields(
    steps: list[PlanStep],
    context: PlanNormalizationContext,
) -> list[PlanStep]:
    return [_normalize_step(step, workspace_root=context.workspace_root) for step in steps]


def _dedupe_steps(
    steps: list[PlanStep],
    context: PlanNormalizationContext,
) -> list[PlanStep]:
    del context
    deduped: list[PlanStep] = []
    seen: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    for step in steps:
        key = (
            step.action,
            step.description,
            tuple(sorted((name, repr(value)) for name, value in step.arguments.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(step)
    return deduped


def _collapse_empty_write_file_into_script_output(
    steps: list[PlanStep],
    context: PlanNormalizationContext,
) -> list[PlanStep]:
    collapsed: list[PlanStep] = []
    index = 0
    while index < len(steps):
        current = steps[index]
        next_step = steps[index + 1] if index + 1 < len(steps) else None
        if (
            current.action == "run_python_script"
            and next_step is not None
            and next_step.action == "write_file"
            and str(next_step.arguments.get("content", "")) == ""
            and "path" in next_step.arguments
        ):
            script = str(current.arguments["script"])
            output_path = _normalize_path(
                str(next_step.arguments["path"]),
                context.workspace_root,
            )
            arguments = dict(current.arguments)
            arguments["script"] = _wrap_script_with_output_writer(script, output_path)
            collapsed.append(
                PlanStep(
                    action="run_python_script",
                    description=current.description,
                    arguments=arguments,
                )
            )
            index += 2
            continue

        collapsed.append(current)
        index += 1
    return collapsed


def _inject_time_anchor_step(
    steps: list[PlanStep],
    context: PlanNormalizationContext,
) -> list[PlanStep]:
    if not _is_time_sensitive(context.task.goal):
        return steps
    if any(step.action == "get_current_date" for step in steps):
        return steps
    return [
        PlanStep(
            action="get_current_date",
            description="Get the current local date before time-sensitive work.",
            arguments={},
        ),
        *steps,
    ]


def _normalize_step(step: PlanStep, *, workspace_root: Path) -> PlanStep:
    arguments = dict(step.arguments)
    action = step.action

    if action == "find_files_by_name" and "query" in arguments and "name" not in arguments:
        arguments["name"] = str(arguments.pop("query"))
    if action == "move_file":
        if "src" in arguments and "source_path" not in arguments:
            arguments["source_path"] = str(arguments.pop("src"))
        if "dst" in arguments and "destination_path" not in arguments:
            arguments["destination_path"] = str(arguments.pop("dst"))
    if action in {"read_file", "read_file_head", "write_file", "delete_file"} and "path" in arguments:
        arguments["path"] = _normalize_path(str(arguments["path"]), workspace_root)
    if action == "insert_text" and "path" in arguments:
        arguments["path"] = _normalize_path(str(arguments["path"]), workspace_root)
    if action == "replace_in_file" and "path" in arguments:
        arguments["path"] = _normalize_path(str(arguments["path"]), workspace_root)
    if action == "move_file":
        if "source_path" in arguments:
            arguments["source_path"] = _normalize_path(str(arguments["source_path"]), workspace_root)
        if "destination_path" in arguments:
            arguments["destination_path"] = _normalize_path(
                str(arguments["destination_path"]),
                workspace_root,
            )

    if action == "run_shell" and "command" in arguments:
        script = _extract_python_c_script(str(arguments["command"]))
        if script is not None:
            converted_arguments = {"script": script}
            working_directory = arguments.get("working_directory")
            if working_directory is not None:
                converted_arguments["working_directory"] = str(working_directory)
            return PlanStep(
                action="run_python_script",
                description=step.description,
                arguments=converted_arguments,
            )

    return PlanStep(action=action, description=step.description.strip(), arguments=arguments)


def _extract_python_c_script(command: str) -> str | None:
    stripped = command.strip()
    prefixes = ("python -c ", "python.exe -c ", "py -c ")
    for prefix in prefixes:
        if stripped.startswith(prefix):
            script = stripped[len(prefix) :].strip()
            if len(script) >= 2 and script[0] == script[-1] and script[0] in {'"', "'"}:
                return script[1:-1]
            return script
    return None


def _normalize_path(raw_path: str, workspace_root: Path) -> str:
    normalized = raw_path.strip().replace("\\", "/")
    if not normalized:
        return normalized

    try:
        candidate = Path(normalized)
        if candidate.is_absolute():
            return candidate.resolve().relative_to(workspace_root.resolve()).as_posix()
    except Exception:
        pass

    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if workspace_root.name in parts:
        anchor = parts.index(workspace_root.name)
        remainder = parts[anchor + 1 :]
        if remainder:
            return "/".join(remainder)
    return "/".join(parts)


def _wrap_script_with_output_writer(script: str, output_path: str) -> str:
    stripped_script = script.rstrip()
    wrapped_script = [
        "from contextlib import redirect_stdout",
        "from io import StringIO",
        "from pathlib import Path",
        "",
        "_output_buffer = StringIO()",
        "with redirect_stdout(_output_buffer):",
        indent(stripped_script or "pass", "    "),
        "_output_text = _output_buffer.getvalue()",
        f"Path({output_path!r}).write_text(_output_text, encoding='utf-8')",
    ]
    return "\n".join(wrapped_script) + "\n"


def _is_time_sensitive(task_text: str) -> bool:
    lowered = task_text.lower()
    direct_markers = (
        "today",
        "tomorrow",
        "yesterday",
        "latest",
        "next ",
        "this week",
        "date",
        "days",
        "current date",
        "current time",
    )
    if any(marker in lowered for marker in direct_markers):
        return True
    return "weather" in lowered and any(
        marker in lowered for marker in ("next ", "today", "tomorrow", "this week", "days")
    )


JSON_TEXT_REPAIR_RULES: tuple[JsonRepairRule, ...] = (
    _strip_code_fences,
    _extract_json_object,
    _escape_invalid_backslashes,
)


PLAN_NORMALIZATION_RULES: tuple[PlanNormalizationRule, ...] = (
    _normalize_step_fields,
    _dedupe_steps,
    _collapse_empty_write_file_into_script_output,
    _inject_time_anchor_step,
)
