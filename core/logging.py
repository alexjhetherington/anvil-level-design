import time

import bpy


DEBUG_KEEP_HOTSPOT_SEAMS = False

_performance_last_mode = None
_performance_pending_transition = None


def debug_log(msg):
    """Print debug message if logging is enabled via the Debug panel."""
    try:
        enabled = bpy.context.scene.level_design_props.debug_logging
    except AttributeError:
        enabled = False
    if enabled:
        print(msg, flush=True)


def performance_logging_enabled():
    """Return whether performance reports are enabled for the current scene."""
    try:
        return bpy.context.scene.level_design_props.performance_logging
    except AttributeError:
        return False


def reset_performance_mode_tracking(mode):
    """Reset mode-transition tracking without emitting a report."""
    global _performance_last_mode, _performance_pending_transition
    _performance_last_mode = mode
    _performance_pending_transition = None


def observe_performance_mode(mode):
    """Record an Object/Edit mode transition for the next depsgraph report."""
    global _performance_last_mode, _performance_pending_transition

    if mode == _performance_last_mode:
        return

    previous_mode = _performance_last_mode
    _performance_last_mode = mode

    if not performance_logging_enabled():
        _performance_pending_transition = None
        return

    if previous_mode == 'OBJECT' and mode == 'EDIT_MESH':
        event_name = "Entering Edit Mode"
    elif previous_mode == 'EDIT_MESH' and mode == 'OBJECT':
        event_name = "Exiting Edit Mode"
    else:
        _performance_pending_transition = None
        return

    _performance_pending_transition = {
        "event_name": event_name,
        "mode": mode,
        "observed_at": time.perf_counter(),
    }


class _PerformanceModeReport:
    def __init__(self, event_name, observed_at):
        self.event_name = event_name
        self.observed_at = observed_at
        self.handler_started_at = time.perf_counter()
        self.details = []
        self.timings = []

    def add_detail(self, label, value):
        self.details.append((label, str(value)))

    def add_timing(self, label, duration):
        self.timings.append((label, duration))


class _PerformanceOperationReport:
    def __init__(self, event_name, scope):
        self.event_name = event_name
        self.scope = scope
        self.started_at = time.perf_counter()
        self.details = []
        self.timings = []

    def add_detail(self, label, value):
        self.details.append((label, str(value)))

    def add_timing(self, label, duration):
        self.timings.append((label, duration))


def begin_performance_mode_report(mode, transition_update_ready):
    """Consume the pending transition and begin its Anvil timing report."""
    global _performance_pending_transition

    observe_performance_mode(mode)
    if not transition_update_ready:
        return None
    pending = _performance_pending_transition
    if pending is None or pending["mode"] != mode:
        return None

    _performance_pending_transition = None
    return _PerformanceModeReport(
        pending["event_name"],
        pending["observed_at"],
    )


def begin_performance_operation_report(event_name, scope):
    """Begin a standalone operation report when performance logging is enabled."""
    if not performance_logging_enabled():
        return None
    return _PerformanceOperationReport(event_name, scope)


def add_performance_detail(report, label, value):
    """Add context to an active performance report."""
    if report is not None:
        report.add_detail(label, value)


class _DisabledPerformanceStage:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _ActivePerformanceStage:
    def __init__(self, report, label):
        self.report = report
        self.label = label
        self.started_at = 0.0

    def __enter__(self):
        self.started_at = time.perf_counter()
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        self.report.add_timing(
            self.label,
            time.perf_counter() - self.started_at,
        )
        return False


_disabled_performance_stage = _DisabledPerformanceStage()


def performance_stage(report, label):
    """Return a timed or reusable no-op stage context manager."""
    if report is None:
        return _disabled_performance_stage
    return _ActivePerformanceStage(report, label)


def finish_performance_mode_report(report):
    """Print one grouped, low-noise report for a completed mode transition."""
    if report is None:
        return

    finished_at = time.perf_counter()
    handler_duration = finished_at - report.handler_started_at
    observed_duration = finished_at - report.observed_at

    lines = [
        "",
        f"[Performance] === {report.event_name} ===",
        "[Performance] Scope: Anvil Python work after the mode notification; "
        "Blender's native conversion and later viewport drawing are excluded",
    ]
    for label, value in report.details:
        lines.append(f"[Performance] {label}: {value}")

    lines.extend((
        f"[Performance] Anvil depsgraph handler total: {handler_duration * 1000.0:.3f} ms",
        f"[Performance] Mode notification to report complete: {observed_duration * 1000.0:.3f} ms",
        "[Performance] Stage timings:",
    ))

    for label, duration in report.timings:
        percentage = 0.0
        if handler_duration > 0.0:
            percentage = duration / handler_duration * 100.0
        lines.append(
            f"[Performance]   {label}: {duration * 1000.0:.3f} ms ({percentage:.1f}%)"
        )

    timed_duration = sum(duration for _label, duration in report.timings)
    unattributed_duration = max(0.0, handler_duration - timed_duration)
    unattributed_percentage = 0.0
    if handler_duration > 0.0:
        unattributed_percentage = unattributed_duration / handler_duration * 100.0
    lines.append(
        f"[Performance]   Other handler work: {unattributed_duration * 1000.0:.3f} ms "
        f"({unattributed_percentage:.1f}%)"
    )

    lines.append(f"[Performance] === End {report.event_name} ===")
    print("\n".join(lines), flush=True)


def finish_performance_operation_report(report):
    """Print one grouped report for a user-triggered operation."""
    if report is None:
        return

    finished_at = time.perf_counter()
    operation_duration = finished_at - report.started_at
    lines = [
        "",
        f"[Performance] === {report.event_name} ===",
        f"[Performance] Scope: {report.scope}",
    ]
    for label, value in report.details:
        lines.append(f"[Performance] {label}: {value}")

    lines.extend((
        f"[Performance] Operation total: {operation_duration * 1000.0:.3f} ms",
        "[Performance] Stage timings:",
    ))
    for label, duration in report.timings:
        percentage = 0.0
        if operation_duration > 0.0:
            percentage = duration / operation_duration * 100.0
        lines.append(
            f"[Performance]   {label}: {duration * 1000.0:.3f} ms ({percentage:.1f}%)"
        )

    timed_duration = sum(duration for _label, duration in report.timings)
    unattributed_duration = max(0.0, operation_duration - timed_duration)
    unattributed_percentage = 0.0
    if operation_duration > 0.0:
        unattributed_percentage = unattributed_duration / operation_duration * 100.0
    lines.append(
        f"[Performance]   Other operation work: {unattributed_duration * 1000.0:.3f} ms "
        f"({unattributed_percentage:.1f}%)"
    )
    lines.append(f"[Performance] === End {report.event_name} ===")
    print("\n".join(lines), flush=True)
