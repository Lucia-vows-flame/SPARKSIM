"""Unified weight scheduling and device-transition contracts."""

from MNSIM.Scheduling.weight_scheduling.artifact import (
    ScheduleArtifact,
    ScheduleArtifactError,
    load_schedule_artifact,
    save_schedule_artifact,
)
from MNSIM.Scheduling.weight_scheduling.device import (
    DeviceModel,
    DeviceModelError,
    JsonDeviceModel,
    load_device_model,
    load_device_model_from_sim_config,
    load_native_device_model_from_sim_config,
    register_device_model,
)
from MNSIM.Scheduling.weight_scheduling.executor import execute_schedule_artifact
from MNSIM.Scheduling.weight_scheduling.legacy_migration import (
    LegacyScheduleMigrationError,
    migrate_legacy_schedule,
)
from MNSIM.Scheduling.weight_scheduling.problem import (
    build_problem_from_round_plan,
    canonical_sha256,
)
from MNSIM.Scheduling.weight_scheduling.custom_schedule import (
    CUSTOM_SCHEDULE_FORMAT,
    CUSTOM_SCHEDULE_SCHEMA_VERSION,
    CustomSchedule,
    CustomScheduleError,
    CustomSchedulePolicy,
    CustomScheduleRound,
    generate_weight_schedule_from_custom,
    load_custom_schedule,
    custom_schedule_from_artifact,
    save_custom_schedule,
)
from MNSIM.Scheduling.weight_scheduling.snn_activity import (
    SNNActivityProfileError,
    bind_profile_edges_to_rounds,
    load_snn_activity_profile,
    validate_snn_activity_profile,
)
from MNSIM.Scheduling.weight_scheduling.policies import (
    FlashSeFPTRPolicy,
    SNNFlashSeFPTRPolicy,
    SNNNativeTemporalPolicy,
    SchedulingPolicy,
    SequentialPolicy,
    get_policy,
    register_policy,
)
from MNSIM.Scheduling.weight_scheduling.scheduler import (
    AssignmentDecision,
    LogicalTile,
    ResidentState,
    RoundSchedulingContext,
    SchedulingConstraints,
    SchedulingError,
    UnifiedWeightScheduler,
    WeightScheduleProblem,
    generate_weight_schedule,
)
from MNSIM.Scheduling.weight_scheduling.transition_cost import (
    AggregatedTransitionCost,
    DeviceTransitionCost,
    TransitionCostEngine,
    TransitionCostView,
)

__all__ = [
    "AggregatedTransitionCost",
    "AssignmentDecision",
    "DeviceModel",
    "DeviceModelError",
    "DeviceTransitionCost",
    "FlashSeFPTRPolicy",
    "SNNFlashSeFPTRPolicy",
    "SNNNativeTemporalPolicy",
    "JsonDeviceModel",
    "LegacyScheduleMigrationError",
    "LogicalTile",
    "ResidentState",
    "RoundSchedulingContext",
    "ScheduleArtifact",
    "ScheduleArtifactError",
    "SchedulingConstraints",
    "SchedulingError",
    "SNNActivityProfileError",
    "SchedulingPolicy",
    "SequentialPolicy",
    "TransitionCostEngine",
    "TransitionCostView",
    "UnifiedWeightScheduler",
    "WeightScheduleProblem",
    "generate_weight_schedule",
    "execute_schedule_artifact",
    "build_problem_from_round_plan",
    "canonical_sha256",
    "CUSTOM_SCHEDULE_FORMAT",
    "CUSTOM_SCHEDULE_SCHEMA_VERSION",
    "CustomSchedule",
    "CustomScheduleError",
    "CustomSchedulePolicy",
    "CustomScheduleRound",
    "generate_weight_schedule_from_custom",
    "load_custom_schedule",
    "custom_schedule_from_artifact",
    "save_custom_schedule",
    "get_policy",
    "load_device_model",
    "load_device_model_from_sim_config",
    "load_native_device_model_from_sim_config",
    "load_schedule_artifact",
    "migrate_legacy_schedule",
    "register_device_model",
    "register_policy",
    "save_schedule_artifact",
    "bind_profile_edges_to_rounds",
    "load_snn_activity_profile",
    "validate_snn_activity_profile",
]
