from __future__ import annotations

from annolid import domain, infrastructure, interfaces, services
from annolid.domain import (
    BehaviorEvent,
    DeepLabCutTrainingImportConfig,
    InstanceRegistry,
    ProjectSchema,
    TimeBudgetRow,
    Track,
)
from annolid.infrastructure import AnnotationStore, configure_ultralytics_cache
from annolid.interfaces.background import TrackingWorker
from annolid.interfaces.bot import ChannelManager
from annolid.interfaces.cli import build_parser, parse_cli
from annolid.interfaces.gui import AnnolidWindow, create_qapp
from annolid.services import (
    build_yolo_dataset_from_index,
    predict_behavior,
    run_agent_pipeline,
    run_behavior_training_cli,
    run_embedding_search,
)


def test_domain_layer_exports() -> None:
    assert domain.ProjectSchema is ProjectSchema
    assert domain.BehaviorEvent is BehaviorEvent
    assert domain.Track is Track
    assert domain.InstanceRegistry is InstanceRegistry
    assert domain.TimeBudgetRow is TimeBudgetRow
    assert domain.DeepLabCutTrainingImportConfig is DeepLabCutTrainingImportConfig


def test_services_layer_exports() -> None:
    assert services.run_agent_pipeline is run_agent_pipeline
    assert services.run_embedding_search is run_embedding_search
    assert services.predict_behavior is predict_behavior
    assert services.run_behavior_training_cli is run_behavior_training_cli
    assert services.build_yolo_dataset_from_index is build_yolo_dataset_from_index


def test_interfaces_layer_exports() -> None:
    assert interfaces.gui.AnnolidWindow is AnnolidWindow
    assert interfaces.gui.create_qapp is create_qapp
    assert interfaces.cli.build_parser is build_parser
    assert interfaces.cli.parse_cli is parse_cli
    assert interfaces.background.TrackingWorker is TrackingWorker
    assert interfaces.bot.ChannelManager is ChannelManager


def test_infrastructure_layer_exports() -> None:
    assert infrastructure.AnnotationStore is AnnotationStore
    assert infrastructure.configure_ultralytics_cache is configure_ultralytics_cache
