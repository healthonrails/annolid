from typing import final


@final
class MemoryScope:
    GLOBAL = "global"
    WORKSPACE_PREFIX = "workspace:"
    PROJECT_PREFIX = "project:"
    DATASET_PREFIX = "dataset:"
    VIDEO_PREFIX = "video:"
    ANNOTATION_SESSION_PREFIX = "annotation_session:"
    BOT_SESSION_PREFIX = "bot_session:"
    AGENT_PREFIX = "agent:"
    USER_PREFIX = "user:"

    @staticmethod
    def workspace(id_: str) -> str:
        return f"{MemoryScope.WORKSPACE_PREFIX}{id_}"

    @staticmethod
    def project(id_: str) -> str:
        return f"{MemoryScope.PROJECT_PREFIX}{id_}"

    @staticmethod
    def dataset(id_: str) -> str:
        return f"{MemoryScope.DATASET_PREFIX}{id_}"

    @staticmethod
    def video(id_: str) -> str:
        return f"{MemoryScope.VIDEO_PREFIX}{id_}"

    @staticmethod
    def annotation_session(id_: str) -> str:
        return f"{MemoryScope.ANNOTATION_SESSION_PREFIX}{id_}"

    @staticmethod
    def bot_session(id_: str) -> str:
        return f"{MemoryScope.BOT_SESSION_PREFIX}{id_}"

    @staticmethod
    def agent(id_: str) -> str:
        return f"{MemoryScope.AGENT_PREFIX}{id_}"

    @staticmethod
    def user(id_: str) -> str:
        return f"{MemoryScope.USER_PREFIX}{id_}"


@final
class MemoryCategory:
    PREFERENCE = "preference"
    FACT = "fact"
    DECISION = "decision"
    ANNOTATION_RULE = "annotation_rule"
    ANNOTATION_NOTE = "annotation_note"
    PROJECT_SCHEMA = "project_schema"
    PROJECT_NOTE = "project_note"
    SETTING = "setting"
    WORKFLOW_RECIPE = "workflow_recipe"
    TROUBLESHOOTING = "troubleshooting"
    PROMPT_TEMPLATE = "prompt_template"
    ENTITY = "entity"
    OTHER = "other"


@final
class MemorySource:
    USER = "user"
    BOT = "bot"
    SYSTEM = "system"
    ANNOTATION = "annotation"
    PROJECT = "project"
    SETTINGS = "settings"
    WORKFLOW = "workflow"
    IMPORT = "import"
