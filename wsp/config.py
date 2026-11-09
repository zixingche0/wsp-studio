"""配置与路径管理。

所有运行期数据集中在 ``WSP_HOME``（默认 ``<项目>/../wsp-data``），
与代码目录分离：

    WSP_HOME/
      models/            托管的 safetensors 模型
      workdir/
        models/<model_id>/revisions/<revision>/   编辑后的 checkpoint
        edits/<edit_id>/                          位级 diff
      wsp.db             SQLite

原始模型目录**永远只读**。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_home() -> Path:
    """数据目录，按优先级：

    1. ``WSP_HOME`` 环境变量（显式指定，最高优先级）
    2. ``WSP_DATA_DIR``
    3. 项目源码旁边的 ``wsp-data``（便携模式：从源码运行时）
    4. 各平台惯例目录（从 pip 安装运行时的兜底）
    """
    for env in ("WSP_HOME", "WSP_DATA_DIR"):
        value = os.environ.get(env)
        if value:
            return Path(value).expanduser().resolve()

    # 便携模式：如果这个包是从源码目录运行的，就把数据放在它旁边
    source_root = Path(__file__).resolve().parent.parent
    if (source_root / "pyproject.toml").is_file():
        sibling = source_root.parent / "wsp-data"
        if sibling.is_dir() or os.access(source_root.parent, os.W_OK):
            return sibling.resolve()

    # 打包安装：用平台惯例目录
    from . import platform as plat

    return plat.data_dir()


@dataclass(frozen=True)
class Paths:
    home: Path

    def __post_init__(self) -> None:
        # 允许传 str，统一转成 Path
        object.__setattr__(self, "home", Path(self.home).expanduser().resolve())

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def workdir(self) -> Path:
        return self.home / "workdir"

    @property
    def revisions(self) -> Path:
        return self.workdir / "models"

    @property
    def edits(self) -> Path:
        return self.workdir / "edits"

    @property
    def db_path(self) -> Path:
        return self.home / "wsp.db"

    @property
    def registry_path(self) -> Path:
        return self.home / "models" / "registry.json"

    def ensure(self) -> "Paths":
        for p in (self.home, self.models, self.workdir, self.revisions, self.edits):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    return Paths(_default_home()).ensure()


# 安全：原始模型目录必须只读。这里做一次显式检查。
def assert_outside(original: str | os.PathLike, target: str | os.PathLike) -> None:
    """确保写入目标不是原始模型目录（或其子目录）。"""
    orig = Path(original).resolve()
    tgt = Path(target).resolve()
    if tgt == orig or orig in tgt.parents:
        raise PermissionError(
            f"拒绝写入原始模型目录: {tgt} 位于只读的 {orig} 之内"
        )
