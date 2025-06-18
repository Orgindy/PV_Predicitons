"""Utilities for managing and inspecting dependencies."""

from __future__ import annotations

import logging
from typing import Dict, List

from packaging.requirements import Requirement
from pkg_resources import working_set


class DependencyManager:
    """Helpers for analysing package dependencies."""

    @staticmethod
    def check_version_conflicts() -> List[str]:
        """Check for version conflicts between installed packages."""
        conflicts: List[str] = []
        for dist in working_set:
            for req in dist.requires():
                requirement = Requirement(str(req))
                installed = working_set.by_key.get(requirement.name)
                if installed and installed.version not in requirement.specifier:
                    conflicts.append(f"{dist.project_name} -> {requirement}")
        if conflicts:
            logging.warning("Version conflicts detected: %s", conflicts)
        return conflicts

    @staticmethod
    def build_dependency_graph() -> Dict[str, List[str]]:
        """Return a mapping of package names to their direct dependencies."""
        graph: Dict[str, List[str]] = {}
        for dist in working_set:
            graph[dist.project_name] = [Requirement(str(r)).name for r in dist.requires()]
        return graph
