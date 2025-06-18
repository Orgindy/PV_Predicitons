from __future__ import annotations
import logging
from typing import Dict, List


class DependencyManager:
    """Utility helpers for managing package dependencies."""

    @staticmethod
    def check_version_conflicts() -> Dict[str, str]:
        """Check for version conflicts between installed packages."""
        try:
            import pkg_resources
        except Exception as exc:
            logging.error(f"pkg_resources unavailable: {exc}")
            return {}

        conflicts: Dict[str, str] = {}
        for dist in pkg_resources.working_set:
            reqs = [str(r) for r in dist.requires()]
            for req in reqs:
                try:
                    pkg_resources.require(req)
                except pkg_resources.VersionConflict as conflict:
                    conflicts[dist.project_name] = str(conflict)
        return conflicts

    @staticmethod
    def build_dependency_graph() -> Dict[str, List[str]]:
        """Build a simple dependency graph of installed packages."""
        try:
            import pkg_resources
        except Exception as exc:
            logging.error(f"pkg_resources unavailable: {exc}")
            return {}

        graph: Dict[str, List[str]] = {}
        for dist in pkg_resources.working_set:
            graph[dist.project_name] = [str(r.project_name) for r in dist.requires()]
        return graph
