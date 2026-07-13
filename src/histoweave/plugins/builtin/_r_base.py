"""Reusable base for containerised R/Bioconductor methods.

Each R method currently duplicates ~116 lines of h5ad I/O, subprocess
management, spatial-layer reconstruction, and error handling.  The
:class:`RContainerMethod` base collapses all of that into a single ``run``
template.  A new R method then only supplies its R script path and a short
``_build_r_args`` method — typically ~35 lines total.

Usage
-----
.. code-block:: python

    @register
    class MyRMethod(RContainerMethod):
        spec = MethodSpec(
            name="my_method",
            category=MethodCategory.INTEGRATION,
            version="0.1.0",
            summary="My R/Bioconductor wrapper.",
            params=(ParamSpec("lambda", "float", 0.5, "Smoothing parameter."),),
            assays=("visium",),
            wraps="Bioconductor::MyMethod",
            language="container",
        )
        r_script = "/usr/local/bin/histoweave-my-method.R"

        def _build_r_args(self, data):
            return [f"lambda={self.params['lambda']}"]

Maturity
--------
This is an **internal implementation detail** of the builtin plugin package.
Third-party plugins should subclass :class:`~histoweave.plugins.interfaces.Method`
directly or copy this pattern.
"""

from __future__ import annotations

import abc
import subprocess
import tempfile
from pathlib import Path

from ...data import SpatialTable
from ..interfaces import Method


class RContainerMethod(Method, abc.ABC):
    """Base for methods that shell out to a containerised R/Bioconductor script.

    Subclasses **must** provide:
    * ``spec`` — a :class:`MethodSpec` with ``language="container"``.
    * ``r_script`` — path to the R script **inside the container image**
      (e.g. ``"/usr/local/bin/histoweave-banksy.R"``).
    * :meth:`_build_r_args` — returns CLI arguments forwarded to the R script.

    Subclasses **may** override:
    * :meth:`_find_r_script` — discovery logic (default checks the container
      path first, then the source tree).
    * :meth:`_validate_r_output` — post-process the result before finalization.
    """

    #: Path to the R script inside the container image.
    r_script: str = ""

    # --- template method ----------------------------------------------------

    def run(self, data: SpatialTable) -> SpatialTable:
        """Write h5ad → Rscript → read h5ad → restore spatial layers → finalize."""

        self._validate_input(data)
        data = data.copy()
        r_script = self._find_r_script()

        with tempfile.TemporaryDirectory() as tmp:
            input_h5ad = Path(tmp) / "input.h5ad"
            output_h5ad = Path(tmp) / "output.h5ad"

            data.to_anndata().write_h5ad(input_h5ad)

            r_args = self._build_r_args(data)
            cmd = ["Rscript", str(r_script), str(input_h5ad), str(output_h5ad)] + r_args

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except FileNotFoundError:
                raise RuntimeError(
                    "Rscript not found on PATH. Install R and the anndata package, "
                    "or run inside the histoweave-r container image."
                ) from None
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"R method '{self.spec.name}' failed (exit {exc.returncode}):\n"
                    f"STDERR:\n{exc.stderr}\nSTDOUT:\n{exc.stdout}"
                ) from exc

            result = SpatialTable.from_anndata(
                _import_anndata().read_h5ad(output_h5ad)
            )

            self._validate_r_output(result)

        # Carry over spatial layers (images/shapes) that the bridge drops.
        result.images = data.images
        result.shapes = data.shapes
        return self.finalize(result, step=self.spec.category.value)

    # --- override points ----------------------------------------------------

    def _validate_input(self, data: SpatialTable) -> None:
        """Validate input before script discovery or h5ad serialization.

        The default is a no-op. Subclasses should override this hook for
        inexpensive structural checks that must fail before bridge I/O begins.
        """

    @abc.abstractmethod
    def _build_r_args(self, data: SpatialTable) -> list[str]:
        """Return CLI arguments forwarded after the R script path.

        Called after the h5ad is written, so the argument list can reference
        computed quantities from ``data`` (e.g. spatial coordinates via
        ``data.obsm['spatial']``).
        """
        ...

    def _validate_r_output(self, data: SpatialTable) -> None:
        """Optional post-process hook — raise if the R output is invalid.

        The default is a no-op.  Override to check that expected columns or
        layer keys are present in the result.
        """

    def _find_r_script(self) -> Path:
        """Locate the R script: container path first, then source-tree fallback.

        The container path is the value of the class attribute ``r_script``.
        The source-tree fallback assumes the script lives at::

            workflows/containers/histoweave-r/<basename of r_script>
        """
        container_path = Path(self.r_script)
        if container_path.exists():
            return container_path

        source_path = (
            Path(__file__).resolve().parents[4]
            / "workflows"
            / "containers"
            / "histoweave-r"
            / Path(self.r_script).name
        )
        if source_path.exists():
            return source_path

        raise FileNotFoundError(
            f"Cannot find {Path(self.r_script).name} — "
            f"expected at {container_path} or {source_path}"
        )


def _import_anndata():
    """Lazy import guard — anndata is an optional dependency."""
    try:
        import anndata
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "anndata is required for the R bridge. "
            "Install with: pip install 'histoweave-spatial[spatial]'"
        ) from exc
    return anndata
