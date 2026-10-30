"""Set native thread limits before importing numeric libraries."""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

from .cli import main  # noqa: E402

raise SystemExit(main())

